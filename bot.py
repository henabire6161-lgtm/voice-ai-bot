import os
import logging
import tempfile
import subprocess
import httpx

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from faster_whisper import WhisperModel

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY",       "YOUR_GROQ_API_KEY")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE",  "large-v2")

SUPPORTED_LANGUAGES = {
    "am": "🇪🇹 Amharic",
    "or": "🇪🇹 Oromiffa",
    "en": "🇬🇧 English",
    "ar": "🇸🇦 Arabic",
    "fr": "🇫🇷 French",
    "sw": "🌍 Swahili",
    "so": "🇸🇴 Somali",
    "ti": "🇪🇷 Tigrinya",
}

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ─── LOAD WHISPER MODEL ONCE ──────────────────────────────────────────────────
logger.info("Loading Whisper '%s' model…", WHISPER_MODEL_SIZE)
model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
logger.info("Whisper model ready ✓")


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def convert_ogg_to_wav(ogg_path: str) -> str:
    wav_path = ogg_path.replace(".ogg", ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return wav_path


def transcribe_audio(file_path: str) -> tuple[str, str]:
    """Auto-detect language and transcribe. Returns (transcript, lang_code)."""
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    transcript = " ".join(seg.text.strip() for seg in segments)
    logger.info("Detected: %s (%.0f%%)", info.language, info.language_probability * 100)
    return transcript.strip(), info.language


async def groq_ai_reply(transcript: str, lang_code: str) -> tuple[str, str]:
    """
    Send transcript to Groq (free Llama 3).
    Returns (english_translation, ai_reply_in_both_languages).
    """
    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())

    system_prompt = (
        "You are a helpful multilingual AI assistant. "
        "You understand Ethiopian languages including Amharic and Oromiffa. "
        "Always be friendly, concise, and culturally respectful."
    )

    user_prompt = f"""The user sent a voice message in {lang_name}.
Transcribed text: "{transcript}"

Please do the following:
1. If not English, translate it to English.
2. Write a helpful and friendly reply to what the user said or asked.
   - First write the reply in English
   - Then write the same reply in {lang_name}

Use this exact format:
TRANSLATION: <English translation, or "Already in English">
REPLY_EN: <Your reply in English>
REPLY_LOCAL: <Your reply in {lang_name}>"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        result = data["choices"][0]["message"]["content"]

    # Parse response
    translation = ""
    reply_en = ""
    reply_local = ""

    for line in result.split("\n"):
        if line.startswith("TRANSLATION:"):
            translation = line.replace("TRANSLATION:", "").strip()
        elif line.startswith("REPLY_EN:"):
            reply_en = line.replace("REPLY_EN:", "").strip()
        elif line.startswith("REPLY_LOCAL:"):
            reply_local = line.replace("REPLY_LOCAL:", "").strip()

    # Fallback
    if not reply_en:
        reply_en = result

    return translation, reply_en, reply_local


# ─── TELEGRAM HANDLERS ────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🎙️ *Voice AI Bot*\n\n"
        "Send me a voice message and I will:\n\n"
        "1️⃣ Transcribe your voice to text\n"
        "2️⃣ Translate it to English\n"
        "3️⃣ Reply with AI in your language\n\n"
        "🇪🇹 Supports Amharic, Oromiffa, Tigrinya, Somali, English, Arabic & more!\n\n"
        "Tap the 🎤 microphone and speak!",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    langs = "\n".join([f"  • {v}" for v in SUPPORTED_LANGUAGES.values()])
    await update.message.reply_text(
        f"🌍 *Supported Languages:*\n{langs}\n\n"
        "Speak naturally — language is detected automatically!\n\n"
        "*Commands:*\n"
        "/start — Welcome\n"
        "/help — This message\n"
        "/model — Show AI model info",
        parse_mode="Markdown",
    )


async def cmd_model(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"🎤 Speech model: `{WHISPER_MODEL_SIZE}` (faster-whisper)\n"
        f"🤖 AI model: `llama3-70b-8192` (Groq — free)",
        parse_mode="Markdown",
    )


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    voice = message.voice or message.audio
    if not voice:
        return

    status = await message.reply_text("🎙️ Transcribing your voice…")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ogg_path = os.path.join(tmp_dir, "voice.ogg")

        # Download
        tg_file = await context.bot.get_file(voice.file_id)
        await tg_file.download_to_drive(ogg_path)

        # Convert
        try:
            audio_path = convert_ogg_to_wav(ogg_path)
        except Exception as e:
            logger.warning("ffmpeg failed, using ogg: %s", e)
            audio_path = ogg_path

        # Transcribe
        try:
            transcript, lang_code = transcribe_audio(audio_path)
        except Exception as e:
            logger.error("Transcription error: %s", e)
            await status.edit_text("❌ Transcription failed. Please try again.")
            return

    if not transcript:
        await status.edit_text("🤷 No speech detected. Please try again.")
        return

    lang_name = SUPPORTED_LANGUAGES.get(lang_code, lang_code.upper())

    # Show transcript while AI is thinking
    await status.edit_text(
        f"📝 *Transcribed ({lang_name}):*\n{transcript}\n\n🤖 AI is thinking…",
        parse_mode="Markdown",
    )

    # Get AI reply from Groq
    try:
        translation, reply_en, reply_local = await groq_ai_reply(transcript, lang_code)
    except Exception as e:
        logger.error("Groq API error: %s", e)
        await status.edit_text(
            f"📝 *Transcribed ({lang_name}):*\n{transcript}\n\n"
            f"⚠️ AI reply failed — check your GROQ\\_API\\_KEY.",
            parse_mode="Markdown",
        )
        return

    # Build final response
    parts = [f"📝 *Transcribed ({lang_name}):*\n{transcript}"]

    if translation and "already in english" not in translation.lower():
        parts.append(f"🌐 *English Translation:*\n{translation}")

    if reply_en:
        parts.append(f"🤖 *AI Reply (English):*\n{reply_en}")

    if reply_local and lang_code != "en":
        parts.append(f"💬 *AI Reply ({lang_name}):*\n{reply_local}")

    await status.edit_text("\n\n".join(parts), parse_mode="Markdown")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages with AI reply."""
    user_text = update.message.text
    status = await update.message.reply_text("🤖 Thinking…")

    try:
        _, reply_en, _ = await groq_ai_reply(user_text, "en")
        await status.edit_text(f"🤖 *AI Reply:*\n\n{reply_en}", parse_mode="Markdown")
    except Exception as e:
        logger.error("Groq error on text: %s", e)
        await status.edit_text("❌ AI reply failed. Please try again.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))
    app.add_handler(CommandHandler("model", cmd_model))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot is running… Press Ctrl+C to stop.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
