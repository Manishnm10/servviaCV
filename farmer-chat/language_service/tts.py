"""
Text-to-Speech (TTS) Service
Current Date and Time (UTC): 2025-11-25 05:23:15
Current User: lil-choco

Features:
- Google Cloud Text-to-Speech integration
- Azure TTS fallback support
- Multi-language support (Kannada, Hindi, English, etc.)
- Proper error handling and logging
"""
import asyncio
import aiohttp
import logging
import uuid
import os
from google.cloud import texttospeech
from google.oauth2 import service_account

from common.constants import Constants
from common.utils import clean_text
from language_service.utils import get_language_by_code
from django_core.config import Config

logger = logging.getLogger(__name__)

# ============================================================
# GOOGLE CLOUD TTS CREDENTIALS
# Current Date and Time (UTC): 2025-11-25 05:23:15
# Current User: lil-choco
# ============================================================

try:
    # ✅ Try to load from settings first
    from django.conf import settings
    
    gcp_cred_path = getattr(settings, 'GCP_TRANSLATION_CREDENTIALS_PATH', 
                            r"C:\Users\cools\Downloads\servvia-google-credentials.json")
    
    if os.path.exists(gcp_cred_path):
        credentials = service_account.Credentials.from_service_account_file(gcp_cred_path)
        logger.info(f"✅ TTS credentials loaded: {gcp_cred_path}")
    elif hasattr(Config, 'GOOGLE_APPLICATION_CREDENTIALS') and os.path.exists(Config.GOOGLE_APPLICATION_CREDENTIALS):
        credentials = service_account.Credentials.from_service_account_file(Config.GOOGLE_APPLICATION_CREDENTIALS)
        logger.info(f"✅ TTS credentials loaded from Config: {Config.GOOGLE_APPLICATION_CREDENTIALS}")
    else:
        logger.error(f"❌ TTS credentials file not found at: {gcp_cred_path}")
        credentials = None
        
except Exception as e:
    logger.error(f"❌ Could not load TTS credentials: {e}")
    credentials = None


async def synthesize_speech_azure(text_to_synthesize, language_code, aiohttp_session):
    """
    Synthesise speech using Azure TTS model.
    Current Date and Time (UTC): 2025-11-25 05:23:15
    Current User: lil-choco
    
    Azure TTS Docs: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/
    
    Args:
        text_to_synthesize: Text to convert to speech
        language_code: Language code (e.g., 'en-US', 'kn-IN')
        aiohttp_session: Async HTTP session
        
    Returns:
        Audio content as bytes or None if failed
    """
    audio_content = None

    # Check if Azure credentials are available
    if not hasattr(Config, 'AZURE_SERVICE_REGION') or not hasattr(Config, 'AZURE_SUBSCRIPTION_KEY'):
        logger.warning("⚠️ Azure TTS credentials not configured")
        return None

    # Use Azure for Speech synthesis
    url = f"https://{Config.AZURE_SERVICE_REGION}.tts.speech.microsoft.com/cognitiveservices/v1"
    headers = {
        "Ocp-Apim-Subscription-Key": Config.AZURE_SUBSCRIPTION_KEY,
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "ogg-48khz-16bit-mono-opus",
    }

    # Select appropriate voice based on language
    AZURE_VOICE = "en-GB-SoniaNeural"  # Default
    
    if language_code == "en-KE":
        AZURE_VOICE = "en-KE-AsiliaNeural"
    elif language_code == "sw-KE":
        AZURE_VOICE = "sw-KE-ZuriNeural"
    elif language_code == "en-NG":
        AZURE_VOICE = "en-NG-EzinneNeural"
    elif language_code == "en-US":
        AZURE_VOICE = "en-US-JennyNeural"
    elif language_code == "hi-IN":
        AZURE_VOICE = "hi-IN-SwaraNeural"

    # The body of the request
    body = f"""
    <speak version='1.0' xml:lang='{language_code}'>
        <voice xml:lang='{language_code}' xml:gender='Female' name='{AZURE_VOICE}'>
            {text_to_synthesize}
        </voice>
    </speak>
    """
    
    try:
        async with aiohttp_session.post(url, data=body, headers=headers) as response:
            if response.status == 200:
                audio_content = await response.read()
                logger.info(f"✅ Azure TTS synthesis successful")
            else:
                logger.error(f"❌ Azure TTS failed with status: {response.status}")
                
    except Exception as e:
        logger.error(f"❌ Azure TTS error: {e}", exc_info=True)

    return audio_content


async def synthesize_speech(
    input_text: str,
    input_language: str,
    id_string: str = None,
    aiohttp_session=None,
    audio_encoding_format=texttospeech.AudioEncoding.OGG_OPUS,
    sample_rate_hertz=48000,
) -> str:
    """
    Synthesise speech using Google TTS with fallback to Azure
    Current Date and Time (UTC): 2025-11-25 05:23:15
    Current User: lil-choco
    
    Google TTS Docs: https://cloud.google.com/text-to-speech/docs/
    
    Args:
        input_text: Text to convert to speech
        input_language: Language code (e.g., 'en', 'kn', 'hi')
        id_string: Unique identifier for the audio file
        aiohttp_session: Async HTTP session for Azure fallback
        audio_encoding_format: Audio format (OGG_OPUS or MP3)
        sample_rate_hertz: Sample rate (default: 48000)
        
    Returns:
        Path to generated audio file or None if failed
    """
    # Generate unique file ID
    id_string = uuid.uuid4() if not id_string else id_string
    file_name = f"response_{id_string}.{Constants.OGG}"
    
    # Clean and validate input text
    input_text = clean_text(input_text)
    
    if not input_text or input_text.strip() == "":
        logger.error("❌ Empty text provided for speech synthesis")
        return None
    
    # Limit text length (Google TTS has 5000 char limit)
    if len(input_text) > 5000:
        logger.warning(f"⚠️ Text too long ({len(input_text)} chars), truncating to 5000")
        input_text = input_text[:4997] + "..."
    
    logger.info(f"🔊 Synthesizing speech: '{input_text[:50]}...' in language: {input_language}")
    
    synthesis_input = texttospeech.SynthesisInput(text=input_text)
    
    # Default language code
    language_code = "en-US"
    
    # Extract base language code (kn-IN → kn)
    input_language_base = input_language.split("-")[0] if "-" in input_language else input_language

    # Determine audio format
    if audio_encoding_format and str(audio_encoding_format).lower() == Constants.MP3:
        audio_encoding_format = texttospeech.AudioEncoding.MP3
        file_name = f"response_{id_string}.{Constants.MP3}"
    else:
        audio_encoding_format = texttospeech.AudioEncoding.OGG_OPUS

    sample_rate_hertz = sample_rate_hertz if sample_rate_hertz else 48000

    try:
        # Try to get language from database
        language = get_language_by_code(input_language_base)
        
        if language:
            language_code = language.get("bcp_code")
            logger.info(f"✅ Using BCP code: {language_code} for language: {input_language_base}")
        else:
            # Fallback: Map common language codes to BCP-47 codes
            language_map = {
                "en": "en-US",
                "hi": "hi-IN",
                "kn": "kn-IN",  # Kannada
                "ta": "ta-IN",  # Tamil
                "te": "te-IN",  # Telugu
                "ml": "ml-IN",  # Malayalam
                "bn": "bn-IN",  # Bengali
                "mr": "mr-IN",  # Marathi
                "gu": "gu-IN",  # Gujarati
                "pa": "pa-IN",  # Punjabi
                "es": "es-ES",  # Spanish
                "fr": "fr-FR",  # French
                "de": "de-DE",  # German
                "zh": "zh-CN",  # Chinese
                "ja": "ja-JP",  # Japanese
                "ko": "ko-KR",  # Korean
            }
            language_code = language_map.get(input_language_base, "en-US")
            logger.warning(f"⚠️ Language {input_language_base} not in database, using: {language_code}")

        # Check if Google TTS credentials are available
        if not credentials:
            logger.error("❌ Google TTS credentials not available")
            
            # Try Azure fallback if available
            if aiohttp_session:
                logger.info("🔄 Attempting Azure TTS fallback...")
                audio_content = await synthesize_speech_azure(input_text, language_code, aiohttp_session)
                
                if audio_content:
                    with open(file_name, "wb") as out:
                        out.write(audio_content)
                    logger.info(f"✅ Azure TTS successful: {file_name}")
                    return file_name
            
            return None

        # Use Google TTS for speech synthesis
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=audio_encoding_format, 
            sample_rate_hertz=sample_rate_hertz
        )
        
        text_to_speech_client = texttospeech.TextToSpeechClient(credentials=credentials)

        try:
            logger.info(f"📡 Calling Google TTS API...")
            
            response = await asyncio.to_thread(
                text_to_speech_client.synthesize_speech,
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            
            audio_content = response.audio_content
            
            if not audio_content:
                logger.error("❌ Google TTS returned empty audio content")
                return None

            # Write audio to file
            with open(file_name, "wb") as out:
                out.write(audio_content)
            
            # Verify file was created and has content
            if os.path.exists(file_name):
                file_size = os.path.getsize(file_name)
                logger.info(f"✅ Successfully wrote voice response to file: {file_name} ({file_size} bytes)")
            else:
                logger.error(f"❌ File not created: {file_name}")
                return None

        except Exception as tts_error:
            logger.error(f"❌ Google TTS API error: {str(tts_error)}", exc_info=True)
            
            # Try Azure fallback
            if aiohttp_session:
                logger.info("🔄 Attempting Azure TTS fallback after Google TTS failure...")
                audio_content = await synthesize_speech_azure(input_text, language_code, aiohttp_session)
                
                if audio_content:
                    with open(file_name, "wb") as out:
                        out.write(audio_content)
                    logger.info(f"✅ Azure TTS fallback successful: {file_name}")
                    return file_name
            
            return None

    except Exception as e:
        logger.error(f"❌ TTS Error: {e}", exc_info=True)
        return None

    return file_name


def get_supported_languages():
    """
    Get list of supported languages for TTS
    
    Returns:
        dict: Mapping of language codes to language names
    """
    return {
        "en-US": "English (United States)",
        "en-GB": "English (United Kingdom)",
        "en-IN": "English (India)",
        "hi-IN": "Hindi (India)",
        "kn-IN": "Kannada (India)",
        "ta-IN": "Tamil (India)",
        "te-IN": "Telugu (India)",
        "ml-IN": "Malayalam (India)",
        "bn-IN": "Bengali (India)",
        "mr-IN": "Marathi (India)",
        "gu-IN": "Gujarati (India)",
        "pa-IN": "Punjabi (India)",
        "es-ES": "Spanish (Spain)",
        "fr-FR": "French (France)",
        "de-DE": "German (Germany)",
        "zh-CN": "Chinese (Simplified)",
        "ja-JP": "Japanese (Japan)",
        "ko-KR": "Korean (South Korea)",
    }