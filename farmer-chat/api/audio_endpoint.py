"""
Audio Endpoint for Voice Input
Current Date and Time (UTC): 2025-11-25 06:08:30
Current User: lil-choco
"""
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from language_service.kannada_corrector import enhance_kannada_transcription
import json
import logging
import tempfile
import os
import base64

logger = logging.getLogger(__name__)

# Google Cloud Speech-to-Text
try:
    from google.cloud import speech
    from google.oauth2 import service_account
    from django.conf import settings
    
    gcp_cred_path = getattr(settings, 'GCP_TRANSLATION_CREDENTIALS_PATH', 
                            r"C:\Users\cools\Downloads\servvia-google-credentials.json")
    
    if os.path.exists(gcp_cred_path):
        speech_credentials = service_account.Credentials.from_service_account_file(gcp_cred_path)
        GOOGLE_SPEECH_AVAILABLE = True
        print("✅ Google Cloud Speech-to-Text available")
    else:
        GOOGLE_SPEECH_AVAILABLE = False
        speech_credentials = None
        print("⚠️ Google Cloud Speech credentials not found")
        
except ImportError:
    GOOGLE_SPEECH_AVAILABLE = False
    speech_credentials = None
    print("❌ Google Cloud Speech not available - install with: pip install google-cloud-speech")

# Basic Speech Recognition (fallback)
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
    print("✅ Basic Speech Recognition available (fallback)")
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️ Basic Speech Recognition not available")

# Audio conversion
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
    print("✅ Pydub available for audio conversion")
except ImportError:
    PYDUB_AVAILABLE = False
    print("⚠️ Pydub not available")


@csrf_exempt
@require_http_methods(["POST"])
def transcribe_audio(request):
    """Transcribe audio using Google Cloud Speech-to-Text with fallback"""
    try:
        data = json.loads(request.body)
        audio_base64 = data.get('audio')
        user_language = data.get('language', 'en')
        
        if not audio_base64:
            return JsonResponse({
                "success": False,
                "error": "No audio data provided",
                "heard_input_query": "",
                "confidence_score": 0
            }, status=400)
        
        # Decode base64 audio
        try:
            if ',' in audio_base64:
                audio_data = base64.b64decode(audio_base64.split(',')[1])
            else:
                audio_data = base64.b64decode(audio_base64)
        except Exception as decode_error:
            logger.error(f"Base64 decode error: {decode_error}")
            return JsonResponse({
                "success": False,
                "error": "Invalid audio data",
                "heard_input_query": "",
                "confidence_score": 0
            }, status=400)
        
        # Save to temp file
        temp_audio_path = tempfile.mktemp(suffix='.webm')
        with open(temp_audio_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Audio received: {len(audio_data)} bytes, language: {user_language}")
        
        # Try Google Cloud STT first
        if GOOGLE_SPEECH_AVAILABLE and speech_credentials:
            try:
                logger.info("Attempting Google Cloud Speech-to-Text...")
                transcription, confidence = transcribe_with_google_cloud(temp_audio_path, user_language)
                
                if transcription:
                    logger.info(f"📝 Raw Google Cloud STT: '{transcription}' (confidence: {confidence:.2%})")
    
                    # ✅ ENHANCE: Apply Kannada corrections if needed
                    if user_language == 'kn':
                        enhanced = enhance_kannada_transcription(transcription, user_language)
        
                        if enhanced['corrected']:
                            logger.info(f"✅ Kannada correction applied: '{enhanced['text']}' (from '{transcription}')")
                            transcription = enhanced['text']
                            confidence = max(confidence, enhanced['confidence'])  # Use higher confidence
            
                            # Add note about correction
                            correction_note = enhanced.get('correction_note', '')
                        else:
                            logger.info(f"✅ No correction needed: '{transcription}'")
                            correction_note = ''
    
                    logger.info(f"✅ Final transcription: '{transcription}' (confidence: {confidence:.2%})")

                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                    
                    return JsonResponse({
                        "success": True,
                        "heard_input_query": transcription,
                        "confidence_score": confidence,
                        "method": "Google Cloud Speech-to-Text",
                        "language": user_language
                    })
                    
            except Exception as google_error:
                logger.warning(f"Google Cloud STT failed: {google_error}, trying fallback...")
        
        # Fallback to basic SR
        if SPEECH_RECOGNITION_AVAILABLE and PYDUB_AVAILABLE:
            try:
                logger.info("Attempting Basic Speech Recognition...")
                transcription, confidence = transcribe_with_basic_sr(temp_audio_path, user_language)
                
                if transcription:
                    logger.info(f"Basic SR: '{transcription}' (confidence: {confidence:.2%})")
                    if os.path.exists(temp_audio_path):
                        os.remove(temp_audio_path)
                    
                    return JsonResponse({
                        "success": True,
                        "heard_input_query": transcription,
                        "confidence_score": confidence,
                        "method": "Basic Speech Recognition",
                        "language": user_language
                    })
                    
            except Exception as sr_error:
                logger.error(f"Basic SR failed: {sr_error}")
        
        # Clean up
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return JsonResponse({
            "success": False,
            "error": "Could not transcribe audio",
            "heard_input_query": "",
            "confidence_score": 0
        }, status=500)
        
    except Exception as error:
        logger.error(f"Audio transcription error: {error}", exc_info=True)
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        
        return JsonResponse({
            "success": False,
            "error": str(error),
            "heard_input_query": "",
            "confidence_score": 0
        }, status=500)


def transcribe_with_google_cloud(audio_path, language='en'):
    """Transcribe with Google Cloud Speech-to-Text"""
    language_map = {
        'en': 'en-US',
        'kn': 'kn-IN',
        'hi': 'hi-IN',
        'ta': 'ta-IN',
        'te': 'te-IN',
        'ml': 'ml-IN',
        'bn': 'bn-IN',
        'mr': 'mr-IN',
    }
    
    gcloud_language = language_map.get(language, 'en-US')
    
    # Convert audio
    if PYDUB_AVAILABLE:
        try:
            audio = AudioSegment.from_file(audio_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            temp_wav_path = tempfile.mktemp(suffix='.wav')
            audio.export(temp_wav_path, format='wav')
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return None, 0.0
    else:
        temp_wav_path = audio_path
    
    try:
        client = speech.SpeechClient(credentials=speech_credentials)
        
        with open(temp_wav_path, 'rb') as audio_file:
            audio_content = audio_file.read()
        
        audio = speech.RecognitionAudio(content=audio_content)
        
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code=gcloud_language,
            enable_automatic_punctuation=True,
            model='medical_conversation' if language == 'en' else 'default',
            use_enhanced=True,
        )
        
        response = client.recognize(config=config, audio=audio)
        
        if temp_wav_path != audio_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        
        if response.results:
            result = response.results[0]
            if result.alternatives:
                transcription = result.alternatives[0].transcript
                confidence = result.alternatives[0].confidence
                return transcription, confidence
        
        return None, 0.0
        
    except Exception as e:
        logger.error(f"Google Cloud STT error: {e}", exc_info=True)
        if temp_wav_path != audio_path and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        raise


def transcribe_with_basic_sr(audio_path, language='en'):
    """Transcribe with basic speech recognition"""
    try:
        audio = AudioSegment.from_file(audio_path)
        temp_wav_path = tempfile.mktemp(suffix='.wav')
        audio.export(temp_wav_path, format='wav')
    except Exception as e:
        logger.error(f"Audio conversion failed: {e}")
        return None, 0.0
    
    try:
        recognizer = sr.Recognizer()
        
        with sr.AudioFile(temp_wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
        
        transcription = recognizer.recognize_google(audio_data, language=language)
        confidence = 0.75
        
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        
        return transcription, confidence
        
    except sr.UnknownValueError:
        logger.warning("Could not understand audio")
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        return None, 0.0
        
    except Exception as e:
        logger.error(f"Basic SR error: {e}")
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        raise