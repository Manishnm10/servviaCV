"""
Kannada Speech Recognition Post-Processor
Current Date and Time (UTC): 2025-11-25 06:28:15
Current User: lil-choco

Fixes common STT errors for Kannada medical phrases
"""
from difflib import SequenceMatcher
import re

# Common Kannada medical phrases and their STT mistakes
KANNADA_MEDICAL_CORRECTIONS = {
    # Fever related
    "nanige jwara ide": ["ke dwara idhar", "nani ge jwara ide", "na nige jwara ide", "ke jvara ide"],
    "jwara ide": ["jvara ide", "jwara hai", "jvar ide", "fever ide"],
    "jwara bantide": ["jwara ban tide", "jvara bantide"],
    
    # Headache
    "nanige thalenovu ide": ["ke tale novu ide", "nani ge tale novu", "ke talen ovu ide"],
    "thalenovu": ["tale novu", "thale novu", "talen ovu"],
    
    # Cough
    "nanige kemmu ide": ["ke kemmu ide", "nani ge kemmu ide", "ke kemu ide"],
    "kemmu": ["kemu", "kammu", "cough"],
    
    # Body pain
    "nanige shareera novu ide": ["ke sharira novu ide", "nani ge sharira novu"],
    "shareera novu": ["sharira novu", "shareera nobu", "body pain"],
    
    # Cold
    "nanige sardi ide": ["ke sardi ide", "nani ge sardi ide"],
    "sardi": ["sardi", "sardi hai"],
    
    # Stomach pain
    "nanige hotte novu ide": ["ke hote novu ide", "nani ge hotte novu"],
    "hotte novu": ["hote novu", "stomach pain"],
}


def correct_kannada_medical_phrase(transcribed_text: str) -> tuple:
    """
    Correct common Kannada medical phrase mistakes from STT
    
    Args:
        transcribed_text: Text from speech recognition
        
    Returns:
        Tuple of (corrected_text, confidence, was_corrected)
    """
    text_lower = transcribed_text.lower().strip()
    
    best_match = None
    best_similarity = 0.0
    
    # Check each known phrase
    for correct_phrase, mistake_variations in KANNADA_MEDICAL_CORRECTIONS.items():
        # Check exact match with correct phrase
        if text_lower == correct_phrase:
            return correct_phrase, 1.0, False  # Already correct
        
        # Check if it matches any known mistake
        for mistake in mistake_variations:
            similarity = SequenceMatcher(None, text_lower, mistake).ratio()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = correct_phrase
    
    # If high similarity found, return correction
    if best_similarity > 0.6:  # 60% similarity threshold
        return best_match, best_similarity, True
    
    # No correction found
    return transcribed_text, 0.0, False


def enhance_kannada_transcription(transcribed_text: str, language: str = 'kn') -> dict:
    """
    Enhance Kannada transcription with corrections
    
    Args:
        transcribed_text: Original transcription
        language: Language code
        
    Returns:
        Enhanced result with corrections
    """
    if language != 'kn':
        # Not Kannada, return as-is
        return {
            'text': transcribed_text,
            'corrected': False,
            'confidence': 0.75,
            'original': transcribed_text
        }
    
    # Try to correct
    corrected_text, similarity, was_corrected = correct_kannada_medical_phrase(transcribed_text)
    
    if was_corrected:
        return {
            'text': corrected_text,
            'corrected': True,
            'confidence': similarity,
            'original': transcribed_text,
            'correction_note': f'Auto-corrected from "{transcribed_text}" (similarity: {similarity:.1%})'
        }
    else:
        return {
            'text': transcribed_text,
            'corrected': False,
            'confidence': 0.75,
            'original': transcribed_text
        }