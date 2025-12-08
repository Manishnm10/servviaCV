from rest_framework. decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from . models import SkinAnalysis
from .disease_detector import SkinDiseaseDetector
import logging

logger = logging.getLogger(__name__)
detector = SkinDiseaseDetector()

@api_view(['POST'])
def analyze_skin_image(request):
    """Endpoint to upload and analyze skin images"""
    try:
        email = request.data.get('email_id')
        if not email:
            return Response({'error': 'Email is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'Image file is required'}, status=status.HTTP_400_BAD_REQUEST)
        
        # Predict disease
        disease, confidence = detector. predict(image_file)
        
        if not disease:
            return Response({'error': 'Unable to analyze image'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        # Get recommendations
        recommendations = detector.get_recommendations(disease)
        
        # Save analysis
        analysis = SkinAnalysis.objects.create(
            email_id=email,
            image=image_file,
            diagnosis=disease,
            confidence_score=confidence,
            recommendations=recommendations
        )
        
        return Response({
            'success': True,
            'diagnosis': disease,
            'confidence': round(confidence * 100, 2),
            'recommendations': recommendations,
            'analysis_id': analysis.id
        })
        
    except Exception as e:
        logger.error(f"Skin analysis error: {e}", exc_info=True)
        return Response({'error': str(e)}, status=status. HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def get_skin_analysis_history(request):
    """Get user's skin analysis history"""
    try:
        email = request.query_params.get('email_id')
        analyses = SkinAnalysis.objects. filter(email_id=email)[:10]
        
        results = [{
            'id': a.id,
            'diagnosis': a.diagnosis,
            'confidence': a.confidence_score,
            'date': a.created_at.strftime('%Y-%m-%d %H:%M'),
        } for a in analyses]
        
        return Response({'success': True, 'history': results})
        
    except Exception as e:
        logger. error(f"History retrieval error: {e}")
        return Response({'error': str(e)}, status=status. HTTP_500_INTERNAL_SERVER_ERROR)