"""
Lab Report Analyzer using Gemini Flash 2.0
Extracts and analyzes lab reports with medical-grade accuracy
"""

import google.generativeai as genai
import json
import logging
from PIL import Image
import io
import os
from django_core.config import ENV_CONFIG

logger = logging.getLogger(__name__)


class LabReportAnalyzer:
    """
    Gemini-powered lab report analyzer
    Extracts text, identifies abnormal values, generates summaries
    """
    
    def __init__(self):
        """Initialize Gemini API"""
        api_key = (
            ENV_CONFIG.get('GOOGLE_API_KEY') or 
            ENV_CONFIG.get('GEMINI_API_KEY') or 
            os.getenv('GOOGLE_API_KEY') or 
            os.getenv('GEMINI_API_KEY')
        )
        
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        logger.info("✅ Lab Report Analyzer initialized with Gemini 2.0 Flash")
    
    def extract_text_from_pdf(self, report_file):
        """
        Extract text from PDF/image using Gemini Vision
        
        Args:
            report_file: Django UploadedFile (PDF or image)
            
        Returns:
            str: Extracted text
        """
        try:
            # Read file
            file_data = report_file.read()
            
            # Try to open as image first
            try:
                image = Image.open(io.BytesIO(file_data))
                
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Extract text using Gemini Vision
                prompt = """Extract ALL text from this lab report image.  
                
Include:
- Test names
- Values
- Units
- Reference ranges
- Dates
- Patient information (if visible)

Return the extracted text as-is, preserving structure."""
                
                response = self. model.generate_content([prompt, image])
                extracted_text = response.text.strip()
                
                logger.info(f"✅ Extracted {len(extracted_text)} characters from image")
                return extracted_text
                
            except Exception as img_error:
                # If not an image, try as PDF
                logger.warning(f"Not an image, treating as PDF: {img_error}")
                
                # For PDFs, Gemini can still read them if we save temporarily
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    tmp.write(file_data)
                    tmp_path = tmp.name
                
                # Note: Gemini 2.0 Flash supports PDF input
                prompt = "Extract all text from this lab report PDF, preserving structure."
                
                # Upload file to Gemini
                uploaded_file = genai.upload_file(tmp_path)
                response = self.model.generate_content([prompt, uploaded_file])
                
                extracted_text = response.text.strip()
                
                # Cleanup
                os.unlink(tmp_path)
                
                logger.info(f"✅ Extracted {len(extracted_text)} characters from PDF")
                return extracted_text
                
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}", exc_info=True)
            return ""
    
    def summarize_report(self, extracted_text, email_id):
        """
        Analyze lab report and generate beautifully formatted summary
        
        Args:
            extracted_text: Raw text from report
            email_id: User's email
            
        Returns:
            dict: Structured analysis with formatted summary
        """
        try:
            if not extracted_text:
                return {
                    'success': False,
                    'error': 'No text extracted from report'
                }
            
            # Create enhanced analysis prompt with formatting instructions
            prompt = f"""You are a medical AI assistant analyzing a lab report.  

**EXTRACTED TEXT:**
{extracted_text}

**TASK:** Analyze this lab report and provide a comprehensive, patient-friendly summary with BEAUTIFUL FORMATTING.

**RESPONSE FORMAT (JSON):**
{{
  "test_type": "Complete Blood Count" or "Lipid Panel" or "Comprehensive Metabolic Panel" etc.,
  "report_date": "2025-12-03" (extract from report if visible),
  "parameters": [
    {{
      "name": "Hemoglobin",
      "value": "10.2",
      "unit": "g/dL",
      "normal_range": "12-16 g/dL",
      "status": "Low" or "Normal" or "High",
      "severity": "Mild" or "Moderate" or "Severe" or "Normal",
      "icon": "🔴",
      "explanation": "Your hemoglobin is slightly low, which may indicate mild anemia."
    }}
  ],
  "abnormal_count": 2,
  "formatted_summary": "**📋 Overall Summary**\\n\\nYour blood counts are mostly normal except for signs of **mild anemia**, specifically microcytic anemia.  \\n\\n**📊 What Each Result Means**\\n\\n### 1.  🔴 Hemoglobin: 12.8 g/dL — Low\\n- **What it is:** Hemoglobin is the protein in red cells that carries oxygen\\n- **Your value:** 12.8 g/dL (Normal: 13. 5-17.5 g/dL)\\n- **What it means:** This is mild anemia\\n\\n### 2. 🟢 WBC: 7.5 — Normal\\n- **What it is:** White blood cells fight infections\\n- **Your value:** Well within normal range\\n- **What it means:** No signs of infection\\n\\n**💡 What Does This Mean?**\\n\\nYou have **mild microcytic anemia**, meaning your red blood cells are slightly smaller and carry less oxygen.  \\n\\n**🔍 Common Causes:**\\n- ✅ Iron deficiency (most common)\\n- ✅ Chronic low-grade blood loss\\n\\n**🩺 What You Might Feel:**\\n- Tiredness or low energy\\n- Shortness of breath on exertion\\n- Pale skin\\n\\n**🎯 Next Steps:**\\n1. Doctor may order iron studies\\n2. Review diet and bleeding history\\n3. Iron supplementation if confirmed",
  "critical_flags": [],
  "recommendations": [
    "🥬 Eat iron-rich foods like spinach, lentils, and lean red meat",
    "🍊 Take vitamin C with iron-rich meals to improve absorption",
    "👨‍⚕️ Consult your doctor if you experience fatigue or dizziness"
  ],
  "follow_up_needed": true,
  "overall_status": "Mostly normal with minor concerns",
  "visual_indicators": {{
    "normal_count": 5,
    "abnormal_count": 2,
    "critical_count": 0
  }}
}}

**FORMATTING RULES FOR formatted_summary:**
1. Use emojis: 🔴 (Low), 🟢 (Normal), 🟠 (High), ⚠️ (Critical)
2. Use **bold** for emphasis
3. Use ### for headers
4. Use numbered lists for action items
5. Use bullet points (- ) for explanations
6. Use \\n\\n for line breaks between sections
7.  Keep paragraphs short and scannable

**STRUCTURE:**

**📋 Overall Summary**
[2-3 sentence overview]

**📊 What Each Result Means**

### 1. [Icon] Parameter: Value — Status
- **What it is:** [Simple explanation]
- **Your value:** [Value vs normal]
- **What it means:** [Implication]

[Repeat for ALL parameters]

**💡 What Does This Mean?**
[Overall interpretation]

**🔍 Common Causes:**
- ✅ [Cause 1]
- ✅ [Cause 2]

**🩺 What You Might Feel:**
- [Symptom 1]
- [Symptom 2]

**🎯 Next Steps:**
1. [Action 1]
2. [Action 2]

**IMPORTANT:**
- Be medically accurate
- Patient-friendly language
- Clear visual hierarchy
- Make it scannable

Analyze the report now:"""
            
            logger.info("🔬 Analyzing lab report with Gemini...")
            
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            analysis = self._parse_json_response(response_text)
            
            if analysis:
                logger.info(f"✅ Analysis complete: {analysis. get('test_type')} - {analysis.get('abnormal_count')} abnormal values")
                
                # Return structured data with formatted summary
                return {
                    'success': True,
                    'analysis': analysis,
                    'summary': analysis.get('formatted_summary', analysis.get('summary', '')),  # Use formatted version
                    'abnormal_values': analysis.get('parameters', []),
                    'recommendations': analysis.get('recommendations', []),
                    'critical_flags': analysis.get('critical_flags', []),
                    'visual_indicators': analysis.get('visual_indicators', {})
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to parse analysis response'
                }
                
        except Exception as e:
            logger.error(f"❌ Report analysis failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _parse_json_response(self, response_text):
        """Parse Gemini's JSON response"""
        try:
            # Extract JSON from markdown code blocks if present
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1]. split('```')[0].strip()
            elif '{' in response_text and '}' in response_text:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                json_text = response_text[start:end]
            else:
                json_text = response_text
            
            return json.loads(json_text)
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing error: {e}")
            logger.error(f"Response: {response_text[:500]}")
            return None
    
    def generate_embedding_text(self, analysis):
        """
        Generate text representation for vector database embedding
        
        Args:
            analysis: Structured analysis dict
            
        Returns:
            str: Text for embedding
        """
        if not analysis or not analysis.get('success'):
            return ""
        
        data = analysis.get('analysis', {})
        
        # Create searchable text
        embedding_text = f"""
Lab Report: {data.get('test_type', 'Unknown')}
Date: {data.get('report_date', 'Unknown')}

Test Results:
"""
        
        for param in data.get('parameters', []):
            embedding_text += f"\n- {param.get('name')}: {param.get('value')} {param.get('unit')} ({param.get('status')})"
            if param.get('status') != 'Normal':
                embedding_text += f" - {param.get('explanation')}"
        
        embedding_text += f"\n\nSummary: {data.get('summary', '')}"
        
        return embedding_text. strip()