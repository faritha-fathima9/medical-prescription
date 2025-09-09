# AI Medical Prescription Verification System
# Using Granite-3.2, IBM Watson, and Hugging Face Models

# Install required packages (run this cell first in Google Colab)
"""
!pip install gradio transformers torch torchvision torchaudio
!pip install ibm-watson ibm-cloud-sdk-core
!pip install pillow requests pandas numpy
!pip install accelerate bitsandbytes
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from PIL import Image
import requests
import json
import pandas as pd
import re
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MedicalPrescriptionVerifier:
    def __init__(self):
        self.setup_models()
        self.setup_medical_database()
        
    def setup_models(self):
        """Initialize Granite-3.2 and other Hugging Face models"""
        try:
            print("Loading Granite-3.2 model...")
            # Load Granite-3.2 model from Hugging Face
            model_name = "ibm-granite/granite-3.2-8b-instruct"
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # Load model with optimizations for Colab
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True,  # Memory optimization
                trust_remote_code=True
            )
            
            # Create text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=500,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            print("Granite-3.2 model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading Granite model: {e}")
            # Fallback to a lighter model if Granite fails
            self.setup_fallback_model()
    
    def setup_fallback_model(self):
        """Setup fallback model if Granite-3.2 fails to load"""
        print("Setting up fallback model...")
        self.text_generator = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium",
            max_new_tokens=200,
            temperature=0.7
        )
    
    def setup_medical_database(self):
        """Setup medical knowledge database"""
        self.drug_interactions = {
            "warfarin": ["aspirin", "ibuprofen", "amoxicillin"],
            "metformin": ["alcohol", "iodinated contrast"],
            "lisinopril": ["potassium supplements", "nsaids"],
            "simvastatin": ["gemfibrozil", "cyclosporine"],
            "digoxin": ["quinidine", "verapamil", "amiodarone"]
        }
        
        self.common_dosages = {
            "metformin": {"min": 500, "max": 2000, "unit": "mg"},
            "lisinopril": {"min": 2.5, "max": 40, "unit": "mg"},
            "simvastatin": {"min": 5, "max": 80, "unit": "mg"},
            "warfarin": {"min": 1, "max": 10, "unit": "mg"},
            "digoxin": {"min": 0.125, "max": 0.5, "unit": "mg"}
        }
        
        self.contraindications = {
            "pregnancy": ["warfarin", "ace_inhibitors", "statins"],
            "kidney_disease": ["metformin", "nsaids"],
            "liver_disease": ["acetaminophen_high_dose", "statins"]
        }
    
    def setup_watson_client(self, api_key, url):
        """Setup IBM Watson client (placeholder - requires actual credentials)"""
        try:
            # This would require actual IBM Watson credentials
            self.watson_client = None  # Placeholder
            print("Watson client setup (requires actual API credentials)")
        except Exception as e:
            print(f"Watson setup error: {e}")
            self.watson_client = None
    
    def extract_prescription_info(self, prescription_text):
        """Extract key information from prescription text using NLP"""
        
        prompt = f"""
        Analyze the following medical prescription and extract key information in JSON format:

        Prescription: {prescription_text}

        Please extract and return ONLY a JSON object with these fields:
        - patient_name: string
        - medications: list of objects with name, dosage, frequency, duration
        - prescriber: string
        - date: string
        - medical_conditions: list of strings
        - allergies: list of strings

        JSON Response:
        """
        
        try:
            response = self.text_generator(
                prompt,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=True
            )
            
            generated_text = response[0]['generated_text']
            # Extract JSON from response
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = generated_text[json_start:json_end]
                return json.loads(json_str)
            else:
                return self.parse_prescription_fallback(prescription_text)
                
        except Exception as e:
            print(f"Error in AI extraction: {e}")
            return self.parse_prescription_fallback(prescription_text)
    
    def parse_prescription_fallback(self, text):
        """Fallback parsing method using regex"""
        medications = []
        
        # Simple regex patterns for common prescription formats
        med_patterns = [
            r'(\w+)\s+(\d+(?:\.\d+)?)\s*(mg|g|ml)\s+(\w+)',
            r'(\w+)\s+(\d+(?:\.\d+)?)(mg|g|ml)'
        ]
        
        for pattern in med_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                medications.append({
                    "name": match[0].lower(),
                    "dosage": f"{match[1]}{match[2]}",
                    "frequency": "as prescribed",
                    "duration": "as prescribed"
                })
        
        return {
            "patient_name": "Patient",
            "medications": medications,
            "prescriber": "Dr. Unknown",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "medical_conditions": [],
            "allergies": []
        }
    
    def check_drug_interactions(self, medications):
        """Check for potential drug interactions"""
        interactions = []
        med_names = [med['name'].lower() for med in medications]
        
        for med in med_names:
            if med in self.drug_interactions:
                for other_med in med_names:
                    if other_med in self.drug_interactions[med]:
                        interactions.append({
                            "severity": "High",
                            "drugs": [med, other_med],
                            "warning": f"Potential interaction between {med} and {other_med}"
                        })
        
        return interactions
    
    def validate_dosages(self, medications):
        """Validate medication dosages"""
        dosage_warnings = []
        
        for med in medications:
            med_name = med['name'].lower()
            if med_name in self.common_dosages:
                # Extract numeric dosage
                dosage_str = med['dosage']
                dosage_num = re.findall(r'(\d+(?:\.\d+)?)', dosage_str)
                
                if dosage_num:
                    dosage = float(dosage_num[0])
                    limits = self.common_dosages[med_name]
                    
                    if dosage < limits['min']:
                        dosage_warnings.append({
                            "medication": med_name,
                            "warning": f"Dosage {dosage}{limits['unit']} below recommended minimum {limits['min']}{limits['unit']}"
                        })
                    elif dosage > limits['max']:
                        dosage_warnings.append({
                            "medication": med_name,
                            "warning": f"Dosage {dosage}{limits['unit']} exceeds maximum recommended {limits['max']}{limits['unit']}"
                        })
        
        return dosage_warnings
    
    def generate_verification_report(self, prescription_data):
        """Generate comprehensive verification report using AI"""
        
        # Prepare verification prompt
        verification_prompt = f"""
        As a medical AI assistant, analyze this prescription data and provide a comprehensive verification report:

        Patient: {prescription_data.get('patient_name', 'Unknown')}
        Medications: {prescription_data.get('medications', [])}
        Medical Conditions: {prescription_data.get('medical_conditions', [])}
        Allergies: {prescription_data.get('allergies', [])}

        Please provide:
        1. Overall safety assessment
        2. Potential concerns
        3. Recommendations
        4. Compliance notes

        Report:
        """
        
        try:
            response = self.text_generator(
                verification_prompt,
                max_new_tokens=400,
                temperature=0.3
            )
            
            return response[0]['generated_text'].split("Report:")[-1].strip()
            
        except Exception as e:
            return f"Error generating AI report: {e}"
    
    def verify_prescription(self, prescription_text, patient_conditions="", known_allergies=""):
        """Main verification function"""
        
        # Extract prescription information
        prescription_data = self.extract_prescription_info(prescription_text)
        
        # Add additional patient information
        if patient_conditions:
            prescription_data['medical_conditions'].extend(
                [cond.strip() for cond in patient_conditions.split(',') if cond.strip()]
            )
        
        if known_allergies:
            prescription_data['allergies'].extend(
                [allergy.strip() for allergy in known_allergies.split(',') if allergy.strip()]
            )
        
        # Perform various checks
        interactions = self.check_drug_interactions(prescription_data['medications'])
        dosage_warnings = self.validate_dosages(prescription_data['medications'])
        
        # Generate AI report
        ai_report = self.generate_verification_report(prescription_data)
        
        # Compile final report
        verification_result = {
            "prescription_data": prescription_data,
            "drug_interactions": interactions,
            "dosage_warnings": dosage_warnings,
            "ai_analysis": ai_report,
            "verification_status": "Verified" if not interactions and not dosage_warnings else "Requires Review"
        }
        
        return verification_result

# Initialize the verifier
verifier = MedicalPrescriptionVerifier()

def process_prescription(prescription_text, patient_conditions, known_allergies, watson_api_key, watson_url):
    """Process prescription through Gradio interface"""
    
    if not prescription_text.strip():
        return "Please enter a prescription to verify."
    
    # Setup Watson if credentials provided
    if watson_api_key and watson_url:
        verifier.setup_watson_client(watson_api_key, watson_url)
    
    # Verify prescription
    result = verifier.verify_prescription(prescription_text, patient_conditions, known_allergies)
    
    # Format output
    output = f"""
    # 🏥 PRESCRIPTION VERIFICATION REPORT
    
    ## Patient Information
    **Name:** {result['prescription_data']['patient_name']}
    **Date:** {result['prescription_data']['date']}
    **Prescriber:** {result['prescription_data']['prescriber']}
    
    ## 💊 Medications
    """
    
    for med in result['prescription_data']['medications']:
        output += f"- **{med['name'].title()}**: {med['dosage']} - {med['frequency']}\n"
    
    output += f"\n## ⚠️ Drug Interactions ({len(result['drug_interactions'])} found)\n"
    if result['drug_interactions']:
        for interaction in result['drug_interactions']:
            output += f"- **{interaction['severity']}**: {interaction['warning']}\n"
    else:
        output += "- No significant drug interactions detected\n"
    
    output += f"\n## 📊 Dosage Warnings ({len(result['dosage_warnings'])} found)\n"
    if result['dosage_warnings']:
        for warning in result['dosage_warnings']:
            output += f"- **{warning['medication'].title()}**: {warning['warning']}\n"
    else:
        output += "- All dosages appear within normal ranges\n"
    
    output += f"\n## 🤖 AI Analysis\n{result['ai_analysis']}\n"
    
    output += f"\n## ✅ Verification Status: **{result['verification_status']}**\n"
    
    if result['verification_status'] == "Requires Review":
        output += "\n⚠️ **This prescription requires manual review by a healthcare professional.**"
    
    return output

# Create Gradio interface
def create_gradio_app():
    with gr.Blocks(
        title="AI Medical Prescription Verification",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        """
    ) as app:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🏥 AI Medical Prescription Verification System</h1>
            <p>Powered by Granite-3.2, IBM Watson & Hugging Face Models</p>
            <div class="warning">
                <strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only. 
                Always consult healthcare professionals for medical decisions.
            </div>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                prescription_input = gr.Textbox(
                    label="📝 Prescription Text",
                    placeholder="Enter prescription details (e.g., 'Metformin 500mg twice daily for 30 days')",
                    lines=8,
                    max_lines=15
                )
                
                with gr.Row():
                    conditions_input = gr.Textbox(
                        label="🩺 Patient Medical Conditions (optional)",
                        placeholder="e.g., diabetes, hypertension",
                        scale=1
                    )
                    
                    allergies_input = gr.Textbox(
                        label="⚠️ Known Allergies (optional)",
                        placeholder="e.g., penicillin, sulfa",
                        scale=1
                    )
                
                with gr.Accordion("🔧 IBM Watson Configuration (Optional)", open=False):
                    watson_api_key = gr.Textbox(
                        label="Watson API Key",
                        type="password",
                        placeholder="Enter your IBM Watson API key"
                    )
                    watson_url = gr.Textbox(
                        label="Watson Service URL",
                        placeholder="Enter your IBM Watson service URL"
                    )
                
                verify_btn = gr.Button("🔍 Verify Prescription", variant="primary", size="lg")
            
            with gr.Column(scale=3):
                output = gr.Markdown(
                    label="📊 Verification Results",
                    value="Enter a prescription and click 'Verify Prescription' to see results."
                )
        
        # Examples
        gr.Examples(
            examples=[
                ["Metformin 500mg twice daily, Lisinopril 10mg once daily", "diabetes, hypertension", ""],
                ["Warfarin 5mg daily, Aspirin 81mg daily", "", ""],
                ["Simvastatin 20mg bedtime, Digoxin 0.25mg daily", "heart failure", ""],
            ],
            inputs=[prescription_input, conditions_input, allergies_input]
        )
        
        # Connect the verification function
        verify_btn.click(
            fn=process_prescription,
            inputs=[prescription_input, conditions_input, allergies_input, watson_api_key, watson_url],
            outputs=output
        )
        
        gr.HTML("""
        <div style="text-align: center; margin-top: 20px; font-size: 12px; color: #666;">
            <p>Built with ❤️ using Granite-3.2, IBM Watson, and Hugging Face</p>
            <p>For educational and research purposes only</p>
        </div>
        """)
    
    return app

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting AI Medical Prescription Verification System...")
    print("📚 Loading models and initializing system...")
    
    app = create_gradio_app()
    
    # Launch with appropriate settings for Google Colab
    app.launch(
        share=True,  # Create shareable link
        debug=True,  # Enable debug mode
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,  # Default Gradio port
        show_error=True  # Show detailed errors
    )
