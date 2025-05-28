from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from typing import List, Dict, Optional
import torch
import re

class MedicalDiagnosis:
    def __init__(self, model_name: str = "DATEXIS/CORe-clinical-diagnosis-prediction"):
        """
        Initialize the MedicalDiagnosis class with the specified model.
        
        Args:
            model_name (str): Name of the pre-trained model from Hugging Face
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Load the tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        
        # Initialize the text classification pipeline
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
    
    def generate_diagnosis(self, medical_text: str, patient_info: Optional[Dict] = None) -> Dict:
        """
        Generate potential diagnoses based on medical text using the clinical diagnosis model.
        
        Args:
            medical_text (str): The medical report or notes to analyze
            patient_info (dict, optional): Additional patient information (age, sex, medical history, etc.)
            
        Returns:
            dict: Dictionary containing diagnosis information
        """
        try:
            # Prepare the input text with patient information if available
            input_text = self._prepare_input_text(medical_text, patient_info)
            
            # Get predictions from the model
            results = self.classifier(
                input_text,
                top_k=5,  # Get top 5 most likely diagnoses
                truncation=True,
                max_length=512
            )
            
            # Format the results
            formatted_diagnoses = self._format_diagnoses(results)
            
            return {
                "diagnosis": formatted_diagnoses,
                "model": self.model_name,
                "diagnoses": results
            }
            
        except Exception as e:
            return {
                "error": f"Error generating diagnosis: {str(e)}",
                "diagnosis": ""
            }
    
    def _prepare_input_text(self, medical_text: str, patient_info: Optional[Dict] = None) -> str:
        """Prepare the input text by combining patient info and medical text."""
        parts = []
        
        if patient_info:
            # Add patient information if available
            if 'age' in patient_info and patient_info['age']:
                parts.append(f"Age: {patient_info['age']}")
            if 'sex' in patient_info and patient_info['sex']:
                parts.append(f"Sex: {patient_info['sex']}")
            if 'chief_complaint' in patient_info and patient_info['chief_complaint']:
                parts.append(f"Chief Complaint: {patient_info['chief_complaint']}")
            if 'medical_history' in patient_info and patient_info['medical_history']:
                parts.append(f"Medical History: {patient_info['medical_history']}")
            if 'current_medications' in patient_info and patient_info['current_medications']:
                parts.append(f"Current Medications: {patient_info['current_medications']}")
            
            parts.append("\nClinical Notes:")
        
        parts.append(medical_text)
        return "\n".join(parts)
    
    def _format_diagnoses(self, diagnoses: List[Dict]) -> str:
        """Format the diagnosis results into a human-readable string."""
        if not diagnoses:
            return "No specific diagnosis could be determined from the provided information."
        
        formatted = ["## Potential Diagnoses (from most to least likely):\n"]
        
        for i, dx in enumerate(diagnoses, 1):
            # Clean up the label (remove 'LABEL_' prefix if present)
            label = dx['label']
            if label.startswith('LABEL_'):
                label = label[6:]  # Remove 'LABEL_' prefix
            
            # Convert to human-readable format (replace underscores with spaces, capitalize words)
            label = ' '.join(word.capitalize() for word in label.split('_'))
            
            # Format the confidence as a percentage
            confidence = dx['score'] * 100
            
            formatted.append(f"{i}. **{label}** (Confidence: {confidence:.1f}%)")
        
        # Add a note about clinical judgment
        formatted.append("\n*Note: This is an AI-generated assessment and should be reviewed by a qualified healthcare professional.*")
        
        return "\n".join(formatted)
