import os
from typing import Dict, Optional
import openai
from dotenv import load_dotenv

def generate_diagnosis(text, patient_info=None, model="llama3-70b-8192"):
    """
    Minimal function to generate diagnosis using Groq API (OpenAI-compatible).
    Args:
        text (str): Clinical notes or medical report.
        patient_info (dict, optional): Patient info like age, sex, etc.
        model (str): Groq model name (e.g., 'llama3-70b-8192').
    Returns:
        str: Diagnosis response from Groq.
    """
    load_dotenv()
    openai.api_key = os.getenv("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")
    openai.base_url = "https://api.groq.com/openai/v1"
    prompt = (
        "You are a clinical diagnosis assistant. Given the following patient information and clinical notes, "
        "list the most likely diagnoses and provide reasoning. "
        "If information is insufficient, state so.\n\n"
    )
    if patient_info:
        for k, v in patient_info.items():
            prompt += f"{k.capitalize()}: {v}\n"
    prompt += f"\nClinical Notes:\n{text}\n"
    prompt += "\nFormat your response as a numbered list."
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.2
    )
    return response["choices"][0]["message"]["content"].strip()

class MedicalDiagnosis:
    def __init__(self, model_name: str = "llama3-70b-8192"):
        """
        Initialize the MedicalDiagnosis class using Groq API.
        Args:
            model_name (str): Groq model name (e.g., 'llama3-70b-8192')
        """
        load_dotenv()
        self.api_key = os.getenv("GROQ_API_KEY")
        openai.api_key = self.api_key
        openai.base_url = "https://api.groq.com/openai/v1"
        self.model_name = model_name

    def generate_diagnosis(self, medical_text: str, patient_info: Optional[Dict] = None) -> Dict:
        """
        Generate potential diagnoses using Groq API based on medical text and patient info.
        Args:
            medical_text (str): The medical report or notes to analyze
            patient_info (dict, optional): Additional patient information (age, sex, medical history, etc.)
        Returns:
            dict: Dictionary containing diagnosis information
        """
        try:
            input_text = self._prepare_input_text(medical_text, patient_info)
            prompt = (
                "You are a clinical diagnosis assistant. Given the following patient information and clinical notes, "
                "list the most likely diagnoses (in order of likelihood) and provide a brief reasoning for each. "
                "If information is insufficient, state so.\n\n"
                f"{input_text}\n\n"
                "Format your response as a numbered list."
            )
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.2
            )
            diagnosis_text = response["choices"][0]["message"]["content"].strip()
            return {
                "diagnosis": diagnosis_text,
                "model": self.model_name,
                "diagnoses": diagnosis_text  # For compatibility, can parse if needed
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
