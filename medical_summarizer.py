import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import re
from typing import List, Dict, Tuple

class MedicalSummarizer:
    def __init__(self):
        # Load medical NLP model
        self.nlp = spacy.load("en_core_web_sm")
        # Load summarization model
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess medical text by removing special characters and normalizing."""
        # Remove multiple newlines and extra spaces
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters while preserving medical abbreviations
        text = re.sub(r'[^\w\s.,;:\-()]+', ' ', text)
        return text.strip()
    
    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from medical report."""
        sections = {
            'chief_complaint': '',
            'history_of_present_illness': '',
            'physical_exam': '',
            'diagnosis': '',
            'treatment': ''
        }
        
        doc = self.nlp(text)
        
        # Look for section headers
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if 'chief complaint' in sent_text:
                sections['chief_complaint'] = sent.text
            elif 'history of present illness' in sent_text:
                sections['history_of_present_illness'] = sent.text
            elif 'physical exam' in sent_text:
                sections['physical_exam'] = sent.text
            elif 'diagnosis' in sent_text:
                sections['diagnosis'] = sent.text
            elif 'treatment' in sent_text:
                sections['treatment'] = sent.text
        
        return sections
    
    def summarize_section(self, text: str, max_length: int = 150) -> str:
        """Summarize a section of text using BART model."""
        try:
            summary = self.summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
            return summary
        except Exception as e:
            print(f"Error summarizing text: {e}")
            return "Error generating summary"

    def summarize(self, text: str) -> str:
        """Summarize the entire medical report."""
        try:
            # Preprocess the text
            processed_text = self.preprocess_text(text)
            
            # Extract key sections
            sections = self.extract_key_sections(processed_text)
            
            # Summarize each section
            summaries = []
            for section, content in sections.items():
                if content:
                    summary = self.summarize_section(content)
                    summaries.append(f"{section.replace('_', ' ').title()}: {summary}")
            
            # Combine all section summaries
            full_summary = "\n\n".join(summaries)
            return full_summary
        except Exception as e:
            print(f"Error in full summarization: {e}")
            return "Error generating summary"

    def generate_summary(self, medical_report: str) -> Dict[str, str]:
        """Generate comprehensive summary of medical report."""
        # Preprocess the text
        clean_text = self.preprocess_text(medical_report)
        
        # Extract key sections
        sections = self.extract_key_sections(clean_text)
        
        # Summarize each section
        summarized_sections = {}
        for section, content in sections.items():
            summarized_sections[section] = self.summarize_section(content)
        
        return summarized_sections

def main():
    # Example usage
    summarizer = MedicalSummarizer()
    
    # Example medical report (replace with actual report)
    medical_report = """
    Chief Complaint:
    65-year-old male with chest pain
    
    History of Present Illness:
    Patient presents with substernal chest pain for 2 hours. Pain is crushing in nature,
    radiates to left arm, and is associated with diaphoresis. No history of similar episodes.
    
    Physical Exam:
    Vitals: BP 140/90, HR 95, RR 18, Temp 98.6
    Cardiac: Regular rate and rhythm, no murmurs
    
    Diagnosis:
    Acute myocardial infarction
    
    Treatment:
    Aspirin 325 mg, Nitroglycerin 0.4 mg SL, Heparin IV
    """
    
    summary = summarizer.generate_summary(medical_report)
    print("\nMedical Report Summary:")
    for section, content in summary.items():
        print(f"\n{section.replace('_', ' ').title()}:")
        print(content)

if __name__ == "__main__":
    main()
