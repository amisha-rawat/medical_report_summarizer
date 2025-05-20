import os
import json
import pandas as pd
from typing import List, Dict
from pathlib import Path

class MedicalDataLoader:
    def __init__(self, dataset_dir: str = 'dataset'):
        self.dataset_dir = Path(dataset_dir)
        self.train_dir = self.dataset_dir / 'train'
        self.test_dir = self.dataset_dir / 'test'
        self.validation_dir = self.dataset_dir / 'validation'
    
    def load_reports(self, split: str = 'train') -> List[Dict]:
        """Load medical reports from a specific split."""
        if split == 'train':
            data_path = self.train_dir / 'train_data.json'
        elif split == 'test':
            data_path = self.test_dir / 'test_data.json'
        elif split == 'validation':
            data_path = self.validation_dir / 'validation_data.json'
        else:
            raise ValueError(f"Invalid split: {split}")
            
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        return pd.read_json(data_path, lines=True).to_dict('records')
    
    def get_section_text(self, report: Dict, section: str) -> str:
        """Extract specific section text from a report."""
        section_map = {
            'chief_complaint': 'Chief Complaint',
            'history_of_present_illness': 'History of Present Illness',
            'physical_exam': 'Physical Exam',
            'diagnosis': 'Diagnosis',
            'treatment': 'Treatment'
        }
        
        for doc in report.get('documents', []):
            if doc.get('section', '').strip().lower() == section_map[section].lower():
                return doc.get('text', '')
        return ''
    
    def format_report(self, report: Dict) -> Dict:
        """Format a report into a standardized structure."""
        formatted_report = {
            'chief_complaint': self.get_section_text(report, 'chief_complaint'),
            'history_of_present_illness': self.get_section_text(report, 'history_of_present_illness'),
            'physical_exam': self.get_section_text(report, 'physical_exam'),
            'diagnosis': self.get_section_text(report, 'diagnosis'),
            'treatment': self.get_section_text(report, 'treatment'),
            'summary': report.get('summary', '')
        }
        return formatted_report
    
    def load_formatted_reports(self, split: str = 'train') -> List[Dict]:
        """Load and format reports from a specific split."""
        reports = self.load_reports(split)
        return [self.format_report(report) for report in reports]

def main():
    """Test loading and formatting of reports."""
    data_loader = MedicalDataLoader()
    try:
        sample_report = data_loader.load_formatted_reports('train')[0]
        print("Sample report:")
        print(json.dumps(sample_report, indent=2))
    except Exception as e:
        print(f"Error loading reports: {e}")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
