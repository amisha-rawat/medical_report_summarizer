+# Medical Report Summarizer

This project provides a medical report summarization system that can automatically extract and summarize key sections of medical reports using NLP techniques.

## Features

- Extracts key sections from medical reports (Chief Complaint, History of Present Illness, Physical Exam, Diagnosis, Treatment)
- Uses state-of-the-art transformer models for summarization
- Preprocesses text to handle medical-specific formatting
- Provides structured summaries of medical reports

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from medical_summarizer import MedicalSummarizer

# Initialize the summarizer
summarizer = MedicalSummarizer()

# Example medical report
medical_report = """...your medical report text here..."""

# Generate summary
summary = summarizer.generate_summary(medical_report)
```

## Output Format

The summarizer returns a dictionary with summarized sections:
- chief_complaint
- history_of_present_illness
- physical_exam
- diagnosis
- treatment

Each section contains a concise summary of the corresponding part of the medical report.
