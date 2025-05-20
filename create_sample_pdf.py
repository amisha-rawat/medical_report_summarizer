from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Medical Report', 0, 1, 'C')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

# Create PDF
pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', '', 12)

# Add content
content = """
Patient Information:
Name: John Smith
Age: 45
Sex: Male
Date: May 20, 2025

Chief Complaint:
Patient presents with chest pain and shortness of breath for the past 24 hours. Pain is described as substernal, crushing, and radiates to the left arm. Associated symptoms include diaphoresis and nausea.

History of Present Illness:
The patient is a 45-year-old male who presents with sudden onset chest pain that started yesterday evening. The pain is described as substernal, crushing in nature, and radiates to the left arm. He also reports diaphoresis and nausea. The pain is not relieved by rest or nitroglycerin. He denies any recent travel or exposure to sick contacts.

Physical Examination:
Vital Signs:
- Temperature: 98.6°F
- Blood Pressure: 140/90 mmHg
- Heart Rate: 110 bpm
- Respiratory Rate: 20 breaths/min
- Oxygen Saturation: 95% on room air

Cardiovascular:
- Regular rate and rhythm
- No murmurs, gallops, or rubs
- No jugular venous distension

Respiratory:
- Clear to auscultation bilaterally
- No wheezes, rales, or rhonchi

Diagnosis:
Acute myocardial infarction

Treatment:
- Aspirin 325 mg chewed
- Nitroglycerin 0.4 mg sublingual
- Oxygen 2L by nasal cannula
- EKG shows ST-segment elevation in leads II, III, and aVF
- Troponin I elevated at 5.2 ng/mL
- Patient transferred to cath lab for emergent PCI
"""

# Split content into lines and add to PDF
for line in content.split('\n'):
    pdf.cell(0, 10, line, 0, 1)

# Save PDF
pdf.output('samples/sample_medical_report.pdf')
