from flask import Flask, request, jsonify, render_template
from medical_summarizer import MedicalSummarizer
import os
import base64
from PyPDF2 import PdfReader
import io

app = Flask(__name__, template_folder='static')
summarizer = MedicalSummarizer()

@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/test')
def test():
    return "Server is running!"

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        summary = summarizer.summarize(text)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/extract-pdf', methods=['POST'])
def extract_pdf():
    try:
        data = request.json
        filename = data.get('filename', '')
        base64_data = data.get('data', '')
        
        if not base64_data:
            return jsonify({'error': 'No PDF data provided'}), 400
            
        # Convert base64 to bytes
        pdf_bytes = base64.b64decode(base64_data)
        
        # Create PDF reader
        pdf_reader = PdfReader(io.BytesIO(pdf_bytes))
        
        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        
        return jsonify({
            'text': text,
            'pages': len(pdf_reader.pages),
            'filename': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
