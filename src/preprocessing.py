# Text preprocessing functions

import nltk
import spacy
import fitz  # PyMuPDF
import io
from PIL import Image  # For image handling
import pytesseract  # For OCR
from typing import Union, Optional, Tuple

# Download necessary NLTK data (run once)
# print("Downloading NLTK resources...")
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt', quiet=True)
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords', quiet=True)
# try:
#     nltk.data.find('corpora/wordnet')
# except nltk.downloader.DownloadError:
#     nltk.download('wordnet', quiet=True)
# print("NLTK resources checked/downloaded.")

# Load spaCy model (ensure you have it downloaded: python -m spacy download en_core_web_sm)
# print("Loading spaCy model...")
# nlp = None # Initialize to None
# try:
#     nlp = spacy.load('en_core_web_sm')
#     print("spaCy model 'en_core_web_sm' loaded.")
# except OSError:
#     print("spaCy model 'en_core_web_sm' not found. Please download it by running: python -m spacy download en_core_web_sm")
#     print("Continuing without spaCy model for now.")

def extract_text_from_pdf_bytes(pdf_bytes):
    """Extracts text from PDF bytes."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
        print(f"Successfully extracted text from PDF bytes.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF bytes: {e}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from all pages of a PDF file using pdf_bytes."""
    try:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        print(f"Reading PDF from path: {pdf_path}")
        return extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as e:
        print(f"Error reading PDF file {pdf_path}: {e}")
        return None

def extract_text_from_image(image_bytes: bytes) -> Tuple[Optional[str], str]:
    """Extracts text from an image using OCR.
    
    Args:
        image_bytes: The image file content as bytes.
        
    Returns:
        A tuple of (extracted_text, status_message)
    """
    try:
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale for better OCR results
        image = image.convert('L')
        
        # Extract text using Tesseract OCR
        text = pytesseract.image_to_string(image)
        
        if not text.strip():
            return None, "No text could be extracted from the image. The image might be blurry or contain no text."
            
        return text, "Text extracted successfully from image."
        
    except Exception as e:
        error_msg = f"Error during OCR processing: {str(e)}"
        print(error_msg)
        return None, error_msg

def preprocess_text(text: str) -> str:
    """Basic text preprocessing: tokenization, stopword removal, lemmatization."""
    print(f"Original text length: {len(text)}")
    
    # Placeholder for actual preprocessing logic
    # 1. OCR (if needed, handled before this function)
    # 2. Noise removal (headers, footers, etc.)
    # 3. Tokenization (e.g., using nltk.word_tokenize)
    # 4. Stopword removal (e.g., using nltk.corpus.stopwords)
    # 5. Lemmatization (e.g., using nltk.stem.WordNetLemmatizer or spaCy)
    
    processed_text = text.lower()  # Simple example, will be expanded
    print(f"Processed text length: {len(processed_text)}")
    return processed_text

if __name__ == '__main__':
    # This block is for testing the preprocessing module directly
    # It's good practice to ensure NLTK resources are available when testing
    # You might want to uncomment the NLTK download lines above for a first run,
    # or handle it in a setup script.
    print("\n--- Preprocessing sample text (module test) ---")
    sample_text = "This is a sample medical report. Patient X shows symptoms of Y. Advised Z."
    cleaned_text = preprocess_text(sample_text)
    print(f"Cleaned sample text: {cleaned_text}")
    print(f"--- End of preprocessing (module test) ---")

    # Example of PDF extraction (assuming a dummy.pdf exists in the 'data' directory)
    # You would need to create a 'dummy.pdf' in the 'data' folder for this to run.
    # import os
    # SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
    # pdf_path = os.path.join(PROJECT_DIR, 'data', 'dummy.pdf') 
    # print(f"\n--- Attempting to extract text from PDF: {pdf_path} ---")
    # if os.path.exists(pdf_path):
    #     pdf_text = extract_text_from_pdf(pdf_path)
    #     if pdf_text:
    #         print(f"Extracted PDF text length: {len(pdf_text)}")
    #         # cleaned_pdf_text = preprocess_text(pdf_text) # Optionally preprocess
    #         # print(f"Cleaned PDF text: {cleaned_pdf_text}")
    # else:
    #     print(f"PDF file not found at {pdf_path}. Skipping PDF extraction test.")
    # print(f"--- End of PDF extraction test ---")
