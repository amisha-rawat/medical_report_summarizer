# At the top of app.py, before any other imports
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import streamlit first
import streamlit as st

# Then import other libraries
import tensorflow as tf

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import your custom modules
from src.preprocessing import preprocess_text, extract_text_from_pdf_bytes, extract_text_from_image
from src.summarization import MedicalSummarizer
from datetime import datetime

# Lazy load PyTorch only when needed
def get_diagnosis_model():
    try:
        from src.diagnosis import MedicalDiagnosis
        return MedicalDiagnosis()
    except ImportError as e:
        st.error(f"Error loading diagnosis model: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="Medical Report Summarizer & Diagnoser",
        page_icon="🏥",
        layout="wide"
    )
    
    # Initialize the summarizer and diagnoser
    summarizer = MedicalSummarizer()
    
    # Initialize session state for patient info if it doesn't exist
    if 'patient_info' not in st.session_state:
        st.session_state.patient_info = {
            'age': '',
            'sex': '',
            'medical_history': '',
            'current_medications': '',
            'chief_complaint': ''
        }
    
    st.title("🏥 Medical Report Summarizer & Diagnoser")
    st.markdown("""
    Upload a medical report in PDF or image format, or paste the text directly.
    The system will extract and summarize the key medical information and provide potential diagnoses.
    """)
    
    # Create tabs for different input methods and diagnosis
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload PDF", "🖼️ Upload Image (OCR)", "✏️ Enter Text", "🩺 Patient Info"])
    
    # Initialize variables to hold extracted and processed text
    if 'extracted_text' not in st.session_state:
        st.session_state.extracted_text = ""
    extracted_text = st.session_state.extracted_text
    processed_text = ""
    
    with tab1:
        st.header("Upload PDF")
        uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
        if uploaded_pdf is not None:
            with st.spinner("Extracting text from PDF..."):
                extracted_text = extract_text_from_pdf_bytes(uploaded_pdf.read())
                if extracted_text and extracted_text.strip():
                    st.session_state.file_uploaded = True
                    st.session_state.text_submitted = True
                    st.session_state.extracted_text = extracted_text
                    st.success("PDF uploaded and text extracted! You can now generate a summary or diagnosis below.")
                    # --- Begin summary/diagnosis workflow in PDF tab ---
                    processed_text = preprocess_text(extracted_text)
                    with st.expander("View/Edit Extracted Text"):
                        processed_text = st.text_area(
                            "Edit the extracted text if needed:",
                            value=processed_text,
                            height=200,
                            key="pdf_text_editor"
                        )
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Generate Summary", key="pdf_summary_btn"):
                            with st.spinner("Generating summary..."):
                                try:
                                    summary = summarizer.summarize_text(processed_text)
                                    st.subheader("📝 Summary")
                                    st.write(summary)
                                    st.session_state.summary = summary
                                except Exception as e:
                                    st.error(f"Error generating summary: {str(e)}")
                    with col2:
                        if st.button("Generate Diagnosis", key="pdf_diag_btn"):
                            with st.spinner("Analyzing for diagnoses..."):
                                try:
                                    diagnoser = get_diagnosis_model()
                                    if diagnoser is None:
                                        st.error("Failed to load diagnosis model. Please check the logs for details.")
                                    else:
                                        patient_info = {k: v for k, v in st.session_state.patient_info.items() if v}
                                        result = diagnoser.generate_diagnosis(processed_text, patient_info)
                                        if not result or 'error' in result:
                                            error_msg = result.get('error', 'Unknown error occurred during diagnosis')
                                            st.error(f"Diagnosis error: {error_msg}")
                                        else:
                                            st.subheader("🩺 Potential Diagnosis")
                                            st.markdown(result['diagnosis'])
                                            st.caption(f"Generated using {result['model']}")
                                            with st.expander("View Raw Diagnosis Output"):
                                                st.subheader("Raw Model Output")
                                                diagnosis_labels = [{"diagnosis": dx['label'].replace('LABEL_', '')} for dx in result['diagnoses']]
                                                st.json({
                                                    "model": result['model'],
                                                    "diagnoses": diagnosis_labels,
                                                    "timestamp": str(datetime.now())
                                                })
                                                st.subheader("Formatted Output")
                                                st.code(result['diagnosis'], language="markdown")
                                            st.session_state.diagnosis = result['diagnosis']
                                except Exception as e:
                                    st.error(f"Error generating diagnosis: {str(e)}")
                                    import traceback
                                    st.error(f"Stack trace: {traceback.format_exc()}")
                    # --- End summary/diagnosis workflow in PDF tab ---
                else:
                    st.error("Failed to extract any text from the uploaded PDF. Please check the file or try another.")
    
    with tab2:
        st.header("Upload Image (OCR)")
        uploaded_image = st.file_uploader("Choose an image file", 
                                        type=["jpg", "jpeg", "png"], 
                                        key="image_uploader")
        
        if uploaded_image is not None:
            with st.spinner("Extracting text from image..."):
                extracted_text = extract_text_from_image(uploaded_image)
                if extracted_text and extracted_text.strip():
                    st.session_state.file_uploaded = True
                    st.session_state.text_submitted = True
                    st.session_state.extracted_text = extracted_text
                    st.success("Image uploaded and text extracted! Scroll down to generate a summary or diagnosis.")
                    with st.expander("View Extracted Text (from Image)"):
                        st.write(extracted_text)
                else:
                    st.error("Failed to extract any text from the uploaded image. Please check the file or try another.")
    
    with tab3:
        st.header("Enter Text")
        with st.form("text_input_form"):
            text_input = st.text_area("Paste the medical report text here:", height=200, key="text_input")
            submit_text = st.form_submit_button("Submit Text")
            
            if submit_text and text_input:
                st.session_state.extracted_text = text_input
                st.session_state.text_submitted = True
            elif 'text_submitted' in st.session_state and st.session_state.text_submitted:
                st.session_state.extracted_text = text_input
            else:
                st.session_state.extracted_text = ""
    
    with tab4:
        st.header("Patient Information")
        st.markdown("Provide additional patient information to improve diagnosis accuracy.")
        
        # Form for patient information
        with st.form("patient_info_form"):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.patient_info['age'] = st.text_input("Age", value=st.session_state.patient_info['age'])
                st.session_state.patient_info['sex'] = st.selectbox(
                    "Sex", 
                    options=['', 'Male', 'Female', 'Other', 'Prefer not to say'],
                    index=['', 'Male', 'Female', 'Other', 'Prefer not to say'].index(st.session_state.patient_info['sex']) if st.session_state.patient_info['sex'] else 0
                )
            with col2:
                st.session_state.patient_info['chief_complaint'] = st.text_area(
                    "Chief Complaint", 
                    value=st.session_state.patient_info['chief_complaint'],
                    height=68
                )
            
            st.session_state.patient_info['medical_history'] = st.text_area(
                "Medical History", 
                value=st.session_state.patient_info['medical_history'],
                placeholder="Any relevant medical history (e.g., chronic conditions, past surgeries, allergies)",
                height=100
            )
            
            st.session_state.patient_info['current_medications'] = st.text_area(
                "Current Medications", 
                value=st.session_state.patient_info['current_medications'],
                placeholder="List current medications and dosages",
                height=100
            )
            
            st.form_submit_button("Save Patient Information")
    
    # Process the extracted text if available
    if st.session_state.extracted_text and (('text_submitted' in st.session_state and st.session_state.text_submitted) or 'file_uploaded' in st.session_state):
        processed_text = preprocess_text(st.session_state.extracted_text)
        
        # Display extracted text with option to edit
        with st.expander("View/Edit Extracted Text"):
            processed_text = st.text_area(
                "Edit the extracted text if needed:", 
                value=processed_text, 
                height=200,
                key="text_editor"
            )
        
        # Create columns for summary and diagnosis
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Summary"):
                with st.spinner("Generating summary..."):
                    try:
                        summary = summarizer.summarize_text(processed_text)
                        st.subheader("📝 Summary")
                        st.write(summary)
                        
                        # Store summary in session state for download
                        st.session_state.summary = summary
                        
                    except Exception as e:
                        st.error(f"Error generating summary: {str(e)}")
        
        with col2:
            if True:
                if st.button("Generate Diagnosis"):
                    with st.spinner("Analyzing for diagnoses..."):
                        try:
                            # Initialize the diagnoser with lazy loading
                            diagnoser = get_diagnosis_model()
                            if diagnoser is None:
                                st.error("Failed to load diagnosis model. Please check the logs for details.")
                            else:
                                # Filter out empty patient info
                                patient_info = {k: v for k, v in st.session_state.patient_info.items() if v}
                                
                                # Generate diagnosis
                                result = diagnoser.generate_diagnosis(processed_text, patient_info)
                                
                                if not result or 'error' in result:
                                    error_msg = result.get('error', 'Unknown error occurred during diagnosis')
                                    st.error(f"Diagnosis error: {error_msg}")
                                else:
                                    st.subheader("🩺 Potential Diagnosis")
                                    st.markdown(result['diagnosis'])
                                    st.caption(f"Generated using {result['model']}")
                                    
                                    # Add expandable section for raw diagnosis output
                                    with st.expander("View Raw Diagnosis Output"):
                                        st.subheader("Raw Model Output")
                                        # Extract just the diagnosis labels without scores
                                        diagnosis_labels = [{"diagnosis": dx['label'].replace('LABEL_', '')} for dx in result['diagnoses']]
                                        st.json({
                                            "model": result['model'],
                                            "diagnoses": diagnosis_labels,
                                            "timestamp": str(datetime.now())
                                        })
                                        
                                        st.subheader("Formatted Output")
                                        st.code(result['diagnosis'], language="markdown")
                                    
                                    # Store diagnosis in session state for download
                                    st.session_state.diagnosis = result['diagnosis']
                        
                        except Exception as e:
                            st.error(f"Error generating diagnosis: {str(e)}")
                            import traceback
                            st.error(f"Stack trace: {traceback.format_exc()}")
            else:
                st.warning("⚠️ The diagnosis feature requires the transformers and torch packages. Please make sure they are installed.")
                
        # Add download buttons if summary or diagnosis exists
        if 'summary' in st.session_state or 'diagnosis' in st.session_state:
            st.markdown("---")
            st.subheader("Download Results")
            
            col1, col2 = st.columns(2)
            
            if 'summary' in st.session_state:
                with col1:
                    st.download_button(
                        label="Download Summary",
                        data=st.session_state.summary,
                        file_name="medical_summary.txt",
                        mime="text/plain"
                    )
            
            if 'diagnosis' in st.session_state:
                with col2:
                    st.download_button(
                        label="Download Diagnosis",
                        data=st.session_state.diagnosis,
                        file_name="medical_diagnosis.txt",
                        mime="text/plain"
                    )

    # Add a sidebar with instructions and info
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        This application helps in summarizing medical reports using advanced NLP techniques.
        
        **Features:**
        - Extract text from PDFs and images (OCR)
        - Clean and preprocess medical text
        - Generate concise summaries of medical reports
        - Provide potential diagnoses based on patient information
        - Download the generated summaries and diagnoses
        
        **Note:** For best results with image-based documents, ensure the text is clear and properly aligned.
        """)
    
    # Cleanup
    summarizer.close()

if __name__ == "__main__":
    main()
