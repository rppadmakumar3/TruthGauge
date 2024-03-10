import streamlit as st
import PyPDF2
import docx
from transformers import pipeline

def document_input():
    st.header("Document Input")

    # Your document-specific input code here
    uploaded_file = st.file_uploader("Upload document file:", type=["pdf", "docx", "txt"])

    # Model prediction code
    if st.button("Predict"):
        if uploaded_file is not None:
            try:
                # Extract text from the document
                text = extract_text_from_document(uploaded_file)

                if text:
                    # Perform model prediction on the extracted text
                    misinformation_score, good_information_score = predict_credibility(text)
                    st.write(f"Input Data: {text[:500]}...")
                    st.write(f"Misinformation Score: {misinformation_score:.2f}%")
                    st.write(f"Good Information Score: {good_information_score:.2f}%")

                    # Add "Rewrite" button if the misinformation score is high
                    if misinformation_score > 70:  # Adjust the threshold as needed
                        if st.button("Rewrite to remove mis information"):
                            rewritten_text = rewrite_text(text)
                            st.write("Rewritten Text:", rewritten_text)

                else:
                    st.warning("Failed to extract text from the document.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload a valid document for prediction.")

# Helper function to extract text from a document
def extract_text_from_document(document_file):
    if document_file.name.endswith(".docx"):
        doc = docx.Document(document_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    elif document_file.name.endswith(".pdf"):
        with open(document_file, "rb") as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page_num in range(pdf_reader.numPages):
                text += pdf_reader.getPage(page_num).extractText()
    else:
        # Handle other document formats as needed
        text = ""  # Implement extraction for other formats
    return text

def predict_credibility(text):
    # Placeholder for model prediction logic
    # Replace this with your actual model prediction code
    misinformation_score = 75.0
    good_information_score = 25.0
    return misinformation_score, good_information_score

def rewrite_text(text):
    # Use the Hugging Face pipeline for rewriting
    rewriter_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    rewritten_text = rewriter_pipeline(text, max_length=150, min_length=50, num_beams=4, length_penalty=2.0)[0]['generated_text']
    return rewritten_text