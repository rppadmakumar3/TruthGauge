import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
import requests
import youtube_dl
import PyPDF2
import docx

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("rppadmakumar/fever_model")

# Define label list
label_list = ["SUPPORTS", "REFUTES"]

# Helper functions for extracting text from different sources
# (Your existing helper functions remain unchanged)

# Prediction function with credibility score
# (Your existing prediction function remains unchanged)

# Streamlit app adjustments for improved user interface
st.title("Credibility Prediction App")
st.write("This app predicts the credibility of a claim by analyzing the provided text.")

# Sidebar for instructions and input type selection
st.sidebar.markdown(
    """
    **Instructions:**
    - Select the input type from the dropdown menu.
    - Enter the claim, upload a file, provide a YouTube link, or enter a webpage link.
    - Click the 'Predict' button to get credibility scores.
    """
)

input_type = st.sidebar.selectbox("Select input type:", ["Text", "PDF", "Document", "YouTube Link", "Webpage Link"])

# Main content area for input and prediction
st.subheader("Enter Your Data:")
input_data = None

if input_type == "Text":
    claim = st.text_area("Enter your claim:", height=200)
    input_data = claim
elif input_type in ["PDF", "Document"]:
    uploaded_file = st.file_uploader(f"Upload {input_type} file:", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        input_data = uploaded_file
elif input_type == "YouTube Link":
    youtube_link = st.text_input("Enter YouTube video link:")
    input_data = youtube_link
elif input_type == "Webpage Link":
    webpage_link = st.text_input("Enter webpage link:")
    input_data = webpage_link

# Prediction button
if st.button("Predict"):
    if input_data:
        try:
            text = ""
            if input_type == "Text":
                text = input_data
            elif input_type in ["PDF", "Document"]:
                text = extract_text_from_pdf(input_data) if input_type == "PDF" else extract_text_from_document(input_data)
            elif input_type == "YouTube Link":
                text = extract_text_from_youtube(input_data)
            elif input_type == "Webpage Link":
                text = extract_text_from_webpage(input_data)

            if text:
                misinformation_score, good_information_score = predict_credibility(text)
                st.subheader("Prediction Results:")
                st.write(f"Input Data: {text[:500]}...")  # Show the first 500 characters followed by ellipsis
                st.write(f"Misinformation Score: {misinformation_score:.2f}%")
                st.write(f"Good Information Score: {good_information_score:.2f}%")
            else:
                st.warning("Failed to extract text from the input.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter valid input based on the selected type.")