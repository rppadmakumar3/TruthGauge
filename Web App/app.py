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
def extract_text_from_pdf(pdf_file):
    with open(pdf_file, "rb") as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        text = ""
        for page_num in range(pdf_reader.numPages):
            text += pdf_reader.getPage(page_num).extractText()
    return text

def extract_text_from_document(document_file):
    if document_file.name.endswith(".docx"):
        doc = docx.Document(document_file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        # Handle other document formats as needed
        text = ""  # Implement extraction for other formats
    return text

def extract_text_from_youtube(youtube_link):
    with youtube_dl.YoutubeDL() as ydl:
        info_dict = ydl.extract_info(youtube_link, download=False)
        if 'entries' in info_dict:
            subtitles = info_dict['entries'][0].get('subtitles')
            if subtitles:
                text = "\n".join(subtitles.values())
                return text
    return ""

def extract_text_from_webpage(webpage_link):
    try:
        response = requests.get(webpage_link)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = " ".join([p.get_text() for p in soup.find_all(['p', 'div'])])
        return text
    except Exception as e:
        st.error(f"Error extracting text from the webpage: {str(e)}")
        return ""

# Prediction function with credibility score
def predict_credibility(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).detach().numpy()[0]
  
    misinformation_score = probabilities[label_list.index("REFUTES")] * 100
    good_information_score = probabilities[label_list.index("SUPPORTS")] * 100
  
    return misinformation_score, good_information_score

# Streamlit app
st.title("Credibility Prediction App")
st.write("This app predicts the credibility of a claim by analyzing the provided text.")

st.sidebar.markdown(
    """
    **Instructions:**
    - Select the input type from the dropdown menu.
    - Enter the claim, upload a file, provide a YouTube link, or enter a webpage link.
    - Click the 'Predict' button to get credibility scores.
    """
)

input_type = st.sidebar.selectbox("Select input type:", ["Text", "PDF", "Document", "YouTube Link", "Webpage Link"])

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