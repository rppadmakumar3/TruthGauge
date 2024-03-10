import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

def web_input():
    st.header("Web Input")

    # Your web-specific input code here
    webpage_link = st.text_input("Enter webpage link:")

    # Model prediction code
    if st.button("Predict"):
        if webpage_link:
            try:
                # Extract text from the webpage
                text = extract_text_from_webpage(webpage_link)

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
                    st.warning("Failed to extract text from the webpage.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a valid webpage link for prediction.")

# Helper function to extract text from a webpage
def extract_text_from_webpage(webpage_link):
    try:
        response = requests.get(webpage_link)
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract text from paragraphs, divs, or other relevant HTML tags
        text = " ".join([p.get_text() for p in soup.find_all(['p', 'div'])])
        return text
    except Exception as e:
        st.error(f"Error extracting text from the webpage: {str(e)}")
        return ""

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