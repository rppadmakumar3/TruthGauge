import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def text_input():
    st.header("Text Input")

    # Your text-specific input code here
    claim = st.text_area("Enter your claim:", height=300)

    # Model prediction code
    if st.button("Predict"):
        if claim:
            try:
                # Load the pre-trained model and tokenizer
                model_name = "distilbert-base-uncased"
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained("Mahesh6392/score_model")

                # Prediction function with credibility score
                inputs = tokenizer(claim, return_tensors="pt", max_length=512, truncation=True)
                outputs = model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1).detach().numpy()[0]

                misinformation_score = probabilities[0] * 100  # Assuming index 0 corresponds to "REFUTES"
                good_information_score = probabilities[1] * 100  # Assuming index 1 corresponds to "SUPPORTS"

                st.write(f"Input Data: {claim[:500]}...")
                st.write(f"Misinformation Score: {misinformation_score:.2f}%")
                st.write(f"Good Information Score: {good_information_score:.2f}%")

                # Add "Rewrite" button if the misinformation score is high
                if misinformation_score > 70:  # Adjust the threshold as needed
                    if st.button("Rewrite to remove mis information"):
                        rewritten_claim = rewrite_text(claim)
                        st.write("Rewritten Claim:", rewritten_claim)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a claim for prediction.")

def rewrite_text(text):
    # Use the Hugging Face pipeline for rewriting
    rewriter_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    rewritten_text = rewriter_pipeline(text, max_length=150, min_length=50, num_beams=4, length_penalty=2.0)[0]['generated_text']
    return rewritten_text