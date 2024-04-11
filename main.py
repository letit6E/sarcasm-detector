import streamlit as st
import random
import time

import torch

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification



def use_bert(text) -> bool:

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = AutoModelForSequenceClassification.from_pretrained('checkpoint-9375/')
    tokenizer = AutoTokenizer.from_pretrained('checkpoint-9375/')
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax().item()
    res = model.config.id2label[predicted_class_id]
    if res == 'POSITIVE':
        return True
    else:
        return False


def detect_sarcasm(text, model):
    
    
    if model=='BERT':
        res = use_bert(text)

    
    return res

def main():
    st.title("Sarcasm detector")

    text = st.text_area("Enter the text to analyze:")

    model = 'BERT'

    if st.button("Analyze"):
        with st.spinner("Processing..."):
            result = detect_sarcasm(text, model)
            if result:
                st.success("Sarcasm detected!")
            else:
                st.error("Sarcam is not detected!")

if __name__ == "__main__":
    main()