import streamlit as st
import nltk
import worker

def detect_sarcasm(text):

    return bool(worker.detect(text))

def main():

    if not nltk.data.find('corpora/stopwords.zip'):
        nltk.download('stopwords')

    if not nltk.data.find('tokenizers/punkt'):
        nltk.download('punkt')

    st.title("Sarcasm detector")
    text = st.text_area("Enter the text to analyze:")

    if st.button("Analyze"):
        with st.spinner("Processing..."):
            result = detect_sarcasm(text)
            if result:
                st.success("Sarcasm detected!")
            else:
                st.error("Sarcam is not detected!")

if __name__ == "__main__":
    main()