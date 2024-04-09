import streamlit as st
import worker

def detect_sarcasm(text):
    return bool(worker.detect(text))

def main():
    worker.prepare_data()

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