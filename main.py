import streamlit as st
from models.LogReg_bow_pred import LogReg_bow_pred


def detect_sarcasm(text):
    return LogReg_bow_pred(text)


def main():
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
