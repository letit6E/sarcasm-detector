import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.badges import badge
from classifiers.sarcasm_classifier import SarcasmClassifier

def main_page(classifier):
    st.subheader("Main page")
    text = st.text_area("Enter the text to analyze:")
    if st.button("Analyze"):
     with st.spinner("Processing..."):
        result = classifier.predict(text)
        st.balloons()
        if result:
              st.success("Sarcasm detected!")
        else:
              st.error("Sarcam is not detected!")

def about_page():
    badge(type="github", name="letit6e/sarcasm-detector")
    with open('about.md', 'r', encoding='utf-8') as f:
        html_string = f.read()
    
    st.markdown(html_string, unsafe_allow_html=True)

@st.cache
def load_model(name):
    classifier = SarcasmClassifier.from_hf(name)
    return classifier

def main():
    PAGE_CONFIGURATION = {
        "page_title": "Sarcasm Detector",
        "page_icon": ":smile:",
        "layout": "centered"
    }
    st.set_page_config(**PAGE_CONFIGURATION)
    classifier = load_model("jkhan447/sarcasm-detection-Bert-base-uncased")
    
    with st.sidebar:
        selected = option_menu(
            menu_title = "Main Menu",
            options = ["Home","About"],
            icons = ["house","activity"],
            menu_icon = "cast",
            default_index = 0
        )  
    
    if selected == "Home":
        main_page(classifier)
    if selected == "About":
        about_page()


if __name__ == '__main__':
    main()