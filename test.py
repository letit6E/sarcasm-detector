import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io import output_file, show
import time, random
from streamlit_option_menu import option_menu

# Функция для предупреждения о радости
def ballons():
    st.balloons()

PAGE_CONFIGURATION = {"page_title":"Моя 2-страницная страница","page_icon":":smile:","layout":"centered"}
st.set_page_config(**PAGE_CONFIGURATION)

def preprocess(text):
        new_text = [
        ]
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

def detect_sarcasm(text, model, tokenizer):
    tokenized_text = tokenizer([preprocess(text)], padding=True, truncation=True, max_length=4096, return_tensors="pt")
    output = model(**tokenized_text)
    probs = output.logits.softmax(dim=-1).tolist()[0]
    prediction = 1 if probs[0] < 0.5 else 0
    return bool(prediction)

def main_page(model, tokenizer):
      st.subheader("Главная")
      text = st.text_area("Enter the text to analyze:")
      if st.button("Analyze"):
         with st.spinner("Processing..."):
            result = detect_sarcasm(text, model, tokenizer)
            st.balloons()
            if result:
                  st.success("Sarcasm detected!")
            else:
                  st.error("Sarcam is not detected!")

def eda_page():
         st.subheader("Страница EDA")

        # Загрузка примера данных
         df = sns.load_dataset('titanic')
         
         st.write(df.describe())
         st.write(df)
         st.markdown("## Гистограммы")
         for col in ['age', 'fare']:
               fig, ax = plt.subplots()
               ax = sns.histplot(data=df, x=col, kde=False)
               st.pyplot(fig)

         st.markdown("## Соотношение выживших/погибших")
         fig, ax = plt.subplots()
         ax = sns.countplot(data=df, x='survived')
         st.pyplot(fig)

         st.markdown("## Соотношение выживших по классу")
         fig, ax = plt.subplots()
         gender_survived = df.groupby(['class', 'survived']).size().unstack()
         gender_survived.plot(kind='bar', stacked=True, ax=ax)
         st.pyplot(fig)

         st.markdown("## График соотношения выживших по полу")
         fig, ax = plt.subplots()
         ax = sns.countplot(data=df, x='sex', hue='survived')
         st.pyplot(fig)

def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import re

    tokenizer = AutoTokenizer.from_pretrained("jkhan447/sarcasm-detection-Bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("jkhan447/sarcasm-detection-Bert-base-uncased")

    with st.sidebar:
        selected = option_menu(
        menu_title = "Main Menu",
        options = ["Home","Analysis"],
        icons = ["house","activity"],
        menu_icon = "cast",
        default_index = 0,
        #orientation = "horizontal",
        )  

    if selected=="Home":
        main_page(model, tokenizer)
    if selected=="Analysis":
        with open('Untitled1.md', 'r', encoding='utf-8') as f:
            html_string = f.read()

        st.markdown(html_string, unsafe_allow_html=True)


if __name__ == '__main__':
 main()