import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io import output_file, show

# Функция для предупреждения о радости
def ballons():
    st.balloons()

PAGE_CONFIGURATION = {"page_title":"Моя 2-страницная страница","page_icon":":smile:","layout":"centered"}
st.set_page_config(**PAGE_CONFIGURATION)

def main():

    st.sidebar.title('Меню')
    if st.sidebar.button('Главная'):
        st.subheader("Главная")
        your_text = st.text_input("Введите текст", "")
        if st.button("Отправить"):
            st.write("Ваш текст: ", your_text)
            ballons()

    if st.sidebar.button("Страница EDA"):
         st.subheader("Страница EDA")

        # Загрузка примера данных
         df = sns.load_dataset('titanic')

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


if __name__ == '__main__':
 main()