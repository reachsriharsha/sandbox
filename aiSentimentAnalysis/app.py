import streamlit as st
from streamlit_option_menu import option_menu
from aimodule import SentimentAnalysis






def sentiment_analysis():
    st.title("Sentiment Analysis")
    st.write("This is a simple sentiment analysis tool.")
    text = st.text_area("Enter some text")
    if st.button("Analyze"):
        sa = SentimentAnalysis()
        results = sa.predict_sentiment(text)
        st.write(results)
        

def bring_fe():

    option_list = ["Sentiment Analysis"]
    icon_list = ["bi-chat-fill"]

    option_list.append("Help")
    icon_list.append("bi-question-circle-fill")

    with st.sidebar:
        selected = option_menu(menu_title=None,
                                   options=option_list,
                                   icons=icon_list,
                                   menu_icon="🔍",
                                   default_index=0)
    
    if selected == "Sentiment Analysis":
        sentiment_analysis()
    if selected == "Help":
        help()


def main():
    #st.title('AI Porjects')
    bring_fe()


if __name__ == "__main__":
    main()