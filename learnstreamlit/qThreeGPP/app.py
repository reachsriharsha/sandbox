import streamlit as st
from utils import *
import constants



def main():
    
     #Fetch  and load 3gpp spec data 
    #read_and_load_3gpp_spec_data()
    #Set title
    st.title('3GPP Specification Assistance') 

  
    ##Captures User Inputs
    question = st.text_input('How can I help you my friend ❓',key="question")  # The box for the text prompt
    #document_count = st.slider('No.Of links to return 🔗 - (0 LOW || 5 HIGH)', 0, 5, 2,step=1)
    #
    submit = st.button("Get Details") 

    if submit:
        qa_bot = query_bot_inst()
        st.write(question)
        st.write("start to ask question to ai model")
        bot_answer = qa_bot({"query": question})
        st.write("Go answer")
        #st.success("Please find the search results :")
        st.write( bot_answer)




if __name__ == "__main__":
    main()
