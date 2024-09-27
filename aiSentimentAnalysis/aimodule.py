from langchain_community.llms import Ollama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_ollama import ChatOllama

#from langchain_experimental.llms.ollama_functions import OllamaFunctions




class SentimentResult(BaseModel):
    """Sentiment Result"""
    sentiment: str = Field(..., description="Sentiment of the text")
    score: float = Field(..., description="Sentiment score of the text")
    summary: str = Field(..., description="Summary of the text")
    input_tokens: int = Field(..., description="Number of tokens in the input text")
    output_tokens: int = Field(..., description="Number of tokens in the output text")
    total_tokens: int = Field(..., description="Number of total tokens")






class SentimentAnalysis:
    def __init__(self):
        pass

    def predict_sentiment(self, text):

        llm = ChatOllama(model="llama3.1:latest",temperature=0,verbose=True)
        #llm = ChatOllama(model="tinyllama:latest",temperature=0,verbose=True)
        #llm = OllamaFunctions(model="llama3")
        messages = [
            ("system", "You are a helpful assistant that summerizes the given text"),
            ("human", text)
        ]

        structured_response = llm.with_structured_output(SentimentResult)

        #prediction = llm.invoke(messages)
        prediction = structured_response.invoke(messages)

        print(type(prediction))


        print(prediction) 
        return prediction
        