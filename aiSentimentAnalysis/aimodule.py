from langchain_community.llms import Ollama
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.llms.ollama_functions import OllamaFunctions




class SentimentResult(BaseModel):
    """Sentiment Result"""
    sentiment: str = Field(..., description="Sentiment of the text")
    score: float = Field(..., description="Sentiment score of the text")





class SentimentAnalysis:
    def __init__(self):
        pass

    def predict_sentiment(self, text):

        #llm = Ollama(model="llama3")
        llm = OllamaFunctions(model="llama3")
        messages = [
            ("system", "You are a helpful assistant that summerizes the given context"),
            ("human", text)
        ]

        structured_response = llm.with_strucured_output(SentimentResult)

        prediction = structured_response.invoke(messages)

        print(f"AI Response: {prediction}") 
        return prediction
        