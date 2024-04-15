
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser


class Model():
    def __init__(self, model_name):
        self.model_name = model_name

    def __str__(self):
        return self.model_name
    '''
    Place holder
    '''
    def save(self):
        print(f"Saving model {self.model_name} to disk")

    def get_model(self,service,name):
        #switch case to return different models
        self.service = service
        self.llm_name = name
        if service == "ollama":
            if name == "llama2":
                return Ollama(model="llama2")
            elif name == "non_existing_model":
                return None
            else:
                return None
        elif service == "openai":
            return None
        else:
            return None
        
    def get_embeddings(self):

        if self.service == "ollama":
            return OllamaEmbeddings(model=self.llm_name)
        elif self.service == "openai":
            return None
        else:
            return None


    def get_parser(self, name):
        if name == "string":
            return StrOutputParser()
        elif name == "non_existing_model":
            return None
        else:
            return None
    

