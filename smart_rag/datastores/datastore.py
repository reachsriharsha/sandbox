
from langchain_community.vectorstores import DocArrayInMemorySearch

class DataStore:
    def __init__(self):
        self.data = {}

    def create_vectore_store(self, data, embeddings):
        self.vectorstore = DocArrayInMemorySearch.from_documents(data, embedding=embeddings)
        print("created vector store")
        
    def get_retriever(self):
        return self.vectorstore.as_retriever()
