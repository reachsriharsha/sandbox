

import os
#from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from operator import itemgetter

class Loader():
    def __init__(self, path):
        self.doc_path = path
        self.pages = None

    def load_documents(self):
        self.loader = PyPDFLoader(self.doc_path)
        self.pages = self.loader.load_and_split()

    def get_data(self):
        return self.pages
    

    