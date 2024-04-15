
import os
#import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = "ls__7f45a38c637741b4ad0e27f103281c6c"

import random

# Vector parameters
_DIM = 128
_INDEX_FILE_SIZE = 32  # max file size of stored index


def load_data():
    
    pass

def insert(collection, num, dim):
    data = [
        [str(i) for i in range(num)],
        [[random.random() for _ in range(dim)] for _ in range(num)],
        [random.randint(1, 10000) for _ in range(num)],
        [random.random() for _ in range(num)],
    ]
    return data[3]

    
def main():
    #load_data()
    inserted_data = insert("test", 10000, _DIM)    
    print(f"Inserted data{inserted_data}")


if __name__  == "__main__":
    main()
    
    