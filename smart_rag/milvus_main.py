import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import Milvus
from operator import itemgetter
import tiktoken


load_dotenv()

simple_rag_data_load = False 
simple_m_vdb_handle = None
indexed_rag_data_load = False
index_m_vdb_handle = None


MY_API_KEY = os.getenv("MY_API_KEY")
MODEL = "llama2"
print(MY_API_KEY)

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = 'ls__4bcd38a01e14462692bd90229091572a'


def load_data_into_milvus():
    global simple_rag_data_load
    global simple_m_vdb_handle
    if simple_rag_data_load == False:
        simple_rag_data_load = True
        embeddings = OllamaEmbeddings(model=MODEL)
        loader = PyPDFLoader("ip_data/Neptune_V9_1_What_s_New_Guide.pdf")
        pages = loader.load_and_split()
        #print(pages)
        print("Loading document completed")
        #vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        milvus_vector_db = Milvus.from_documents(pages, embedding=embeddings,
                                                 connection_args={"host": "localhost", "port": 19530 },)
        print("created vector store")
        simple_m_vdb_handle = milvus_vector_db
        return simple_m_vdb_handle
    else:
        return simple_m_vdb_handle

def index_data_into_milvus():
    global indexed_rag_data_load
    global index_m_vdb_handle
    if indexed_rag_data_load == False:
        indexed_rag_data_load = True
        embeddings = OllamaEmbeddings(model=MODEL)
        loader = PyPDFLoader("ip_data/Neptune_V9_1_What_s_New_Guide.pdf")
        pages = loader.load_and_split()
        #print(pages)
        print("Loading document completed")
        #vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
        milvus_vector_db = Milvus.from_documents(pages, embedding=embeddings,
                                                 connection_args={"host": "localhost", "port": 19530 },)
        print("created vector store")
        index_m_vdb_handle = milvus_vector_db
        return index_m_vdb_handle
    else:
        return index_m_vdb_handle

def get_model():
    model = Ollama(model=MODEL)
    #print(model.invoke("Tell me a joke"))
    return model
    

def get_prompt():
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}
    
    Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    prompt.format(context="Here is some context", question="Here is a question")
    print(prompt)
    return prompt

def get_parser():
    parser = StrOutputParser()
    return parser





def sample_rag(milvus_v_db):
    modl = get_model()
    prpt = get_prompt()
    prsr = get_parser()
    rtrv = milvus_v_db.as_retriever()

    chain = prpt | modl | prsr
    chain = (
        {
            "context": itemgetter("question") | rtrv,
            "question": itemgetter("question"),
        }
        | prpt
        | modl
        | prsr
    )

    total = 0
    while True:
        total +=1
        question = input("Enter a question (or 'quit' to quit): ")
        if question == 'quit':
            break
        else:
            print(f"Current question count: {total}")
            print(f"Question: {question}")
            print(f"Answer: {chain.invoke({'question': question})}")
            print()


    print("Exiting the loop...")


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def indexed_rag():
    milvus_index_hdl = index_data_into_milvus()
    retriever = milvus_index_hdl.as_retriever(search_kwargs={"k": 1})
    print("Created retriever")

    total = 0
    while True:
        total +=1
        question = input("Enter a question (or 'quit' to quit): ")
        if question == 'quit':
            break
        else:
            docs = retriever.get_relevant_documents(question)
            print(docs)

    print("Exiting the loop...")



def main():

    while True:
        try:
            print("Choose type of RAG:\n")
            print ("1. Simple RAG\n 2.Indexed Search 3.TBD\n")
            ragType = int(input("Enter your input: "))
            if ragType == 1:
                print("Simple RAG selected")
                milvus_v_db = load_data_into_milvus()
                sample_rag(milvus_v_db)
                break
            elif ragType == 2:
                print("Indexed Search selected")
                indexed_rag()
                break
            else:
                print("Invalid input. Please try again.")
        except Exception as e:
            print(f"Error: {e}")
            continue
        else:
            break

    

if __name__  == "__main__":
    main()