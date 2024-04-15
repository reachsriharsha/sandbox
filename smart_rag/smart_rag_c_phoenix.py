import os

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3



from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter
import tiktoken
from langchain_community.vectorstores import Chroma
import numpy as np
from langchain.prompts import ChatPromptTemplate
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.load import dumps, loads

from phoenix.trace.llama_index import OpenInferenceTraceCallbackHandler
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument()


# View the traces in the Phoenix UI
session = px.launch_app()
print(f'please launch your phoenix ui at {session.url}')

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

chroma_db_persist_path = "./data/db/chroma"

def load_data_into_vector_db():
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
        #milvus_vector_db = Milvus.from_documents(pages, embedding=embeddings,
        #                                        connection_args={"host": "localhost", "port": 19530 },)

        if os.path.exists(chroma_db_persist_path):
            vector_db = Chroma(embedding_function=embeddings,
                                    persist_directory=chroma_db_persist_path)
        else:
            vector_db = Chroma.from_documents(documents=pages,
                                          embedding=embeddings,
                                          persist_directory=chroma_db_persist_path)
            vector_db.persist()
        
        print("created vector store")
        simple_m_vdb_handle = vector_db
        return simple_m_vdb_handle
    else:
        return simple_m_vdb_handle

def index_data_into_vector_db():
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
        #milvus_vector_db = Milvus.from_documents(pages, embedding=embeddings,
        #        
        if os.path.exists(chroma_db_persist_path):
           vector_db = Chroma(embedding_function=embeddings,
                              persist_directory=chroma_db_persist_path)
        else:
            vector_db = Chroma.from_documents(pages, embedding=embeddings,
                                          persist_directory=chroma_db_persist_path) 
            vector_db.persist()
        print("created vector store")
        index_m_vdb_handle = vector_db
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

    #chain = prpt | modl | prsr
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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

def indexed_rag():
    index_hdl = index_data_into_vector_db()
    retriever = index_hdl.as_retriever(search_kwargs={"k": 1})
    print("Created retriever")

    total = 0
    while True:
        total +=1
        question = input("Enter a question (or 'quit' to quit): ")
        n_tokens = num_tokens_from_string(question, "cl100k_base")
        print(f"Number of tokens in the question: {n_tokens}")
       
        if question == 'quit':
            break
        else:
            #Retrieval of the documents
            docs = retriever.get_relevant_documents(question)
            print(docs[0].metadata)
            #emb = OllamaEmbeddings(model=MODEL)
            #query_result = emb.embed_query(question)
            #doc_result = emb.embed_query(docs.text)
            #similarity = cosine_similarity(query_result, doc_result)
            #print("Cosine Similarity:", similarity)

            #Generation of the answer
            #prompting is used here to generate the answer
            # Prompt
            template = """Answer the question based only on the following context:
            {context}

            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)
            #print(prompt)
            llm = get_model()
            #chain = prompt | llm
            #print(f"Answer: {chain.invoke({'context': docs, 'question': question})}")

            #prompt_hub_rag = hub.pull("rlm/rag-prompt")
            rag_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            print(f"Answer: {rag_chain.invoke({'question':question})}")





    print("Exiting the loop...")
from typing import List

def get_unique_union(documents: List[list]):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return [loads(doc) for doc in unique_docs]


def rag_multi_query():
    index_hdl = index_data_into_vector_db()
    retriever = index_hdl.as_retriever(search_kwargs={"k": 1})
    print("Created retriever")

    total = 0
    while True:
        total +=1
        question = input("Enter a question (or 'quit' to quit): ")
        n_tokens = num_tokens_from_string(question, "cl100k_base")
        print(f"Number of tokens in the question: {n_tokens}")
       
        if question == 'quit':
            break
        else:
            # Prompt
            # Multi Query: Different Perspectives
            template = """You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of the distance-based similarity search. 
            Provide these alternative questions separated by newlines. Original question: {question}"""
            prompt_perspectives = ChatPromptTemplate.from_template(template)
            llm = get_model()
            generate_queries = (
                prompt_perspectives 
                | llm 
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )
            # Retrieve
            retrieval_chain = generate_queries | retriever.map() | get_unique_union
            docs = retrieval_chain.invoke({"question":question})
            print("Total questions generated:",len(docs))
            print(f"Question list:{docs}")

            # RAG
            template = """Answer the following question based on this context:
            {context}
            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)

            final_rag_chain = (
                {"context": retrieval_chain, 
                 "question": itemgetter("question")} 
                | prompt
                | llm
                | StrOutputParser()
            )

            print(f"Answer: {final_rag_chain.invoke({'question':question})}")

    print("Exiting the loop...")


def reciprocal_rank_fusion(results: List[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results

def rag_fusion_search():
    index_hdl = index_data_into_vector_db()
    retriever = index_hdl.as_retriever(search_kwargs={"k": 1})
    print("Created retriever")

    total = 0
    while True:
        total +=1
        question = input("Enter a question (or 'quit' to quit): ")
        n_tokens = num_tokens_from_string(question, "cl100k_base")
        print(f"Number of tokens in the question: {n_tokens}")
       
        if question == 'quit':
            break
        else:
            # Prompt
            # RAG-Fusion: Related
            template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
            Generate multiple search queries related to: {question} \n
            Output (4 queries):"""
            prompt_rag_fusion = ChatPromptTemplate.from_template(template)
            llm = get_model()
            generate_queries = (
                prompt_rag_fusion 
                | llm
                | StrOutputParser() 
                | (lambda x: x.split("\n"))
            )
            # Retrieve
            retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
            docs = retrieval_chain_rag_fusion.invoke({"question": question})
            print(len(docs))

            # RAG
            template = """Answer the following question based on this context:

            {context}       

            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)

            final_rag_chain = (
                {"context": retrieval_chain_rag_fusion, 
                 "question": itemgetter("question")} 
                | prompt
                | llm
                | StrOutputParser()
            )

            print(f"Answer: {final_rag_chain.invoke({'question':question})}")

    print("Exiting the loop...")


def main():

    while True:
        try:
            print("Choose type of RAG:\n")
            print ("1.RAG: Simple\n 2.RAG: Indexed Generative Search\n 3.RAG: Multi query\n 4.RAG Fusion \n5.TBD\n")
            ragType = int(input("Enter your input: "))
            if ragType == 1:
                print("Simple RAG selected")
                v_db = load_data_into_vector_db()
                sample_rag(v_db)
                break
            elif ragType == 2:
                print("Indexed Search selected")
                indexed_rag()
                break
            elif ragType == 3:
                print("Multi Query Search") 
                rag_multi_query()
                break
            elif ragType == 4:
                print("RAG Fusion Search")
                rag_fusion_search()

            else:
                print("Invalid input. Please try again.")
        except Exception as e:
            print(f"Error: {e}")
            continue
        else:
            break

    

if __name__  == "__main__":
    main()