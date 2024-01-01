import os

from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    )
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DATA_PATH: str = os.path.join(ABS_PATH, "specdata")
DB_DIR: str = os.path.join(ABS_PATH, "db")

prompt_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.
The example of your response should be:

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def read_and_load_3gpp_spec_data():
    '''
        Creates the vector databse using document loaders and embeddings.

        input files from "specdata directory"
        stores all data into "db" directory
        
    '''
    pdf_loader = DirectoryLoader("specdata/", glob="**/*.pdf",loader_cls=PyPDFLoader)
    text_loader = DirectoryLoader("specdata/", glob="**/*.txt", loader_cls=TextLoader)

    all_loaders = [pdf_loader, text_loader]

    #Load the specs

    loaded_documents = []
    for loader in all_loaders:
        loaded_documents.extend(loader.load())


    #Split the specs into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    chunks = splitter.split_documents(loaded_documents)

    #Create embeddings instance
    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    #Create and load the vector database persistantly

    vector_database = Chroma.from_documents(
        documents=chunks,
        embedding=hf_embeddings,
        persist_directory=DB_DIR,
    )
    vector_database.persist()


def load_model(
    model_path="/home/sharsha/src/agiclass/models/llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    max_new_tokens=1024,
    temperature=0.7,
):
    """
    Load a locally downloaded model.

    Parameters:
        model_path (str): The path to the model to be loaded.
        model_type (str): The type of the model.
        max_new_tokens (int): The maximum number of new tokens for the model.
        temperature (float): The temperature parameter for the model.

    Returns:
        CTransformers: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        SomeOtherException: If the model file is corrupt.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    # Additional error handling could be added here for corrupt files, etc.

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
       # max_new_tokens=max_new_tokens,  # type: ignore
        max_new_tokens=1024,  # type: ignore
        temperature=temperature,  # type: ignore
    )

    return llm

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    return prompt

def create_retrieval_qa_chain(llm, prompt, db):
    """
    Creates a Retrieval Question-Answering (QA) chain using a given language model, prompt, and database.

    This function initializes a RetrievalQA object with a specific chain type and configurations,
    and returns this QA chain. The retriever is set up to return the top 3 results (k=3).

    Args:
        llm (any): The language model to be used in the RetrievalQA.
        prompt (str): The prompt to be used in the chain type.
        db (any): The database to be used as the retriever.

    Returns:
        RetrievalQA: The initialized QA chain.
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def query_bot_inst(
    model_name = "sentence-transformers/all-MiniLM-L6-v2",
    persist_dir = DB_DIR,
    device = "cpu",
):
    if not os.path.exists(persist_dir):
        raise ValueError(f"Persist directory {persist_dir} does not exist.")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
        )
    except ValueError as e:
        raise ValueError(f"Could not load embeddings: {e}")
    
    print("Embeddings loaded successfully...")
    db = Chroma(persist_directory=persist_dir,embedding_function=embeddings)    

    try:
        llm = load_model()  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")
    
    print("Model loaded successfully...")

    qa_prompt = (
        set_custom_prompt()
    )  # Assuming this function exists and works as expected

    print("Prompt set successfully...  ")
    try:
        qa = create_retrieval_qa_chain(
            llm=llm, prompt=qa_prompt, db=db
        )  # Assuming this function exists and works as expected
    except Exception as e:
        raise Exception(f"Failed to create retrieval QA chain: {str(e)}")
    print("Retrieval QA chain created successfully...  ")
    return qa