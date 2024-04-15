import os
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from operator import itemgetter


load_dotenv()

MY_API_KEY = os.getenv("MY_API_KEY")
MODEL = "llama2"
print(MY_API_KEY)


model = Ollama(model=MODEL)
#print(model.invoke("Tell me a joke"))
embeddings = OllamaEmbeddings(model=MODEL)

parser = StrOutputParser()

#chain = model | parser 
#print(chain.invoke("Tell me a nice joke"))

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""
prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")
print(prompt)



chain = prompt | model | parser

#print(chain.invoke({"context": "My parents named me Santiago", "question": "What's your name'?"}))

loader = PyPDFLoader("ip_data/Neptune_V9_1_What_s_New_Guide.pdf")
pages = loader.load_and_split()
#print(pages)
print("Loading document completed")
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
print("created vector store")
retriever = vectorstore.as_retriever()
#print(retriever.invoke("can you list new features in release v9.1?"))

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)

#questions = [
#    "What are new features in Neptune?",
#    "How load balancing done?",
#    "is it possible to bypass Authentication?",
#    "What is syslog?",
#    "What does NPT-1012D include?",
#    "What are new features in Neptune V9.1?",
#]

questions = [
    "What are new features in Neptune?",
]

for question in questions:
    print(f"Question: {question}")
    print(f"Answer: {chain.invoke({'question': question})}")
    print()
