#import models as Model
from models.models import Model
from prompts.prompts import Prompt
from dataloader.loaders import Loader
from datastores.datastore import DataStore


def invoke_smart_rag():
    print("Invoking SmartRag")
    llm = Model("SmartRagModel")
    print("LLM Created")
    model = llm.get_model("ollama","llama2")
    embeddings = llm.get_embeddings()
    parser = llm.get_parser("string")

    promptObj = Prompt("SmartRagTemplate")
    prompt = promptObj.get_prompt()
    #promptObj.set_prompt_format(context="Here is some context", question="Here is a question")
    print(prompt)

    print("Data loaded starting")

    data_loader = Loader("ip_data/Neptune_V9_1_What_s_New_Guide.pdf")
    pages = data_loader.get_data()
    print(pages)
    data_store = DataStore()
    data_store.create_vectore_store(pages, embeddings)
    retriever = data_store.get_retriever()


    chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
    )

    questions = [
    "What are new features in Neptune?",
    "How load balancing done?",
    ]

    for question in questions:
        print(f"Question: {question}")
        print(f"Answer: {chain.invoke({'question': question})}")
        print()

    
def main():
    invoke_smart_rag()


if __name__  == "__main__":
    main()
    