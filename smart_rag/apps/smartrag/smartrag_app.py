from models import Model
from prompts import Prompt
from dataloader import Loader
from datastores import Datastore


def invoke_smart_rag():
    print("Invoking SmartRag")
    llm = Model("SmartRagModel")
    print("LLM Created")
    model = llm.get_model("ollama","llama2")
    embeddings = llm.get_embeddings()
    parser = llm.get_parser("string")

    promptObj = Prompt("SmartRagTemplate")
    prompt = promptObj.get_prompt()
    prompt.format(context="Here is some context", question="Here is a question")
    print(prompt)

    print("Data loaded starting")

    data_loader = Loader("ip_data/Neptune_V9_1_What_s_New_Guide.pdf")
    pages = data_loader.get_data()
    data_store = Datastore()
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
    