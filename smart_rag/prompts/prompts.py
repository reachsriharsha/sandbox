from langchain.prompts import PromptTemplate


class Prompt():
    def __init__(self, template):
        self.template_name = template
        #self.prompt = None

    def get_prompt(self):
        template = """
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        self.prompt = PromptTemplate.from_template(template)
        self.prompt.format(context="Here is some context", question="Here is a question")
        return self.prompt
    
    #def set_prompt_format(self, context, question):
    #    self.prompt.format(context=context, question=question)
    #    return self.prompt

    def add_prompt(self):
        pass
    def get_prompt(self):
        pass    
    def remove_prompt(self):
        pass
