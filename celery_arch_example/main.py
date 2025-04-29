

import tasks
from dotenv import load_dotenv
load_dotenv()



def execute_compliance():


    #read xls file into the python pandas

    '''
    import pandas as pd

    xls_file = pd.ExcelFile("questions.xlsx")
    for sheet_name in xls_file.sheet_names:
        sheet = xls_file.parse(sheet_name)
        for index, row in sheet.iterrows():
            question = row["question"]
            if not pd.isna(question):
                print(f"Question from sheet {sheet_name}, row {index}: {question}")
                
    '''        
            

            

    
    got_results =  tasks.check_compliance("What is the capital of France?", ["wikipedia", "wikidata"])

    # This will block until the entire workflow completes
    #final_result = got_results.get()

    

    print("**" * 20)
    print(f'{got_results}')
    print("**" * 20)


def main():
    print("Hello, World!")
    execute_compliance()


if __name__ == '__main__':
    main()