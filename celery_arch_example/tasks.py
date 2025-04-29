from celery_config import celery_app
#import pandas as pd
from datetime import datetime, timedelta
from celery import Task, Celery, chord, chain, group
from celery.result import AsyncResult

from typing import List, Dict
import time

#celery -A tasks worker --loglevel=info
'''
Scaling techniques
celery -A your_app worker --concurrency=8 -l INFO

app.conf.task_routes = {
    'your_app.query_knowledge_base': {'queue': 'kb_queries'},
    'your_app.aggregate_answers': {'queue': 'processing'},
    'your_app.create_summary': {'queue': 'processing'},
    'your_app.determine_compliance': {'queue': 'processing'},
}

# Task retry settings
app.conf.task_acks_late = True  # Tasks are acknowledged after completion
app.conf.task_reject_on_worker_lost = True  # Tasks are requeued if worker dies

# Update state to mark as processing (within the task)
self.update_state(state='PROGRESS', meta={'kb_name': kb_name, 'progress': 0})
self.update_state(state='PROGRESS', meta={'kb_name': kb_name, 'progress': 50})

# In your Celery configuration
app.conf.task_default_rate_limit = '1000/m'  # Limit rate of tasks
app.conf.worker_prefetch_multiplier = 1  # Don't prefetch too many tasks
app.conf.worker_max_tasks_per_child = 200  # Restart workers periodically to prevent memory leaks


# Start workers with specific memory limits and concurrency
celery -A your_app worker -Q kb_queries --concurrency=16 -l INFO --max-memory-per-child=512000


'''
from dotenv import load_dotenv
load_dotenv()

class MetaTask(Task):

    def __call__(self, *args, **kwargs):
        pass
    def before_start(self, task_id, args, kwargs):
        print(f"[Task] {task_id} started with args: {args} and kwargs: {kwargs}")
    
    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        print(f"[Task] {task_id} finished with status: {status} and retval: {retval}")

        if self.task.name == 'kb_addition':
            print(f"[Task] {task_id} finished with status: {status} and retval: {retval}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print(f"[Task] {task_id} failed with exc: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        print(f"[Task] {task_id} retried with exc: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        print(f"[Task] {task_id} succeeded with retval: {retval}")

    def update_state(self, task_id=None, state=None, meta=None, **kwargs):
        print(f"[Task] {task_id} updated with state: {state} and meta: {meta}")

    def on_success(self, retval, task_id, args, kwargs):
        print(f"[Task] {task_id} succeeded with retval: {retval}")
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        print(f"[Task] {task_id} failed with exc: {exc}")



@celery_app.task
def get_answer(kb_name:str, 
               kb_tag: str,
               question: str):
    try:
        
        return {
            'status': 'completed',
            'data': 'answer'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }


#Knowledge Base Queries (Parallel) → Aggregation → Summarization → Compliance Determination


app = Celery('compliance_pipeline', broker='redis://localhost:6379/0')

# Task 1: Query a specific knowledge base
@app.task
def query_knowledge_base(question: str, kb_name: str) -> Dict:

    """Query a single knowledge base and return the answer."""
    # Implement your knowledge base query logic here
    # ...
    print(f'Quering the knowledge base started from {kb_name} for question: {question}')
    return {
        'kb_name': kb_name,
        'answer': f"Answer from {kb_name} for question: {question}"
    }

# Task 2: Aggregate all answers
@app.task
def aggregate_answers(answers: List[Dict]) -> Dict:
    """Collect and organize all answers from different knowledge bases."""
    aggregated = {
        'question': answers[0].get('question', ''),
        'kb_answers': {}
    }
    print("**" * 20)

    for answer in answers:
        print(f'Answer: {answer}')

    #print(f'Aggregating answers for question: {answers}')
    print("**" * 20)

    
    for answer in answers:
        aggregated['kb_answers'][answer['kb_name']] = answer['answer']
    
    return aggregated

# Task 3: Summarize the aggregated answers
@app.task
def create_summary(aggregated_data: Dict) -> Dict:
    """Create a summary of all the answers."""
    # Implement your summarization logic here
    # This could use NLP techniques, extraction, etc.
    
    print(f'Summarizing answers for question: {aggregated_data["question"]}')
    summary = "Summary of answers from all knowledge bases: ..."
    aggregated_data['summary'] = summary
    return aggregated_data

# Task 4: Determine compliance
@app.task
def determine_compliance(summarized_data: Dict) -> Dict:
    """Determine if the data indicates compliance or non-compliance."""
    # Implement your compliance determination logic
    # This could be based on rules, ML model, etc.
    
    # Example logic
    is_compliant = True  # Your actual determination logic here
    
    print(f'Determining compliance for question: {summarized_data["question"]}')
    summarized_data['compliance_result'] = 'compliant' if is_compliant else 'not compliant'
    return summarized_data


# Function to check status
def get_compliance_status(task_id):
    """Check the status of a compliance workflow."""
    print(f'Checking status of task: {task_id}')
    result = AsyncResult(task_id)
    
    if result.ready():
        if result.successful():
            return {
                'status': 'COMPLETED',
                'result': result.get()
            }
        else:
            return {
                'status': 'FAILED',
                'error': str(result.result)
            }
    else:
        return {
            'status': 'PENDING',
            'info': result.info  # May contain progress information if provided
        }
    
# Main workflow function
def check_compliance(question: str, kb_list: List[str]):
    """
    Start the compliance checking workflow for a given question
    against multiple knowledge bases.
    """
    # Step 1: Create parallel tasks for querying each knowledge base
    kb_tasks = [query_knowledge_base.s(question, kb_name) for kb_name in kb_list]

    print(f'List of tasks: {kb_tasks}')
    
    # Step 2-4: Chain the aggregation, summarization, and compliance determination
    workflow = chord(
        header=group(kb_tasks),
        body=chain(
            aggregate_answers.s(),
            create_summary.s(),
            determine_compliance.s()
        )
    )
    
    # Execute the workflow
    result = workflow.apply_async()
    return get_compliance_status(result.id)
    

