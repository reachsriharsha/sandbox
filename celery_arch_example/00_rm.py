from celery import Celery, chord, chain, group
from celery.result import AsyncResult
from typing import List, Dict, Any
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Celery app
app = Celery('compliance_pipeline', 
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/1')

# Configure task routes for better distribution
app.conf.task_routes = {
    'compliance_pipeline.query_knowledge_base': {'queue': 'kb_queries'},
    'compliance_pipeline.aggregate_answers': {'queue': 'processing'},
    'compliance_pipeline.create_summary': {'queue': 'processing'},
    'compliance_pipeline.determine_compliance': {'queue': 'processing'},
}

# Task retry settings
app.conf.task_acks_late = True  # Tasks are acknowledged after completion
app.conf.task_reject_on_worker_lost = True  # Tasks are requeued if worker dies

# Task 1: Query a specific knowledge base
@app.task(bind=True, max_retries=3, name='compliance_pipeline.query_knowledge_base')
def query_knowledge_base(self, question: str, kb_name: str) -> Dict[str, Any]:
    """Query a single knowledge base and return the answer."""
    logger.info(f"Querying {kb_name} for question: {question}")
    
    try:
        # Update state to mark as processing
        self.update_state(state='PROGRESS', meta={'kb_name': kb_name, 'progress': 0})
        
        # Simulate knowledge base query with potential failure
        if kb_name == "problematic_kb" and self.request.retries < 2:
            logger.warning(f"Simulated failure for {kb_name}")
            raise Exception("Simulated knowledge base query failure")
        
        # Simulate processing time
        time.sleep(2)
        
        # Update progress
        self.update_state(state='PROGRESS', meta={'kb_name': kb_name, 'progress': 50})
        
        # Simulate more processing
        time.sleep(1)
        
        # Return the result
        return {
            'kb_name': kb_name,
            'question': question,
            'answer': f"Answer from {kb_name} for question: {question}",
            'timestamp': time.time()
        }
    except Exception as exc:
        logger.error(f"Error querying {kb_name}: {str(exc)}")
        # Retry with exponential backoff
        self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

# Task 2: Aggregate all answers
@app.task(name='compliance_pipeline.aggregate_answers')
def aggregate_answers(answers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collect and organize all answers from different knowledge bases."""
    logger.info(f"Aggregating answers from {len(answers)} knowledge bases")
    
    # Extract the original question from the first answer
    question = answers[0].get('question', 'Unknown question')
    
    aggregated = {
        'question': question,
        'kb_answers': {},
        'kb_count': len(answers),
        'aggregation_timestamp': time.time()
    }
    
    for answer in answers:
        aggregated['kb_answers'][answer['kb_name']] = answer['answer']
    
    return aggregated

# Task 3: Summarize the aggregated answers
@app.task(name='compliance_pipeline.create_summary')
def create_summary(aggregated_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of all the answers."""
    logger.info(f"Creating summary for question: {aggregated_data['question']}")
    
    # Simulate summary creation
    time.sleep(2)
    
    # Example summary logic
    answers = list(aggregated_data['kb_answers'].values())
    summary = f"Summary based on {len(answers)} knowledge bases: " + \
              f"Combined insights from various sources regarding '{aggregated_data['question']}'."
    
    aggregated_data['summary'] = summary
    aggregated_data['summary_timestamp'] = time.time()
    
    return aggregated_data

# Task 4: Determine compliance
@app.task(name='compliance_pipeline.determine_compliance')
def determine_compliance(summarized_data: Dict[str, Any]) -> Dict[str, Any]:
    """Determine if the data indicates compliance or non-compliance."""
    logger.info(f"Determining compliance for: {summarized_data['question']}")
    
    # Simulate compliance determination
    time.sleep(1)
    
    # Example determination logic - could be more complex in reality
    is_compliant = True
    for answer in summarized_data['kb_answers'].values():
        if "non-compliant" in answer.lower() or "violation" in answer.lower():
            is_compliant = False
            break
    
    summarized_data['compliance_result'] = 'compliant' if is_compliant else 'not compliant'
    summarized_data['determination_timestamp'] = time.time()
    
    return summarized_data

# Main workflow function
def check_compliance(question: str, kb_list: List[str]):
    """Start the compliance checking workflow for a given question."""
    logger.info(f"Starting compliance check for: {question}")
    
    # Step 1: Create parallel tasks for querying each knowledge base
    kb_tasks = [query_knowledge_base.s(question, kb_name) for kb_name in kb_list]
    
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
    return result.id

# Batch processing function
@app.task(name='compliance_pipeline.batch_compliance_check')
def batch_compliance_check(questions: List[str], kb_list: List[str]):
    """Process a batch of compliance questions."""
    logger.info(f"Starting batch compliance check for {len(questions)} questions")
    
    results = {}
    
    for i, question in enumerate(questions):
        # Log progress
        logger.info(f"Processing question {i+1}/{len(questions)}: {question}")
        
        # Start the workflow for this question
        task_id = check_compliance(question, kb_list)
        
        # Store the task ID
        results[question] = task_id
    
    return results

# Utility function to check status of a batch
def check_batch_status(task_ids: Dict[str, str]):
    """Check the status of a batch of compliance checks."""
    logger.info(f"Checking status of {len(task_ids)} tasks")
    
    statuses = {}
    
    for question, task_id in task_ids.items():
        result = AsyncResult(task_id)
        
        if result.ready():
            if result.successful():
                statuses[question] = {
                    'status': 'COMPLETED',
                    'result': result.get()
                }
            else:
                statuses[question] = {
                    'status': 'FAILED',
                    'error': str(result.result)
                }
        else:
            # Check if it's a chord
            if result.parent:
                # Try to get status of header group
                parent = result.parent
                if hasattr(parent, 'children'):
                    child_statuses = []
                    for child_id in parent.children:
                        child = AsyncResult(child_id)
                        child_statuses.append({
                            'id': child_id,
                            'status': child.status
                        })
                    statuses[question] = {
                        'status': 'IN_PROGRESS',
                        'header_tasks': child_statuses
                    }
                else:
                    statuses[question] = {'status': 'PENDING'}
            else:
                statuses[question] = {'status': 'PENDING'}
    
    return statuses

# Example usage
if __name__ == "__main__":
    # List of questions to process
    questions = [
        "Is our data retention policy compliant with GDPR?",
        "Do our security practices meet ISO 27001 standards?",
        "Are we compliant with industry regulations for financial reporting?"
    ]
    
    # List of knowledge bases to query
    kb_list = [
        "legal_kb",
        "security_kb",
        "regulatory_kb",
        "industry_standards_kb"
    ]
    
    # Option 1: Process questions one by one
    individual_results = {}
    for question in questions:
        task_id = check_compliance(question, kb_list)
        individual_results[question] = task_id
    
    # Option 2: Process all questions in a batch task
    batch_task = batch_compliance_check.delay(questions, kb_list)
    batch_results = batch_task.get()  # This blocks until the batch task completes
    
    # Check statuses (non-blocking example)
    print("Initial statuses:")
    statuses = check_batch_status(batch_results)
    
    # Wait a bit and check again
    time.sleep(10)
    print("Updated statuses:")
    statuses = check_batch_status(batch_results)