

import pandas as pd


# ## Ingestion
# 



df= pd.read_csv('data.csv')


# In[4]:


df


# In[5]:


df.columns


# In[6]:


get_ipython().system('wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py')


# In[7]:


documents=df.to_dict(orient='records')


# In[8]:


documents


# In[9]:


import minsearch


# In[10]:


index = minsearch.Index(
    text_fields=['cell_nuclei_type', 'dataset_name', 'tasks', 'models'],
    keyword_fields=["id"]
)


# In[11]:


index.fit(documents)


# ## RAG Flow

# In[15]:


from openai import OpenAI
client = OpenAI()


# In[20]:


query="give me datasets most suitable for Plasma Cell"


# In[21]:


index.search(query)


# In[13]:


import os


# In[14]:


os.environ['OPENAI_API_KEY']='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'


# In[16]:


def search(query):
    boost = {}
    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[17]:


documents[0]


# In[18]:


prompt_template = """
You are a histopathology expert. Answer the QUESTION based on the CONTEXT from our cell nuclei dataset.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

entry_template = """
cell_nuclei_type: {cell_nuclei_type}
dataset_name: {dataset_name}
tasks: {tasks}
models: {models}
""".strip()

def build_prompt(query, search_results):
   

    context = ""
    
    for doc in search_results:
        context = context + entry_template.format(**doc) + "\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


# In[22]:


search_results=search(query)
prompt=build_prompt(query,search_results)


# In[23]:


print(prompt)


# In[349]:


def llm(prompt, model='gpt-4o-mini'):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


# In[350]:


def rag(query, model='gpt-4o-mini'):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    #print(prompt)
    answer = llm(prompt, model=model)
    return answer


# In[275]:


question='What are the segmentation tasks available for the Epithelial cell nuclei type?'
answer=rag(question)
print(answer)


# ## Retrieval Evaluation

# In[312]:


df_question=pd.read_csv('data/ground_truth.csv')


# In[313]:


df_question


# In[314]:


ground_truth=df_question.to_dict(orient='records')


# In[315]:


ground_truth[0]


# In[316]:


def hit_rate(relevance_total):
    cnt = 0

    for line in relevance_total:
        if True in line:
            cnt = cnt + 1

    return cnt / len(relevance_total)
def mrr(relevance_total):
    total_score = 0.0

    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)

    return total_score / len(relevance_total)


# In[317]:


def minsearch_search(query):
    boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[318]:


def evaluate(ground_truth, search_function):
 relevance_total = []

 for q in tqdm(ground_truth):
     doc_id = q['id']
     results = search_function(q)

     # Check for relevance
     relevance = [d['id'] == doc_id for d in results]
     relevance_total.append(relevance)

 return {
     'hit_rate': hit_rate(relevance_total),
     'mrr': mrr(relevance_total),
 }


# In[319]:


from tqdm.auto import tqdm


# In[320]:


evaluate(ground_truth, lambda q: minsearch_search(q['question'])) 


# In[289]:


from tqdm import tqdm

def evaluate(ground_truth, search_function):
    relevance_total = []
    no_results_queries = []
    low_rank_queries = []

    for q in tqdm(ground_truth):
        doc_id = q['id']
        results = search_function(q['question'])
        
        if not results:
            print(f"No results found for query ID: {doc_id}")
            no_results_queries.append(q)
            continue
        
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)
        
        if True in relevance and relevance.index(True) > 9:  # If correct result is not in top 10
            low_rank_queries.append((q, relevance.index(True)))

    hit_rate_value = hit_rate(relevance_total)
    mrr_value = mrr(relevance_total)

    print(f"Queries with no results: {len(no_results_queries)}")
    print(f"Queries with low-ranked correct results: {len(low_rank_queries)}")
    
    return {
        'hit_rate': hit_rate_value,
        'mrr': mrr_value,
        'no_results_queries': no_results_queries,
        'low_rank_queries': low_rank_queries
    }

# Usage
results = evaluate(ground_truth, lambda q: minsearch_search(q))
print(f"Hit Rate: {results['hit_rate']:.4f}")
print(f"MRR: {results['mrr']:.4f}")

# Analysis of problematic queries
print("\nSample of queries with no results:")
for q in results['no_results_queries'][:5]:  # Print first 5
    print(f"ID: {q['id']}, Question: {q['question']}")

print("\nSample of queries with low-ranked correct results:")
for q, rank in results['low_rank_queries'][:5]:  # Print first 5
    print(f"ID: {q['id']}, Rank: {rank}, Question: {q['question']}")


# In[290]:


import numpy as np
from typing import List, Dict, Any

def precision_at_k(relevance: List[bool], k: int) -> float:
    """
    Calculate Precision@k
    
    Args:
    relevance (List[bool]): List of boolean values indicating relevance of each result
    k (int): The 'k' in Precision@k
    
    Returns:
    float: Precision@k score
    """
    if len(relevance) < k:
        return sum(relevance) / len(relevance)
    return sum(relevance[:k]) / k

def average_precision(relevance: List[bool]) -> float:
    """
    Calculate Average Precision
    
    Args:
    relevance (List[bool]): List of boolean values indicating relevance of each result
    
    Returns:
    float: Average Precision score
    """
    if not any(relevance):
        return 0.0
    score = 0.0
    num_relevant = 0
    for i, rel in enumerate(relevance, 1):
        if rel:
            num_relevant += 1
            score += num_relevant / i
    return score / num_relevant

def ndcg(relevance: List[bool], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain
    
    Args:
    relevance (List[bool]): List of boolean values indicating relevance of each result
    k (int, optional): The 'k' in NDCG@k. If None, use full list.
    
    Returns:
    float: NDCG score
    """
    if k is None:
        k = len(relevance)
    
    dcg = sum((rel / np.log2(i + 2) for i, rel in enumerate(relevance[:k])))
    idcg = sum((1 / np.log2(i + 2) for i in range(min(sum(relevance), k))))
    
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_search(ground_truth: List[Dict[str, Any]], search_function: callable) -> Dict[str, float]:
    relevance_total = []
    for i, q in enumerate(ground_truth):
        doc_id = q['id']
        results = search_function(q)
        if not results:
            print(f"No results found for query ID: {doc_id}")
            continue
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)
        if i < 5:  # Print details for the first 5 queries
            print(f"Query {i+1}:")
            print(f"  Document ID: {doc_id}")
            print(f"  Relevance list: {relevance[:10]}")  # First 10 results
            print(f"  Precision@1: {precision_at_k(relevance, 1)}")
            print(f"  Precision@5: {precision_at_k(relevance, 5)}")
            print(f"  NDCG@5: {ndcg(relevance, 5)}")
            print()
    
    hit_rate_value = hit_rate(relevance_total)
    mrr_value = mrr(relevance_total)
    
    metrics = {
        'hit_rate': hit_rate_value,
        'mrr': mrr_value,
        'precision@1': np.mean([precision_at_k(rel, 1) for rel in relevance_total]),
        'precision@5': np.mean([precision_at_k(rel, 5) for rel in relevance_total]),
        'precision@10': np.mean([precision_at_k(rel, 10) for rel in relevance_total]),
        'map': np.mean([average_precision(rel) for rel in relevance_total]),
        'ndcg@5': np.mean([ndcg(rel, 5) for rel in relevance_total]),
        'ndcg@10': np.mean([ndcg(rel, 10) for rel in relevance_total]),
    }
    
    return metrics

# Example usage
results = evaluate_search(ground_truth, lambda q: minsearch_search(q['question']))
print(results)


# In[321]:


df_validation=df_question[:100]
df_test=df_question[100:]


# ## Finding Best Parameters

# In[274]:


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope


# In[322]:


import random

def simple_optimize(param_ranges, objective_function, n_iterations=10):
    best_params = None
    best_score = float('-inf')  # Assuming we're minimizing. Use float('-inf') if maximizing.

    for _ in range(n_iterations):
        # Generate random parameters
        current_params = {}
        for param, (min_val, max_val) in param_ranges.items():
            if isinstance(min_val, int) and isinstance(max_val, int):
                current_params[param] = random.randint(min_val, max_val)
            else:
                current_params[param] = random.uniform(min_val, max_val)
        
        # Evaluate the objective function
        current_score = objective_function(current_params)
        
        # Update best if current is better
        if current_score > best_score:  # Change to > if maximizing
            best_score = current_score
            best_params = current_params
    
    return best_params, best_score


# In[323]:


gt_val = df_validation.to_dict(orient='records')


# In[324]:


evaluate(ground_truth, lambda q: minsearch_search(q['question'])) 


# In[325]:


def minsearch_search(query, boost=None):
    if boost is None:
        boost = {}

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results


# In[326]:


documents[0]


# In[327]:


param_ranges = {
    'cell_nuclei_type': (0.0, 3.0),
    'dataset_name': (0.0, 3.0),
    'tasks': (0.0, 3.0),
    'models': (0.0, 3.0),
}

def objective(boost_params):
    def search_function(q):
        return minsearch_search(q['question'], boost_params)

    results = evaluate(gt_val, search_function)
    return results['mrr']


# In[329]:


simple_optimize(param_ranges, objective, n_iterations=20)


# In[330]:


def minsearch_improved(query):
    boost = {
        'cell_nuclei_type': 1.69,
        'dataset_name': 0.98,
        'tasks': 1.34,
        'models': 2.75,
    }

    results = index.search(
        query=query,
        filter_dict={},
        boost_dict=boost,
        num_results=10
    )

    return results

evaluate(ground_truth, lambda q: minsearch_improved(q['question']))                                        


# In[331]:


prompt2_template = """
You are an expert evaluator for a RAG system.
Your task is to analyze the relevance of the generated answer to the given question.
Based on the relevance of the generated answer, you will classify it
as "NON_RELEVANT", "PARTLY_RELEVANT", or "RELEVANT".

Here is the data for evaluation:

Question: {question}
Generated Answer: {answer_llm}

Please analyze the content and context of the generated answer in relation to the question
and provide your evaluation in parsable JSON without using code blocks:

{{
  "Relevance": "NON_RELEVANT" | "PARTLY_RELEVANT" | "RELEVANT",
  "Explanation": "[Provide a brief explanation for your evaluation]"
}}
""".strip()


# In[332]:


len(ground_truth)


# In[336]:


record = ground_truth[0]
question=record['question']
answer_llm=rag(question)


# In[337]:


print(answer_llm)


# In[338]:


prompt = prompt2_template.format(question=question, answer_llm=answer_llm)
print(prompt)


# In[339]:


llm(prompt)


# In[340]:


import json
df_sample = df_question.sample(n=200, random_state=1)
sample = df_sample.to_dict(orient='records')
evaluations = []


# In[341]:


sample


# In[342]:


for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question) 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)

    evaluations.append((record, answer_llm, evaluation))


# In[343]:


df_eval = pd.DataFrame(evaluations, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']


# In[345]:


df_eval.relevance.value_counts()


# In[347]:


df_eval.to_csv('data/rag-eval-gpt-4o-mini.csv', index=False)


# In[348]:


df_eval[df_eval.relevance == 'NON_RELEVANT']


# In[351]:


evaluations_gpt4o = []

for record in tqdm(sample):
    question = record['question']
    answer_llm = rag(question, model='gpt-4o') 

    prompt = prompt2_template.format(
        question=question,
        answer_llm=answer_llm
    )

    evaluation = llm(prompt)
    evaluation = json.loads(evaluation)
    
    evaluations_gpt4o.append((record, answer_llm, evaluation))


# In[352]:


df_eval = pd.DataFrame(evaluations_gpt4o, columns=['record', 'answer', 'evaluation'])

df_eval['id'] = df_eval.record.apply(lambda d: d['id'])
df_eval['question'] = df_eval.record.apply(lambda d: d['question'])

df_eval['relevance'] = df_eval.evaluation.apply(lambda d: d['Relevance'])
df_eval['explanation'] = df_eval.evaluation.apply(lambda d: d['Explanation'])

del df_eval['record']
del df_eval['evaluation']
df_eval.relevance.value_counts()


# In[353]:


df_eval.relevance.value_counts(normalize=True)


# In[354]:


df_eval.to_csv('data/rag-eval-gpt-4o.csv', index=False)


# In[ ]:




