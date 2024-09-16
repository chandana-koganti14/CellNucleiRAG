import pandas as pd

import requests

#df = pd.read_csv("/Users/saichandanakoganti/CellNucleiRAG/notebooks/data/ground_truth.csv")
#question = df.sample(n=1).iloc[0]['question']

#print("question: ", question)

url = "http://127.0.0.1:5001/question"

question=("What are the segmentation tasks available for the Epithelial cell nuclei type?")
data = {"question": question}

response = requests.post(url, json=data)
print(response.content)

print(response.json())