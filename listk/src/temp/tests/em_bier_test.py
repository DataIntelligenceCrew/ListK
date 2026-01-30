from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from semtopk import MultiPivot
import time
import pandas as pd
import json
from sentence_transformers import SentenceTransformer

dataset = "scifact"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = "/localdisk/shin,jason-HonorsThesis/datasets/scifact_bier"
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_folder=data_path ).load(split="test")

document_set = []
document_id = []
for c in corpus:
    document_set.append(str(corpus[f'{c}']))
    document_id.append(c)

query_doc = []
query_id = []
for q in queries:
    query_doc.append(str(queries[f'{q}']))
    query_id.append(q)

results = pd.read_csv("/localdisk/shin,jason-HonorsThesis/solicedb/src/solicedb/llm/bier_result_unsorted.csv")
q_count = 0
formatted_results = []
for qid in query_id:
    current = results[results['q'] == q_count]
    unformatted_string = current['did'].tolist()[0]
    unformatted_string = unformatted_string.replace("[", "")
    unformatted_string = unformatted_string.replace("]", "")
    unformatted_string = unformatted_string.replace("'", "")
    unformatted_string = unformatted_string.replace(" ", "")
    list_string = unformatted_string.split(",")
    actual_values = []
    for i in list_string:
        actual_values.append(i)
    formatted_results.append([q_count, qid, actual_values])
    q_count = q_count + 1

formatted = {}
e_model = SentenceTransformer("/localdisk/shin,jason-HonorsThesis/st_model")
for r in formatted_results:
    temp = []
    for docs in r[2]:
        d_id = document_id.index(docs)
        temp.append(document_set[d_id])
    document_embeddings = e_model.encode(temp)
    query_embedding = e_model.encode([query_doc[r[0]]])
    similarities = e_model.similarity(query_embedding, document_embeddings)[0]
    num_assign = {}
    for s in range(len(temp)):
        d_id = document_set.index(temp[s])
        num_assign[f'{document_id[d_id]}'] = similarities[s].item()
    current_id = r[1]
    print(num_assign)
    formatted[f'{current_id}'] = num_assign
try:
    with open("bier_formatted_em.json", 'w') as f:
        json.dump(formatted, f, indent=4)
except:
    None
evaluator = EvaluateRetrieval()
metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], results['time'].mean()]
temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
temp.to_csv(f'bier_metrics_em.csv', index=False)