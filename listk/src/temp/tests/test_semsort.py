from semsort import MultiPivot
import json
import time
import pandas as pd

def run_test(
    g,
    k,
    p,
    x,
    q,
    r
):
    gathered_data = []
    data = []
    with open("/localdisk/shin,jason-HonorsThesis/datasets/data/corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_object = json.loads(line.strip())
                data.append(json_object['title'])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()} - {e}")
    data = data[:5000]
    for n in g:
        test = MultiPivot(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=n)
        for m in k:
            for i in p:
                for j in x:
                    if j > i:
                        continue
                    for z in range(r):
                        start_time = time.perf_counter()
                        result = test.sem_sort(query=q, documents=data, pivots=i, group_size=j, pivot_selection_method= "embedding", embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model", pivot_hint=True)
                        end_time = time.perf_counter()
                        print(len(result))
                        print(end_time-start_time)
        test.stop_models()
result = run_test(q="list the titles from most related to least related to diabetes", g=[4], k=[20], p=[2], x=[2], r=1)