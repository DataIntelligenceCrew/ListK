from semtopk import MultiPivot
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
    data = data[:500]
    for n in g:
        test = MultiPivot(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=n)
        for m in k:
            for i in p:
                for j in x:
                    if j > i:
                        continue
                    for z in range(r):
                        start_time = time.perf_counter()
                        result = test.top_k(query=q, documents=data, top_k=m, pivots=i, group_size=j, pivot_selection_method= "embedding", tournament_filter=[False, 200, 100, 10], embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model",
                                            embedding_filter=[False, 200], early_stop=True)
                        end_time = time.perf_counter()
                        gathered_data.append([q, i, j, test.call_count, end_time-start_time])
                        print([q, i, j, test.call_count, end_time-start_time])
                        temp = pd.DataFrame(gathered_data, columns=['query', 'p', 'x', 'n_call', 'time'])
                        temp.to_csv('test_run_results_t_500.csv', index=False)
        test.stop_models()
    return pd.DataFrame(gathered_data, columns=['query', 'p', 'x', 'n_call', 'time'])

result = run_test(q="list the titles from most related to least related to diabetes", g=[4], k=[20], p=[1,2,4,8,16], x=[1,2,4,8,16], r=20)
result.to_csv('test_run_results_t_500.csv', index=False)

