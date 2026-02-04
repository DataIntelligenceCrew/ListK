from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from listk.semop.semtopk import MultiPivot
from listk.semop.semsort import MultiPivot_sort
from listk.semop.semtour import MultiPivot_tour
from listk.semop.gensemtopk import GenMultiPivot
from listk.semop.gensemtour import GenMultiPivot_tour
from listk.semop.gensemsort import GenMultiPivot_sort
import time
import random
import pandas as pd
import json
import torch
import gc
import numpy as np

#listk.semop.

def run_test_unsorted(
    scifact_dir,
    rankzephyr_path,
    filter_path,
    device_num,
    k,
    l,
    test_num
):
    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    out_dir = scifact_dir
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

    results = pd.read_csv(filter_path)
    q_count = 0
    formatted_results = []
    while q_count < 25:
        current = results[results['q'] == q_count]
        unformatted_string = current['ids'].tolist()[0]
        unformatted_string = unformatted_string.replace("[", "")
        unformatted_string = unformatted_string.replace("]", "")
        unformatted_string = unformatted_string.replace("'", "")
        unformatted_string = unformatted_string.replace(" ", "")
        list_string = unformatted_string.split(",")
        actual_values = []
        for i in list_string:
            actual_values.append(i)
        formatted_results.append([actual_values, current['time'].tolist()[0]])
        q_count = q_count + 1
    test = MultiPivot_tour(model_path=rankzephyr_path, n_devices=device_num)

    q = 0
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        current = formatted_results[q]
        current_time = current[1]
        to_sort = []
        for ids in current[0]:
            d_id = document_id.index(ids)
            to_sort.append(document_set[d_id])
        start_time = time.perf_counter()
        result = test.tournament_top_k(query=query_doc[q], documents=to_sort, k=k)
        end_time = time.perf_counter()
        ids = []
        for r in result:
            d_id = document_set.index(r)
            ids.append(document_id[d_id])
        i_ids = []
        stats = [q, query_id[q], ids, current_time + (end_time-start_time)]
        print(stats)
        time_sum = time_sum + (current_time + (end_time-start_time))
        running.append(stats)
        temp = pd.DataFrame(running, columns=['q', 'qid', 'did', 'time'])
        temp.to_csv(f'bier_result_unsorted_tour_{k}_{l}_{test_num}.csv', index=False)
        q = q + 1
    test.stop_models()

    formatted = {}
    from lm import RankZephyrLM
    for r in running:
        num_assign = {}
        for s in range(len(r[2])):
            num_assign[f'{r[2][s]}'] = len(r[2])-s
        current_id = r[1]
        formatted[f'{current_id}'] = num_assign
    try:
        with open(f"bier_formatted_tour_{k}_{l}_{test_num}.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
    except:
        None
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
    temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(running))]
    temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
    temp.to_csv(f'bier_metrics_tour_{k}_{l}_{test_num}.csv', index=False)

#Confirm Later
import sys
run_test_unsorted(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
