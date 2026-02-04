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

def sort_evaluator(
    path,
    scifact_dir,
    rankzephyr_path,
    embedding_path,
    device_num,
    p,
    x,
    l,
    w,
    test_num,
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
    
    results = pd.read_csv(path)
    q_count = 0
    formatted_results = []
    while q_count < 25:
        current = results[results['q'] == q_count]
        try:
            unformatted_string = current['did'].tolist()[0]
            unformatted_string = unformatted_string.replace("[", "")
            unformatted_string = unformatted_string.replace("]", "")
            unformatted_string = unformatted_string.replace("'", "")
            unformatted_string = unformatted_string.replace(" ", "")
            list_string = unformatted_string.split(",")
            actual_values = []
            for i in list_string:
                actual_values.append(i)
            formatted_results.append([q_count, query_id[q_count], actual_values, current['time'].tolist()[0]])
        except:
            try:
                unformatted_string = current['ids'].tolist()[0]
                unformatted_string = unformatted_string.replace("[", "")
                unformatted_string = unformatted_string.replace("]", "")
                unformatted_string = unformatted_string.replace("'", "")
                unformatted_string = unformatted_string.replace(" ", "")
                list_string = unformatted_string.split(",")
                actual_values = []
                for i in list_string:
                    actual_values.append(i)
                formatted_results.append([q_count, query_id[q_count], actual_values, current['time'].tolist()[0]])
            except:
                None
            None
        q_count = q_count + 1

    test = MultiPivot_sort(model_path=rankzephyr_path, n_devices=device_num, window_size = w)
    data = []
    formatted = {}
    time_sum = 0.0
    for r in formatted_results:
        temp = []
        for docs in r[2]:
            d_id = document_id.index(docs)
            temp.append(document_set[d_id])
        random.shuffle(temp)
        start_time = time.perf_counter()
        result = test.sem_sort(query=query_doc[r[0]], documents=temp, pivots=2, group_size=2, pivot_selection_method= 'embedding', embedding_path= embedding_path)
        end_time = time.perf_counter()
        time_sum = time_sum + r[3] + (end_time-start_time)
        num_assign = {}
        ids = []
        for s in range(len(result)):
            d_id = document_set.index(result[s])
            ids.append(document_id[d_id])
            num_assign[f'{document_id[d_id]}'] = 10-s
        current_id = r[1]
        formatted[f'{current_id}'] = num_assign
        data.append([r[0], r[1], ids, r[3] + (end_time-start_time)])
        df = pd.DataFrame(data, columns=['q', 'qid', 'did', 'time'])
        df.to_csv(f'bier_sorted_10_16_2_1_2.csv', index=False)
    try:
        with open(f"bier_formatted_16_2_1_2_25.json", 'w') as f:
            json.dump(formatted, f, indent=4)
    except:
        None
    test.stop_models()
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
    temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(formatted_results))]
    temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
    temp.to_csv(f'bier_metrics_16_2_1_2_10_25.csv', index=False)

import sys

#python generate_data.py <scifact_path> <rankzephyr_path> <embedding_path> <device_num> <p> <x> <selection_mathod> <test_num>
#python clean_data.py <result_path> <scifact_path> <rankzephyr_path> <embedding_path> <device_num> <p> <x> <w> <test_num>
#python generate_data.py /localdisk/shin,jason-HonorsThesis/datasets/scifact_bier /localdisk/shin,jason-HonorsThesis/rankzephyr /localdisk/shin,jason-HonorsThesis/st_model 4 16 2 embedding 1;
#python clean_data.py bier_result_unsorted_16_2_1.csv /localdisk/shin,jason-HonorsThesis/datasets/scifact_bier /localdisk/shin,jason-HonorsThesis/rankzephyr /localdisk/shin,jason-HonorsThesis/st_model 4 16 2 2 1
sort_evaluator(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]))