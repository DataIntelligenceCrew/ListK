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

    data = []
    formatted = {}
    time_sum = 0.0
    for f in formatted_results:
        num_assign = {}
        for vals in range(len(f[2][:l])):
            num_assign[f'{f[2][vals]}'] = len(f[2][:l]) - vals
        print(num_assign)
        current_id = f[1]
        formatted[f'{current_id}'] = num_assign
        time_sum = time_sum + f[3]
    try:
        with open(f"bier_formatted_sorted_{w}_{l}_25.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
    except:
        None
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, formatted, k_values=[l])
    temp = [metrics[0][f'NDCG@{l}'], metrics[1][f'MAP@{l}'], metrics[2][f'Recall@{l}'], metrics[3][f'P@{l}'], time_sum/(len(formatted_results))]
    temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
    temp.to_csv(f'bier_metrics_sorted_{w}_{l}_10_25.csv', index=False)

import sys

#python generate_data.py <scifact_path> <rankzephyr_path> <embedding_path> <device_num> <p> <x> <selection_mathod> <test_num>
#python clean_data.py <result_path> <scifact_path> <rankzephyr_path> <embedding_path> <device_num> <p> <x> <w> <test_num>
#python generate_data.py /localdisk/shin,jason-HonorsThesis/datasets/scifact_bier /localdisk/shin,jason-HonorsThesis/rankzephyr /localdisk/shin,jason-HonorsThesis/st_model 4 16 2 embedding 1;
#python clean_data.py bier_result_unsorted_16_2_1.csv /localdisk/shin,jason-HonorsThesis/datasets/scifact_bier /localdisk/shin,jason-HonorsThesis/rankzephyr /localdisk/shin,jason-HonorsThesis/st_model 4 16 2 2 1
sort_evaluator(f'../bier_sort_result_{int(sys.argv[2])}_25.csv', '../../../../models_and_benchmarks/scifact', '../../../../models_and_benchmarks/rankzephyr', "../../../../models_and_benchmarks/em_model", int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[2]), 25)