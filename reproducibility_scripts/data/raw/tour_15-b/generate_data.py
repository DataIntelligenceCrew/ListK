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
    embedding_path,
    device_num,
    p,
    x,
    selection_method,
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

    test = MultiPivot(model_path=rankzephyr_path, n_devices=device_num)

    q = 0
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        start_time = time.perf_counter()
        result = test.top_k(query=query_doc[q], documents=document_set, top_k=10, pivots=p, group_size=x, pivot_selection_method= selection_method, tournament_filter=[True, 3887, 100, 10], embedding_path= embedding_path,
                            embedding_filter=[False, 200], early_stop=True)
        end_time = time.perf_counter()
        ids = []
        for r in result:
            d_id = document_set.index(r)
            ids.append(document_id[d_id])
        i_ids = []
        for i in test.intermediary_definitely_set:
            current = []
            for items in i:
                d_id = document_set.index(items)
                current.append(document_id[d_id])
            i_ids.append(current)
        stats = [q, query_id[q], ids, test.call_count, end_time-start_time, test.definitely_set_size, test.maybe_set_size, test.latency, test.call_counts, i_ids]
        print(stats)
        time_sum = time_sum + (end_time-start_time)
        running.append(stats)
        temp = pd.DataFrame(running, columns=['q', 'qid', 'did', 'n_call', 'time', 'd_size', 'm_size', 'l_latency', 'c_count', 'i_der'])
        temp.to_csv(f'bier_result_unsorted_{p}_{x}_{test_num}.csv', index=False)
        q = q + 1
    test.stop_models()

    formatted = {}
    from lm import RankZephyrLM
    test = RankZephyrLM(model_path=rankzephyr_path)
    for r in running:
        temp = []
        for docs in r[2]:
            d_id = document_id.index(docs)
            temp.append(document_set[d_id])
        f, t, sorted_list = test.call(queries=[(query_doc[r[0]],temp)])
        num_assign = {}
        for s in range(len(sorted_list[0][1])):
            d_id = document_set.index(sorted_list[0][1][s])
            num_assign[f'{document_id[d_id]}'] = len(sorted_list[0][1])-s
        current_id = r[1]
        formatted[f'{current_id}'] = num_assign
    try:
        with open(f"bier_formatted_{p}_{x}_{test_num}.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
    except:
        None
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
    temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(running))]
    temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
    temp.to_csv(f'bier_metrics_{p}_{x}_{test_num}.csv', index=False)
    total = []
    for r in running:
        m_n = []
        m_m = []
        m_r = []
        m_p = []
        for values in r[9]:
            formatted = {}
            if values != []:
                temp = []
                for docs in values:
                    d_id = document_id.index(docs)
                    temp.append(document_set[d_id])
                f, t, sorted_list = test.call(queries=[(query_doc[r[0]],temp)])
                num_assign = {}
                for s in range(len(sorted_list[0][1])):
                    d_id = document_set.index(sorted_list[0][1][s])
                    num_assign[f'{document_id[d_id]}'] = len(sorted_list[0][1])-s
                current_id = r[1]
                formatted[f'{current_id}'] = num_assign
                metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
                m_n.append(metrics[0]['NDCG@10'])
                m_m.append(metrics[1]['MAP@10'])
                m_r.append(metrics[2]['Recall@10'])
                m_p.append(metrics[3]['P@10'])
            else:
                m_n.append(0.0)
                m_m.append(0.0)
                m_r.append(0.0)
                m_p.append(0.0)
        temp = [r[0], m_n, m_m, m_r, m_p]
        total.append(temp)
        current = pd.DataFrame(total, columns=['q', 'NDCG@10', 'MAP@10', 'Recall@10', 'P@10'])
        current.to_csv(f'bier_i_metrics_{p}_{x}_{test_num}.csv', index=False)
    total = pd.DataFrame(total, columns=['q', 'NDCG@10', 'MAP@10', 'Recall@10', 'P@10'])
    total.to_csv(f'bier_i_metrics_{p}_{x}_{test_num}.csv', index=False)

import sys
run_test_unsorted('../../../../models_and_benchmarks/scifact', '../../../../models_and_benchmarks/rankzephyr', "../../../../models_and_benchmarks/em_model", int(sys.argv[1]), 16, 2, "embedding", 25)
