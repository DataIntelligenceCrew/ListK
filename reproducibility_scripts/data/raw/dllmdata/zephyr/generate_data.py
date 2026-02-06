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
    mpath,
    embedding_path,
    name,
    device_num,
    p,
    x,
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

    test = GenMultiPivot(model_path=mpath, n_devices=device_num)

    q = 0
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        start_time = time.perf_counter()
        result = test.top_k(query=query_doc[q], documents=document_set, top_k=10, pivots=p, group_size=x, pivot_selection_method= 'embedding', tournament_filter=[False, 518, 100, 10], embedding_path= embedding_path,
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
        temp.to_csv(f'bier_result_unsorted_{name}_{p}_{x}_{test_num}.csv', index=False)
        q = q + 1
    test.stop_models()

import sys
run_test_unsorted('../../../../models_and_benchmarks/scifact', '../../../../models_and_benchmarks/zephyr7b', "../../../../models_and_benchmarks/em_model", "zephyr7b", int(sys.argv[1]), 16, 2, 25)
