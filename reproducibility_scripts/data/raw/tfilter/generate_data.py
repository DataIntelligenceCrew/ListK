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
    device_num,
    cuttoff,
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
        result = test.tournament_filter(query=query_doc[q], documents=document_set, cutoff=cuttoff, leniancy=100, kill_loop = 2)
        end_time = time.perf_counter()
        ids = []
        for r in result:
            d_id = document_set.index(r)
            ids.append(document_id[d_id])
        time_sum = time_sum + (end_time-start_time)
        running.append([q, len(ids), end_time-start_time, ids])
        print(len(ids))
        temp = pd.DataFrame(running, columns=['q', 'doc_num', 'time', 'ids'])
        temp.to_csv(f'bier_tfilter_result_{cuttoff}_{test_num}.csv', index=False)
        q = q + 1
    test.stop_models()

#Confirm Later
import sys
run_test_unsorted('../../../../models_and_benchmarks/scifact', '../../../../models_and_benchmarks/rankzephyr', int(sys.argv[1]), int(sys.argv[2]), 25)
