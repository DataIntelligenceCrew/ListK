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
    llmgt_path,
    device_num,
    size,
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
    
    path = llmgt_path
    test = MultiPivot_sort(model_path=rankzephyr_path, n_devices=device_num)
    q = 0
    stop = test_num
    running = []
    while q < stop:
        current = path + f'/{query_id[q]}.parquet'
        sorted_ids = list(pd.read_parquet(current)['doc_id'])[:size]
        documents = []
        for s in sorted_ids:
            d_id = document_id.index(s)
            documents.append(document_set[d_id])
        random.shuffle(documents)
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        print(len(documents))
        start_time = time.perf_counter()
        result = test.sem_sort(query=query_doc[q], documents=documents, pivots=2, group_size=2, pivot_selection_method= 'embedding', embedding_path= embedding_path)
        end_time = time.perf_counter()
        ids = []
        for r in result:
            d_id = document_set.index(r)
            ids.append(document_id[d_id])
        print(len(ids))
        running.append([q, query_id[q], end_time - start_time, len(ids), ids])
        current = pd.DataFrame(running, columns=['q', 'qid', 'time', 'id_len', 'ids'])
        current.to_csv(f'bier_sort_result_{size}_{test_num}.csv', index=False)
        q = q + 1
    test.stop_models()

#Confirm Later
import sys
run_test_unsorted('../../../../models_and_benchmarks/scifact', '../../../../models_and_benchmarks/rankzephyr', "../../../../models_and_benchmarks/em_model", '../../derived/llm-topk-gt/data/phase7_combined_rankings/scifact', int(sys.argv[1]), int(sys.argv[2]), 25)
