from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from semtopk import MultiPivot
from semsort import MultiPivot_sort
from semtour import MultiPivot_tour
from gensemtopk import GenMultiPivot
from gensemtour import GenMultiPivot_tour
from gensemsort import GenMultiPivot_sort
import time
import random
import pandas as pd
import json
import torch
import gc
import numpy as np

def run_test(
    cutoff,
    window_size,
    test_num
):
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

    test = MultiPivot(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4, window_size=window_size)

    q = 0
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        if cutoff != None:
            temp = test.embedding_filter(query=query_doc[q], documents=document_set, cutoff=cutoff, embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model")
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        start_time = time.perf_counter()
        result = test.top_k(query=query_doc[q], documents=document_set, top_k=10, pivots=1, group_size=1, pivot_selection_method= "embedding", tournament_filter=[False, 200, 100, 10], embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model",
                            embedding_filter=[False, 500], early_stop=True)
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
        if window_size == 2:
            if cutoff != None:
                temp.to_csv(f'bier_result_unsorted_L_{test_num}_{cutoff}.csv', index=False)
            else:
                temp.to_csv(f'bier_result_unsorted_L_{test_num}_A.csv', index=False)
        else:
            if cutoff != None:
                temp.to_csv(f'bier_result_unsorted_E_{window_size}_{test_num}_{cutoff}.csv', index=False)
            else:
                temp.to_csv(f'bier_result_unsorted_E_{window_size}_{test_num}_A.csv', index=False)
        q = q + 1
    test.stop_models()

    formatted = {}
    from lm import RankZephyrLM
    test = RankZephyrLM(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr")
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
        if window_size == 2:
            if cutoff != None:
                with open(f"bier_formatted_L_{test_num}_{cutoff}.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
            else:
                with open(f"bier_formatted_L_{test_num}_A.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
        else:
            if cutoff != None:
                with open(f"bier_formatted_E_{window_size}_{test_num}_{cutoff}.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
            else:
                with open(f"bier_formatted_E_{window_size}_{test_num}_A.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
    except:
        None
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
    temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(running))]
    temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
    if window_size == 2:
        if cutoff != None:
            temp.to_csv(f'bier_metrics_L_{test_num}_{cutoff}.csv', index=False)
        else:
            temp.to_csv(f'bier_metrics_L_{test_num}_A.csv', index=False)
    else:
        if cutoff != None:
            temp.to_csv(f'bier_metrics_E_{window_size}_{test_num}_{cutoff}.csv', index=False)
        else:
            temp.to_csv(f'bier_metrics_E_{window_size}_{test_num}_A.csv', index=False)
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
        if window_size == 2:
            if cutoff != None:
                current.to_csv(f'bier_i_metrics_L_{test_num}_{cutoff}.csv', index=False)
            else:
                current.to_csv(f'bier_i_metrics_L_{test_num}_A.csv', index=False)
        else:
            if cutoff != None:
                current.to_csv(f'bier_i_metrics_E_{window_size}_{test_num}_{cutoff}.csv', index=False)
            else:
                current.to_csv(f'bier_i_metrics_E_{window_size}_{test_num}_A.csv', index=False)
    total = pd.DataFrame(total, columns=['q', 'NDCG@10', 'MAP@10', 'Recall@10', 'P@10'])
    if window_size == 2:
        if cutoff != None:
            total.to_csv(f'bier_i_metrics_L_{test_num}_{cutoff}.csv', index=False)
        else:
            total.to_csv(f'bier_i_metrics_L_{test_num}_A.csv', index=False)
    else:
        if cutoff != None:
            total.to_csv(f'bier_i_metrics_E_{window_size}_{test_num}_{cutoff}.csv', index=False)
        else:
            total.to_csv(f'bier_i_metrics_E_{window_size}_{test_num}_A.csv', index=False)

def run_single_pivot_test(
    test_num,
    cutoff,
    window_sizes
):
    for w in window_sizes:
        run_test(cutoff, w, test_num)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(5)

def run_test_2(
    p,
    x,
    selection_method,
    test_num
):
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

    test = MultiPivot(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4)

    q = 0
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        start_time = time.perf_counter()
        result = test.top_k(query=query_doc[q], documents=document_set, top_k=10, pivots=p, group_size=x, pivot_selection_method= selection_method, tournament_filter=[True, 518, 100, 10], embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model",
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
    test = RankZephyrLM(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr")
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

def run_test_3(
    cuttoff,
    test_num
):
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

    test = MultiPivot(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4)

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

def run_test_4(
    p,
    x,
    k,
    selection_method,
    test_num
):
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

    test = MultiPivot(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4)

    q = 0
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        start_time = time.perf_counter()
        result = test.top_k(query=query_doc[q], documents=document_set, top_k=k, pivots=p, group_size=x, pivot_selection_method= selection_method, tournament_filter=[False, 3800, 100, 10], embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model",
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
        temp.to_csv(f'bier_result_unsorted_{p}_{x}_{k}_{test_num}.csv', index=False)
        q = q + 1
    test.stop_models()

    formatted = {}
    from lm import RankZephyrLM
    test = RankZephyrLM(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr")
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
        with open(f"bier_formatted_{p}_{x}_{k}_{test_num}.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
    except:
        None
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
    temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(running))]
    temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
    temp.to_csv(f'bier_metrics_{p}_{x}_{k}_{test_num}.csv', index=False)
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
        current.to_csv(f'bier_i_metrics_{p}_{x}_{k}_{test_num}.csv', index=False)
    total = pd.DataFrame(total, columns=['q', 'NDCG@10', 'MAP@10', 'Recall@10', 'P@10'])
    total.to_csv(f'bier_i_metrics_{p}_{x}_{k}_{test_num}.csv', index=False)

def test_5(
    size,
    test_num
):
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
    
    path = './data/llm-topk-gt/data/phase7_combined_rankings/scifact/'
    test = MultiPivot_sort(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4)
    q = 0
    stop = test_num
    running = []
    while q < stop:
        current = path + f'{query_id[q]}.parquet'
        sorted_ids = list(pd.read_parquet(current)['doc_id'])[:size]
        documents = []
        for s in sorted_ids:
            d_id = document_id.index(s)
            documents.append(document_set[d_id])
        random.shuffle(documents)
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        print(len(documents))
        start_time = time.perf_counter()
        result = test.sem_sort(query=query_doc[q], documents=documents, pivots=2, group_size=2, pivot_selection_method= 'embedding', embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model")
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

def run_test_6(
    test_num
):
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
    
    import pandas as pd
    import lotus
    from lotus.models import LM
    lm = LM(model='hosted_vllm//localdisk/shin,jason-HonorsThesis/zephyr7b',
        api_base='http://localhost:8000/v1',
        max_ctx_len=4096,
        max_tokens=1000)
    lotus.settings.configure(lm=lm)

    data = {
    "Datapoints": document_set
    }
    df = pd.DataFrame(data)

    q = 19
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        start_time = time.perf_counter()
        sorted_df, stats = df.sem_topk(
            "{Datapoints}" + query_doc[q],
            K=10,
            method="quick",
            return_stats=True,
        )
        end_time = time.perf_counter()
        ids = []
        result = list(sorted_df['Datapoints'])
        for r in result:
            d_id = document_set.index(r)
            ids.append(document_id[d_id])
        time_sum = time_sum + (end_time-start_time)
        running.append([q, query_id[q], end_time-start_time, ids, stats])
        temp = pd.DataFrame(running, columns=['q', 'qid', 'time', 'ids', 'stats'])
        temp.to_csv(f'bier_lotus_semtopk_result.csv', index=False)
        q = q + 1

def run_test_7(
    k,
    test_num
):
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

    test = MultiPivot_tour(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4)

    q = 0
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        start_time = time.perf_counter()
        result = test.tournament_top_k(query=query_doc[q], documents=document_set, k=k)
        end_time = time.perf_counter()
        ids = []
        for r in result:
            d_id = document_set.index(r)
            ids.append(document_id[d_id])
        i_ids = []
        stats = [q, query_id[q], ids, end_time-start_time]
        print(stats)
        time_sum = time_sum + (end_time-start_time)
        running.append(stats)
        temp = pd.DataFrame(running, columns=['q', 'qid', 'did', 'time'])
        temp.to_csv(f'bier_result_unsorted_tour_{k}_{test_num}.csv', index=False)
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
        with open(f"bier_formatted_tour_{k}_{test_num}.json", 'w') as f:
                    json.dump(formatted, f, indent=4)
    except:
        None
    evaluator = EvaluateRetrieval()
    metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
    temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(running))]
    temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
    temp.to_csv(f'bier_metrics_tour_{k}_{test_num}.csv', index=False)

def sort_evaluator(
    type_data,
    path,
    w,
    p,
    x,
    l,
    mpath
):
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
    
    if type_data == "lotus":
        formatted = {}
        time_sum = 0.0
        for f in formatted_results:
            num_assign = {}
            for s in range(len(f[2])):
                num_assign[f'{f[2][s]}'] = len(f[2])-s
            current_id = f[1]
            formatted[f'{current_id}'] = num_assign
            time_sum = time_sum + f[3]
        try:
            with open(f"bier_formatted_lotus_{len(formatted_results[0][2])}_25.json", 'w') as f:
                        json.dump(formatted, f, indent=4)
        except:
            None
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
        temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(formatted_results))]
        temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
        temp.to_csv(f'bier_metrics_lotus_{len(formatted_results[0][2])}_25.csv', index=False)
    elif type_data == "tour":
        formatted = {}
        time_sum = 0.0
        for f in formatted_results:
            num_assign = {}
            for vals in range(len(f[2])):
                num_assign[f'{f[2][vals]}'] = len(f[2]) - vals
            print(num_assign)
            current_id = f[1]
            formatted[f'{current_id}'] = num_assign
            time_sum = time_sum + f[3]
        try:
            with open(f"bier_formatted_tour_{len(formatted_results[0][2])}_25.json", 'w') as f:
                        json.dump(formatted, f, indent=4)
        except:
            None
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
        temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(formatted_results))]
        temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
        temp.to_csv(f'bier_metrics_tour_10_25.csv', index=False)
    elif type_data == "r":
        formatted = {}
        time_sum = 0.0
        for f in formatted_results:
            num_assign = {}
            for vals in range(len(f[2])):
                num_assign[f'{f[2][vals]}'] = len(f[2]) - vals
            print(num_assign)
            current_id = f[1]
            formatted[f'{current_id}'] = num_assign
            time_sum = time_sum + f[3]
        try:
            with open(f"bier_formatted_sorted_{p}_{x}_{l}_25.json", 'w') as f:
                        json.dump(formatted, f, indent=4)
        except:
            None
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
        temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(formatted_results))]
        temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
        temp.to_csv(f'bier_metrics_sorted_{p}_{x}_{l}_10_25.csv', index=False)
    elif type_data == "s":
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
    elif type_data == "g":
        test = GenMultiPivot_sort(model_path=mpath, n_devices=4, window_size = w)
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
            result = test.sem_sort(query=query_doc[r[0]], documents=temp, pivots=2, group_size=2, pivot_selection_method= 'embedding', embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model")
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
            df.to_csv(f'bier_sorted_10_{p}_{x}_{l}_{w}.csv', index=False)
        try:
            with open(f"bier_formatted_sorted_{p}_{x}_{l}_{w}_25.json", 'w') as f:
                json.dump(formatted, f, indent=4)
        except:
            None
        test.stop_models()
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
        temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(formatted_results))]
        temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
        temp.to_csv(f'bier_metrics_sorted_{p}_{x}_{l}_{w}_10_25.csv', index=False)
    else:
        test = MultiPivot_sort(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4, window_size = w)
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
            result = test.sem_sort(query=query_doc[r[0]], documents=temp, pivots=2, group_size=2, pivot_selection_method= 'embedding', embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model")
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
            df.to_csv(f'bier_sorted_10_{p}_{x}_{l}_{w}.csv', index=False)
        try:
            with open(f"bier_formatted_sorted_{p}_{x}_{l}_{w}_25.json", 'w') as f:
                json.dump(formatted, f, indent=4)
        except:
            None
        test.stop_models()
        evaluator = EvaluateRetrieval()
        metrics = evaluator.evaluate(qrels, formatted, k_values=[10])
        temp = [metrics[0]['NDCG@10'], metrics[1]['MAP@10'], metrics[2]['Recall@10'], metrics[3]['P@10'], time_sum/(len(formatted_results))]
        temp = pd.DataFrame([temp], columns=['NDCG@10', 'MAP@10', 'Recall@10', 'P@10', 'time'])
        temp.to_csv(f'bier_metrics_sorted_{p}_{x}_{l}_{w}_10_25.csv', index=False)

def run_test_8(
    p,
    x,
    path,
    name,
    test_num
):
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

    test = GenMultiPivot(model_path=path, n_devices=4)

    q = 17
    stop = test_num
    running = []
    time_sum = 0.0
    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        start_time = time.perf_counter()
        result = test.top_k(query=query_doc[q], documents=document_set, top_k=10, pivots=p, group_size=x, pivot_selection_method= 'embedding', tournament_filter=[False, 518, 100, 10], embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model",
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

def run_test_10(
    test_num
):
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
    
    import pandas as pd
    import lotus
    from lotus.models import LM
    lm = LM(model='hosted_vllm//localdisk/shin,jason-HonorsThesis/qwen',
        api_base='http://localhost:8000/v1',
        max_ctx_len=4096,
        max_tokens=1000)
    lotus.settings.configure(lm=lm)

    data = {
    "Datapoints": document_set
    }
    df = pd.DataFrame(data)

    q = 0
    stop = test_num
    running = []
    time_sum = 0.0

    tag = """Review: {Datapoints}

    Only provide the score number (1-5000) with no other comments."""


    while q < stop:
        print(str(q) + "/" + str(len(query_doc)) + ": " + str(query_doc[q]))
        prompt_header = f"""Score from 1 to 5000 based on {query_doc[q]}

        Rubrics:
        1: Very related to the scoring criteria.
        5000: Very unrelated to the scoring criteria.
        
        """
        start_time = time.perf_counter()
        sorted_df = df.sem_map(prompt_header + tag)
        end_time = time.perf_counter()
        results = []
        for _, row in sorted_df.iterrows():
            score = row["_map"]
            # Ensure score is numeric and within range 1-5
            try:
                numeric_score = float(score)
                if 1 <= numeric_score <= 5000:
                    results.append(
                        {
                            "text": row["Datapoints"],
                            "reviewScore": numeric_score,
                        }
                    )
                else:
                    # Default to 3 if score is out of range
                    results.append(
                        {"text": row["Datapoints"], "reviewScore": 2500.0}
                    )
            except (ValueError, TypeError):
                # Default to 3 if score is not numeric
                results.append(
                    {"text": row["Datapoints"], "reviewScore": 2500.0}
                )
        result_df = pd.DataFrame(results)
        result_sorted = result_df.sort_values(by='reviewScore', ascending=True)
        ids = []
        result = list(result_sorted['text'])[:10]
        for r in result:
            d_id = document_set.index(r)
            ids.append(document_id[d_id])
        time_sum = time_sum + (end_time-start_time)
        running.append([q, query_id[q], end_time-start_time, ids])
        temp = pd.DataFrame(running, columns=['q', 'qid', 'time', 'ids'])
        temp.to_csv(f'bier_lotus_map_result_q.csv', index=False)
        q = q + 1

def run_test_11(
    k,
    l,
    path,
    test_num
):
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

    results = pd.read_csv(path)
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
    test = MultiPivot_tour(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4)

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

import sys

run_single_pivot_test(25, None, [int(sys.argv[1])])
#run_test_2(int(sys.argv[1]), int(sys.argv[2]), "embedding", 25)
#run_test_3(int(sys.argv[1]), 25)
#run_test_4(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), "embedding", 25)
#test_5(int(sys.argv[1]), 25)
#run_test_6(25)
#run_test_7(int(sys.argv[1]), 25)
#sort_evaluator(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7])
#run_test_8(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], 25)
#run_test_9(int(sys.argv[1]), sys.argv[2], sys.argv[3], 25)
#run_test_10(25)
#run_test_11(10, int(sys.argv[1]), sys.argv[2], 25)