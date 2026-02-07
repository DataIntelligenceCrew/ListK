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
    model_path,
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
    
    import pandas as pd
    import lotus
    from lotus.models import LM
    lpath = 'hosted_vllm/' + model_path
    lm = LM(model=lpath,
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
        temp.to_csv(f'bier_lotus_map_result.csv', index=False)
        q = q + 1

import sys
run_test_unsorted('../../../../models_and_benchmarks/scifact', sys.argv[1], 25)
