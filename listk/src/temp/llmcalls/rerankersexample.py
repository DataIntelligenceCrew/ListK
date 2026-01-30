'''
    This is code from https://github.com/castorini/rank_llm/blob/main/docs/external-integrations.md
    which is about external integrations of rankllm
'''
from rerankers import Reranker

ranker = Reranker('/home/jshin/LLMtest/first_qwen3_0.6b', model_type="rankllm")

results = ranker.rank(query="I love you", docs=["I hate you", "I really like you"], doc_ids=[0,1])
print(results)