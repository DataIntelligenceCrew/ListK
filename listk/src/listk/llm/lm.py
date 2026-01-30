"""
    Allows for calls to any LLM utilizing VLLM or RankZephyr utilizing 
    rankllm. For both the generic and rankzephyr classes it supports
    initialization as well as batch calling.

    Some initial simplifications and assumptions: currently the assumption
    is all documents (wether they be rows/columns in a table or text) 
    are fed in as either text (rankllm) or a formatted prompt which assumes 
    all documents are text (vllm). Some initial simplifications are that the batch
    size of the VLLM implementation is currently fixed to the default and the rankzephyr
    input size is fixed to be a maximum of 20 (with a stride of 10) meaning
    any list of documents longer than 20 will have a sliding window performed to get
    the ordered list.
"""

import time
import torch
from vllm import LLM, SamplingParams
from typing import Any
from rank_llm.data import Candidate, Query, Request, Result
from rank_llm.rerank.listwise import ZephyrReranker
from rank_llm.rerank.listwise import RankListwiseOSLLM

class GenericRankLM:
    """
        Initializes and allows calls to RankZephyr utilizing rankllm (note this requires having openjdk 21 with maven)
    """
    genrank: RankListwiseOSLLM
    init_time: float

    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        prompt_template_path: str = "",
        max_tokens: int = 4096,
        window_size: int = 20,
        batch_size: int = 32,
    ) -> None:
        """
            Loads the LLM and takes the time of initialization
        """
        self .model_path = model_path
        self.device = device
        self.prompt_template_path = prompt_template_path
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.batch_size = batch_size
        start_time = time.perf_counter()
        if self.prompt_template_path != "":
            GenericRankLM.genrank = RankListwiseOSLLM(model = self.model_path, context_size = self.max_tokens, device = self.device, prompt_template_path = self.prompt_template_path,
                                                 window_size = self.window_size, batch_size = self.batch_size)
        else:
            GenericRankLM.genrank = RankListwiseOSLLM(model = self.model_path, context_size = self.max_tokens, device = self.device, window_size = self.window_size)
        end_time = time.perf_counter()
        GenericRankLM.init_time = end_time - start_time 

    def generate_request(
        self,
        query: str,
        query_id: int,
        documents: list[str],
    )->Request:
        """
            rankllm takes specific request objects. This method takes the list of documents and the query of
            a given call and returns the formatted request.
        """
        return Request(
            query = Query(text=query, qid=query_id),
            candidates = [Candidate(doc = {"text":d}, docid = documents.index(d), score = 1) for d in documents]
        )
    
    def clean_result(
        self,
        result: Result
    )-> tuple[str,list[str]]:
        """
            Takes the output object for the call result and outputs a tuple which
            has the first element being the user query and the second the
            ordered list of documents.
        """
        return (result.query.text, [c.doc['text'] for c in result.candidates])
         
    def call(
        self, 
        queries: list[tuple[str,list[str]]],
    )-> tuple[bool,float,list[tuple[str,list[str]]]]:
        """
            Allows multiple calls to Rankzephyr with an input of a list of requests that are
            (user query, list of documents) where the list of documents is a list of strings.
            It outputs a check of if the number of inputs matches the outputs, the time of execution,
            and the cleaned results as a list.
        """
        start_time = time.perf_counter()
        outputs = GenericRankLM.genrank.rerank_batch([self.generate_request(query = q[0], query_id = queries.index(q), documents = q[1]) for q in queries])
        end_time = time.perf_counter()
        return True if len(queries) == len(outputs) else False, (end_time - start_time), [self.clean_result(result) for result in outputs]

class GenericLM:
    """
        Initializes and allows calls to a LLM using VLLM.
    """
    llm: LLM
    sampling_params: SamplingParams
    init_time: float

    def __init__(
        self, 
        model_path: str, 
        temeprature: float = 0.8, 
        top_p: float = 0.95, 
        max_tokens: int = 4096,
        gpu_memory_utilization: float = 0.85
    ) -> None:
        """
            Loads the LLM and takes the time of initialization
        """
        self.model_path = model_path
        self.temeprature = temeprature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        start_time = time.perf_counter()
        GenericLM.sampling_params = SamplingParams(temperature= self.temeprature, top_p= self.top_p, max_tokens = self.max_tokens)
        GenericLM.llm = LLM(model=self.model_path, gpu_memory_utilization =  self.gpu_memory_utilization)
        end_time = time.perf_counter()
        GenericLM.init_time = end_time - start_time

    def clean_result(
        self,
        result: str
        )->tuple[str,str]:
        """
            Cleans the text string result of the LLM to return a tuple with the first being
            the reasoning (if any) and the second the answer.
        """
        reasoning = ""
        answer = ""
        astart = result.find("Answer:")
        rstart = result.find("<think>")
        rend = result.find("</think>")
        if rstart != -1 and rend != -1:
            reasoning = result[rstart + len("<think>"):rend].strip("\n")
        if astart != -1:
            answer = result[astart + len("Answer:"):].strip("\n")
        elif astart == -1 and rend != -1:
            answer = result[rend + len("<\/think>"):].strip("\n")
        else:
            answer = result
        return reasoning, answer

    def call(
        self, 
        queries: list[list[dict[str:Any]]]
    )-> tuple[bool, float, list[(str,str)]]:
        """
            Makes calls to the LLM with the queries being inputted after being formatted.
            Outputs the condition of if the number of outputs is equal to the inputs, the
            time it took to complete all queries, and the cleaned outputs.
        """
        start_time = time.perf_counter()
        outputs = GenericLM.llm.chat(queries, GenericLM.sampling_params, use_tqdm=True)
        end_time = time.perf_counter()
        return True if len(queries) == len(outputs) else False, (end_time - start_time), [self.clean_result(result.outputs[0].text) for result in outputs]

class RankZephyrLM:
    """
        Initializes and allows calls to RankZephyr utilizing rankllm (note this requires having openjdk 21 with maven)
    """
    zephyr: ZephyrReranker
    init_time: float

    def __init__(
        self, 
        model_path: str, 
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        prompt_template_path: str = "",
        max_tokens: int = 4096,
        window_size: int = 20,
        batch_size: int = 32,
    ) -> None:
        """
            Loads the LLM and takes the time of initialization
        """
        self .model_path = model_path
        self.device = device
        self.prompt_template_path = prompt_template_path
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.batch_size = batch_size
        start_time = time.perf_counter()
        if self.prompt_template_path != "":
            RankZephyrLM.zephyr = ZephyrReranker(model_path = self.model_path, context_size = self.max_tokens, device = self.device, prompt_template_path = self.prompt_template_path,
                                                 window_size = self.window_size, batch_size = self.batch_size)
        else:
            RankZephyrLM.zephyr = ZephyrReranker(model_path = self.model_path, context_size = self.max_tokens, device = self.device, window_size = self.window_size)
        end_time = time.perf_counter()
        RankZephyrLM.init_time = end_time - start_time 

    def generate_request(
        self,
        query: str,
        query_id: int,
        documents: list[str],
    )->Request:
        """
            rankllm takes specific request objects. This method takes the list of documents and the query of
            a given call and returns the formatted request.
        """
        return Request(
            query = Query(text=query, qid=query_id),
            candidates = [Candidate(doc = {"text":d}, docid = documents.index(d), score = 1) for d in documents]
        )
    
    def clean_result(
        self,
        result: Result
    )-> tuple[str,list[str]]:
        """
            Takes the output object for the call result and outputs a tuple which
            has the first element being the user query and the second the
            ordered list of documents.
        """
        return (result.query.text, [c.doc['text'] for c in result.candidates])
         
    def call(
        self, 
        queries: list[tuple[str,list[str]]],
    )-> tuple[bool,float,list[tuple[str,list[str]]]]:
        """
            Allows multiple calls to Rankzephyr with an input of a list of requests that are
            (user query, list of documents) where the list of documents is a list of strings.
            It outputs a check of if the number of inputs matches the outputs, the time of execution,
            and the cleaned results as a list.
        """
        start_time = time.perf_counter()
        outputs = RankZephyrLM.zephyr.rerank_batch([self.generate_request(query = q[0], query_id = queries.index(q), documents = q[1]) for q in queries])
        end_time = time.perf_counter()
        return True if len(queries) == len(outputs) else False, (end_time - start_time), [self.clean_result(result) for result in outputs]