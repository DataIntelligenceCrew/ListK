from solicedb.llm.plm import MultiRankZephyrLM
import random
import os
import time
import multiprocessing
from typing import Any
import math

class MultiPivot_tour():
    call_count = 0
    def __init__(
        self,
        model_path: str, 
        n_devices: int = 1,
        prompt_template_path: str = "",
        max_tokens: int = 4096,
        window_size: int = 20,
        batch_size: int = 32
    )->None:
        self.model_path = model_path
        self.n_devices = n_devices
        self.prompt_template_path = prompt_template_path
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.batch_size = batch_size
        self.instances = MultiRankZephyrLM(model_path=self.model_path, n_devices=self.n_devices, 
                                prompt_template_path=self.prompt_template_path, max_tokens=self.max_tokens,
                                window_size=self.window_size, batch_size= self.batch_size)

    def produce_pivot_ordering_hint(
        self,
        document_list: list[str],
        sorted_pivots: list[str],
    )->str:
        indexes = []
        for pivots in sorted_pivots:
            i = 0
            while document_list[i] != pivots:
                i = i + 1
            indexes.append(i)
        output= ""
        for i in range(len(indexes)-1):
            output = output + "[" + str(indexes[i]) + "]>"
        output = output + "[" + str(indexes[len(indexes)-1]) + "]"
        return output
        
    def query_wrapper(
        self,
        query: str,
        hints: list[str] = [],
        hint_information: list[Any] = [],
    )->str:
        if hints == []:
            return query
        else:
            output = {'user query': query}
            hint_dict = {}
            for i in range(len(hints)):
                if hints[i] == 'pivot ordering':
                    hint_string = self.produce_pivot_ordering_hint(document_list=hint_information[i][0], sorted_pivots=hint_information[i][1])
                    hint_dict['pivot ordering'] = hint_string
            output['hints'] = str(hint_dict)
            return str(output)
        
    def package_query_document_batch(
        self,
        query:str,
        batch:list[list[str]],
        pivot_hint: bool = False,
        pivot_ordering: list[str] = [],
    )->list[tuple[str,list[str]]]:
        result = []
        for b in batch:
            hint = []
            hint_info = []
            if pivot_hint:
                hint.append('pivot ordering')
                hint_info.append((b,pivot_ordering))
            temp = self.query_wrapper(query=query, hints=hint, hint_information=hint_info)
            result.append((temp, b))
        return result
    
    def handle_batch_call(
        self,
        query:str,
        documents:list[list[str]],
        pivot_hint: bool = False,
        pivot_ordering: list[str] = [],
    )->list[str]:
        completed, completion_time, output = self.instances.call(queries=self.package_query_document_batch(query= query, batch=documents, pivot_hint=pivot_hint, pivot_ordering=pivot_ordering))
        combined_results = []
        for items in output:
            combined_results.append(items[1])
        return combined_results
    
    def handle_i_call(
        self,
        query:str,
        documents:list[list[str]],
        instance: int = 0
    )->list[str]:
        if instance < 0:
            instance = 0
        elif instance >= self.instances.n_devices:
            instance = self.instances.n_devices - 1
        completed, completion_time, output = self.instances.call_i(queries=self.package_query_document_batch(query= query, batch=documents), index=instance)
        combined_results = []
        for items in output:
            combined_results.append(items[1])
        return combined_results

    def organize_documents(
        self,
        maybe_set: list[str],
    )->list[list[str]]:
        cuttoff = self.window_size
        batch_size = self.batch_size * self.instances.n_devices
        counter = 0
        current = []
        organized_documents = []
        for documents in maybe_set:
            if counter <= cuttoff:
                current.append(documents)
                counter = counter + 1
            else:
                random.shuffle(current)
                organized_documents.append(current.copy())
                current = []
                current.append(documents)
                counter = 1
        random.shuffle(current)
        organized_documents.append(current)
        MultiPivot_tour.call_count = MultiPivot_tour.call_count + len(organized_documents)
        counter = 0
        batched_documents = []
        current_batch = []
        for o in organized_documents:
            if counter < batch_size:
                current_batch.append(o)
                counter = counter + 1
            else:
                batched_documents.append(current_batch.copy())
                current_batch =  []
                current_batch.append(o)
                counter = 1
        batched_documents.append(current_batch)
        return batched_documents

    def process(
        self,
        query: str,
        batched_documents: list[list[str]],
        instance: int,
        pivot_hint: bool = False,
    )->list[list[str]]:
        output_list = []
        for batch in batched_documents:
            batch_outputs = []
            if instance != -1:
                batch_outputs = self.handle_i_call(query=query, documents=batch, instance=instance)
            else:
                batch_outputs = self.handle_batch_call(query=query, documents=batch, pivot_hint=pivot_hint, pivot_ordering=[])
            output_list = output_list + batch_outputs
        return output_list

    def process_result(
        self,
        formatted_documents: list[list[str, list[str]]],
        documents: list[str],
        result: list[list[str]],
    )-> list[list[str, list[str]]]:
        for r in result:
            rolling = []
            for ordering in r:
                fid = documents.index(ordering)
                formatted_documents[fid][1] = formatted_documents[fid][1] + rolling.copy()
                rolling.append(ordering)
        return formatted_documents

    def tournament_top_k(
        self,
        query: str,
        documents: list[str],
        k: int,
    )->list[str]:
        formatted_documents = []
        for d in documents:
            formatted_documents.append([d,[]])
        concurrent = []
        solution_set = []
        while len(solution_set) != k:
            for f in formatted_documents:
                if f[1] == [] and f[0] not in solution_set:
                    concurrent.append(f)
            if len(concurrent) == 1:
                solution_set.append(concurrent[0][0])
                for f in formatted_documents:
                    if concurrent[0][0] in f[1]:
                        f[1].remove(concurrent[0][0])
            elif concurrent == []:
                return solution_set
            else:
                document_strings = []
                for d in concurrent:
                    document_strings.append(d[0])
                organized_documents = self.organize_documents(document_strings)
                results = self.process(query=query, batched_documents = organized_documents, instance = -1, pivot_hint = False)
                formatted_documents = self.process_result(formatted_documents, documents, results)
            concurrent = []
        return solution_set

    def stop_models(
        self
    )->None:
        self.instances.stop()