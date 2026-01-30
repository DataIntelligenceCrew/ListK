from solicedb.llm.plm import MultiGenericRankLM
import random
import os
import time
import multiprocessing
from typing import Any
import math

class GenMultiPivot_sort():
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
        self.instances = MultiGenericRankLM(model_path=self.model_path, n_devices=self.n_devices, 
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

    def pivot_selection(
        self,
        pivots: int,
        query: str,
        documents: list[str],
        method: str,
        embedding_path: str = "",
    )->list[str]:
        '''
            Implement different methods later
        '''
        pivot_list = []
        consider = documents
        if method == "embedding" and embedding_path != "":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import SentenceTransformer
            e_model = SentenceTransformer(embedding_path)
            document_embeddings = e_model.encode(documents)
            query_embedding = e_model.encode([query])
            similarities = e_model.similarity(query_embedding, document_embeddings)[0]
            temp, consider = zip(*sorted(zip(similarities, documents), reverse=True))
            if pivots != 1:
                cut = math.floor(len(consider)/pivots)
                for p in range(pivots):
                    pivot_list.append(consider[min(len(consider)-1, p*cut)])
                pivot_list = list(set(pivot_list))
            else:
                pivot_list = [consider[math.floor(len(consider)/2)]]
        else:
            if len(consider) <= pivots:
                pivots = len(consider) - 1
            selected_num = []
            first_value = random.randint(0,len(consider)-1)
            selected_num.append(first_value)
            pivot_list.append(consider[first_value])
            for p in range(pivots-1):
                candidate_num = random.randint(0,len(consider)-1)
                while candidate_num in selected_num:
                    candidate_num = random.randint(0,len(consider)-1)
                selected_num.append(candidate_num)
                pivot_list.append(consider[candidate_num])
        if len(pivot_list) == 1:
            return pivot_list
        GenMultiPivot_sort.call_count = GenMultiPivot_sort.call_count + 1
        return self.handle_batch_call(query, [pivot_list])[0]
    
    def group_pivots(
        self,
        group_size: int,
        pivot_list: list[str]
    )->list[list[str]]:
        counter = 0
        output_list = []
        current_group = []
        for pivots in pivot_list:
            if counter < group_size:
                current_group.append(pivots)
                counter = counter + 1
            else:
                output_list.append(current_group.copy())
                current_group = [pivots]
                counter = 1
        if current_group != []:
            output_list.append(current_group.copy())
        return output_list
    
    def assign_pivot_instance(
        self,
        num_pivots_blocks: int
    )->list[int]:
        instance_assignment = []
        for p in range(num_pivots_blocks):
            instance_assignment.append(p%self.instances.n_devices)
        return instance_assignment

    def organize_documents(
        self,
        pivot_documents: list[str],
        maybe_set: list[str],
        grouping: int = 1,
    )->list[list[str]]:
        cuttoff = self.window_size - len(pivot_documents)
        batch_size = self.batch_size * grouping
        counter = 0
        organized_documents = []
        current = pivot_documents.copy()
        for documents in maybe_set:
            if counter < cuttoff:
                current.append(documents)
                counter = counter + 1
            else:
                random.shuffle(current)
                organized_documents.append(current.copy())
                current = pivot_documents.copy()
                current.append(documents)
                counter = 1
        random.shuffle(current)
        organized_documents.append(current)
        GenMultiPivot_sort.call_count = GenMultiPivot_sort.call_count + len(organized_documents)
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
    
    def calculate_result(
        self,
        pivot: str,
        result: list[str]
    )->tuple[set[str], set[str]]:
        larger = set()
        smaller = set()
        pivot_found = False
        for item in result:
            if item == pivot:
                pivot_found = True
            elif not pivot_found:
                larger = larger | {item,}
            else:
                smaller = smaller | {item,}
        return (larger,smaller)
    
    def process_pivot(
        self,
        query: str,
        pivot_list: list[str],
        batched_documents: list[list[str]],
        instance: int,
        pivot_hint: bool = False,
    )->tuple[set[str],set[str]]:
        larger = [set()]*len(pivot_list)
        smaller = [set()]*len(pivot_list)
        for batch in batched_documents:
            batch_outputs = []
            if instance != -1:
                batch_outputs = self.handle_i_call(query=query, documents=batch, instance=instance)
            else:
                batch_outputs = self.handle_batch_call(query=query, documents=batch, pivot_hint=pivot_hint, pivot_ordering=pivot_list)
            for items in batch_outputs:
                for i in range(len(pivot_list)):
                    major, minor = self.calculate_result(pivot=pivot_list[i], result=items)
                    larger[i] = larger[i] | (major - set(pivot_list))
                    smaller[i] = smaller[i] | (minor - set(pivot_list))
        output_list = []
        for i in range(len(pivot_list)):
            output_list.append((larger[i], smaller[i]))
        return output_list
    
    def process_handler(
        self,
        query: str,
        pivot_list: list[str],
        instance: int,
        documents: list[str],
        output: list[Any],
        index: int,
    )->None:
        result = self.process_pivot(query=query, pivot_list=pivot_list, batched_documents=documents, instance=instance)
        output[index] = result

    def sem_sort(
        self,
        query: str,
        documents: list[str],
        pivot_selection_method: str,
        pivots: int = 1,
        group_size: int = 1,
        embedding_path: str = "",
        pivot_hint: bool = False,
    )->list[str]:
        if len(documents) == 1:
            return documents
        elif len(documents) == 0:
            return []
        elif len(documents) <= self.window_size:
            GenMultiPivot_sort.call_count = GenMultiPivot_sort.call_count + 1
            return self.handle_batch_call(query=query, documents=[list(documents)])[0]
        if pivots <= 0:
            pivots = 1
        if pivots > self.window_size:
            pivots = self.window_size - 1
        if group_size <= 0:
            group_size = 1
        if group_size > pivots:
            group_size = pivots
        if group_size == self.window_size:
            group_size = self.window_size - 1
        GenMultiPivot_sort.call_count = GenMultiPivot_sort.call_count + 1
        ordered_pivots = self.pivot_selection(pivots=pivots, query=query, documents=documents, method=pivot_selection_method, embedding_path = embedding_path)
        organized_pivots = self.group_pivots(group_size=group_size, pivot_list=ordered_pivots)
        maybe_set = list(set(documents) - set(ordered_pivots))
        grouped_pivot_result = []
        for p in organized_pivots:
            organized_maybe_set = self.organize_documents(pivot_documents=p, maybe_set=maybe_set, grouping = self.n_devices)
            grouped_pivot_result.append(self.process_pivot(query=query, pivot_list=p, batched_documents=organized_maybe_set, instance = -1, pivot_hint=pivot_hint))
        pivot_result = []
        for p in grouped_pivot_result:
            pivot_result = pivot_result + p
        result = []
        for index in range(len(pivot_result)):
            current_large, current_small = pivot_result[index]
            if index != 0:
                current_large = current_large & pivot_result[index-1][1] - pivot_result[index-1][0]
                for j in range(index-1):
                    current_large = current_large - pivot_result[j][0]
            if index != len(pivot_result) - 1:
                result = result + self.sem_sort(query=query, documents=list(current_large), pivot_selection_method=pivot_selection_method, pivots=pivots, 
                        group_size=group_size,embedding_path=embedding_path, pivot_hint=pivot_hint) + [ordered_pivots[index]]
            else:
                current_small = current_small - pivot_result[index-1][0]
                for j in range(index-1):
                    current_small = current_small - pivot_result[j][0]
                result = result + self.sem_sort(query=query, documents=list(current_large), pivot_selection_method=pivot_selection_method, pivots=pivots, 
                        group_size=group_size,embedding_path=embedding_path, pivot_hint=pivot_hint) + [ordered_pivots[index]] + self.sem_sort(query=query, 
                        documents=list(current_small), pivot_selection_method=pivot_selection_method, pivots=pivots, group_size=group_size,embedding_path=embedding_path, pivot_hint=pivot_hint)
        return result

    def stop_models(
        self
    )->None:
        self.instances.stop()