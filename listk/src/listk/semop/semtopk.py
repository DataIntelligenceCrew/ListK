from solicedb.llm.plm import MultiRankZephyrLM
import random
import time
import os
import multiprocessing
from typing import Any

class MultiPivot():
    call_count = 0
    start_time_total = 0.0
    maybe_set_size = []
    definitely_set_size = []
    intermediary_definitely_set = []
    latency = []
    call_counts = []
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

    def pivot_selection(
        self,
        pivots: int,
        query: str,
        documents: list[str],
        method: str,
        k: int = 0,
        embedding_path: str = "",
    )->list[str]:
        '''
            Implement different methods later
        '''
        pivot_list = []
        consider = documents
        if method == "embedding" and k != 0 and embedding_path != "":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            from sentence_transformers import SentenceTransformer
            e_model = SentenceTransformer(embedding_path)
            document_embeddings = e_model.encode(documents)
            query_embedding = e_model.encode([query])
            similarities = e_model.similarity(query_embedding, document_embeddings)[0]
            temp, consider = zip(*sorted(zip(similarities, documents), reverse=True))
            if pivots != 1:
                pivot_list = consider[k:min(len(consider)-1, k+pivots)]
            else:
                pivot_list = [consider[min(len(consider)-1, k)]]
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
        MultiPivot.call_count = MultiPivot.call_count + 1
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
        MultiPivot.call_count = MultiPivot.call_count + len(organized_documents)
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

    def top_k_ne(
        self,
        query: str,
        documents: list[str],
        top_k: int,
        pivot_selection_method: str,
        pivots: int = 1,
        group_size: int = 1,
        embedding_path: str = "",
        pivot_hint: bool = False,
    )->list[str]:
        definitely_set = set()
        if pivots <= 0:
            pivots = 1
        if pivots > self.window_size: #For now it is baked to 20 but this may change later
            pivots = self.window_size - 1
        if group_size <= 0:
            group_size = 1
        if group_size > pivots:
            group_size = pivots
        if group_size == self.window_size:
            group_size = self.window_size - 1
        if len(documents) <= top_k - len(definitely_set):
            return list(documents)
        MultiPivot.call_count = MultiPivot.call_count + 1
        ordered_pivots = self.pivot_selection(pivots=pivots, query=query, documents=documents, method=pivot_selection_method, k = top_k - len(definitely_set), embedding_path = embedding_path)
        organized_pivots = self.group_pivots(group_size=group_size, pivot_list=ordered_pivots)
        #instance_assignment = self.assign_pivot_instance(num_pivots_blocks= len(organized_pivots))
        maybe_set = list(set(documents) - set(ordered_pivots))
        if maybe_set == []:
            if len(ordered_pivots) <= top_k:
                return list(ordered_pivots)
            else:
                return list(ordered_pivots[:top_k])
        while len(definitely_set) != top_k or maybe_set != []:
            #processes = []
            new_maybe_set = set()
            grouped_pivot_result = []
            for p in organized_pivots:
                organized_maybe_set = self.organize_documents(pivot_documents=p, maybe_set=maybe_set, grouping = self.n_devices)
                grouped_pivot_result.append(self.process_pivot(query=query, pivot_list=p, batched_documents=organized_maybe_set, instance = -1, pivot_hint=pivot_hint))
            '''
            pivot_result = []
            for p in range(len(ordered_pivots)):
                organized_maybe_set = self.organize_documents(pivot_document=ordered_pivots[p], maybe_set=maybe_set)
                pivot_result.append(self.process_pivot(query = query, batched_documents = organized_maybe_set, pivot = ordered_pivots[p], instance = instance_assignment[p]))
            
            grouped_pivot_result = multiprocessing.Manager().list([None]*len(organized_pivots))
            for p in range(len(organized_pivots)):
                organized_maybe_set = self.organize_documents(pivot_documents=organized_pivots[p], maybe_set=maybe_set)
                process = multiprocessing.Process(target = self.process_handler, args = (query, organized_pivots[p], instance_assignment[p], organized_maybe_set, grouped_pivot_result, p))
                processes.append(process)
                process.start()
            for p in processes:
                p.join()
            '''
            pivot_result = []
            for p in grouped_pivot_result:
                pivot_result = pivot_result + p
            index = 0
            leave = False
            while index < len(pivot_result) and leave == False:
                current_large, current_small = pivot_result[index]
                if index != 0:
                    current_large = current_large & pivot_result[index-1][1]
                if len(current_large) == top_k - len(definitely_set):
                    definitely_set = definitely_set | current_large
                    return list(definitely_set)
                elif len(current_large) + 1 == top_k - len(definitely_set):
                    definitely_set = definitely_set | current_large | {ordered_pivots[index],}
                    return list(definitely_set)
                elif len(current_large) + 1 < top_k - len(definitely_set):
                    definitely_set = definitely_set | current_large | {ordered_pivots[index],}
                    new_maybe_set = current_small
                elif len(current_large) > top_k - len(definitely_set):
                    new_maybe_set = current_large
                    leave = True
                index = index + 1
            maybe_set = new_maybe_set.copy()
            MultiPivot.maybe_set_size.append(len(maybe_set))
            MultiPivot.latency.append(time.perf_counter() - MultiPivot.start_time_total)
            MultiPivot.call_counts.append(MultiPivot.call_count)
            MultiPivot.definitely_set_size.append(len(definitely_set))
            MultiPivot.intermediary_definitely_set.append(list(definitely_set.copy()))
            if len(maybe_set) <= top_k - len(definitely_set):
                    return list(definitely_set | set(maybe_set))
            elif len(maybe_set) <= self.window_size:
                MultiPivot.call_count = MultiPivot.call_count + 1
                maybe_set = self.handle_batch_call(query=query, documents=[list(maybe_set)])[0]
                if len(maybe_set) <= top_k - len(definitely_set):
                    definitely_set = definitely_set | set(maybe_set)
                    return list(definitely_set)
                else:
                    definitely_set = definitely_set | set(maybe_set[:top_k - len(definitely_set)])
                    return list(definitely_set)
            if len(maybe_set) <= pivots:
                pivots = len(maybe_set) - 1
            if group_size > pivots:
                group_size = pivots
            MultiPivot.call_count = MultiPivot.call_count + 1
            ordered_pivots = self.pivot_selection(pivots=pivots, query=query, documents=list(maybe_set), method=pivot_selection_method, k = top_k - len(definitely_set), embedding_path = embedding_path)
            organized_pivots = self.group_pivots(group_size=group_size, pivot_list=ordered_pivots)
            #instance_assignment = self.assign_pivot_instance(num_pivots_blocks= len(organized_pivots))
            maybe_set = list(maybe_set - set(ordered_pivots))
        return list(definitely_set)
    
    def top_k_e(
        self,
        query: str,
        documents: list[str],
        top_k: int,
        pivot_selection_method: str,
        pivots: int = 1,
        group_size: int = 1,
        embedding_path: str = "",
        pivot_hint: bool = False,
    )->list[str]:
        definitely_set = set()
        if pivots <= 0:
            pivots = 1
        if pivots > self.window_size: #For now it is baked to 20 but this may change later
            pivots = self.window_size - 1
        if group_size <= 0:
            group_size = 1
        if group_size > pivots:
            group_size = pivots
        if group_size == self.window_size:
            group_size = self.window_size - 1
        if len(documents) <= top_k - len(definitely_set):
            return list(documents)
        MultiPivot.call_count = MultiPivot.call_count + 1
        ordered_pivots = self.pivot_selection(pivots=pivots, query=query, documents=documents, method=pivot_selection_method, k = top_k - len(definitely_set), embedding_path = embedding_path)
        organized_pivots = self.group_pivots(group_size=group_size, pivot_list=ordered_pivots)
        #instance_assignment = self.assign_pivot_instance(num_pivots_blocks= len(organized_pivots))
        maybe_set = list(set(documents) - set(ordered_pivots))
        if maybe_set == []:
            if len(ordered_pivots) <= top_k:
                return list(ordered_pivots)
            else:
                return list(ordered_pivots[:top_k])
        while len(definitely_set) != top_k or maybe_set != []:
            new_maybe_set = set()
            pivot_result = []
            p_index = 0
            index = 0
            e_stop = False
            while p_index < len(organized_pivots) and e_stop == False:
                grouped_pivot_result = []
                organized_maybe_set = self.organize_documents(pivot_documents=organized_pivots[p_index], maybe_set=maybe_set, grouping = self.n_devices)
                grouped_pivot_result.append(self.process_pivot(query=query, pivot_list=organized_pivots[p_index], batched_documents=organized_maybe_set, instance = -1, pivot_hint=pivot_hint))
                for p in grouped_pivot_result:
                    pivot_result = pivot_result + p
                while index < len(pivot_result) and e_stop == False:
                    current_large, current_small = pivot_result[index]
                    if index != 0:
                        current_large = current_large & pivot_result[index-1][1]
                    if len(current_large) == top_k - len(definitely_set):
                        definitely_set = definitely_set | current_large
                        return list(definitely_set)
                    elif len(current_large) + 1 == top_k - len(definitely_set):
                        definitely_set = definitely_set | current_large | {ordered_pivots[index],}
                        return list(definitely_set)
                    elif len(current_large) + 1 < top_k - len(definitely_set):
                        definitely_set = definitely_set | current_large | {ordered_pivots[index],}
                        new_maybe_set = current_small
                    elif len(current_large) > top_k - len(definitely_set):
                        new_maybe_set = current_large
                        e_stop = True
                    index = index + 1
                p_index = p_index + 1
            maybe_set = new_maybe_set.copy()
            MultiPivot.maybe_set_size.append(len(maybe_set))
            MultiPivot.latency.append(time.perf_counter() - MultiPivot.start_time_total)
            MultiPivot.call_counts.append(MultiPivot.call_count)
            MultiPivot.definitely_set_size.append(len(definitely_set))
            MultiPivot.intermediary_definitely_set.append(list(definitely_set.copy()))
            if len(maybe_set) <= top_k - len(definitely_set):
                    return list(definitely_set | set(maybe_set))
            elif len(maybe_set) <= self.window_size:
                MultiPivot.call_count = MultiPivot.call_count + 1
                maybe_set = self.handle_batch_call(query=query, documents=[list(maybe_set)])[0]
                if len(maybe_set) <= top_k - len(definitely_set):
                    definitely_set = definitely_set | set(maybe_set)
                    return list(definitely_set)
                else:
                    definitely_set = definitely_set | set(maybe_set[:top_k - len(definitely_set)])
                    return list(definitely_set)
            if len(maybe_set) <= pivots:
                pivots = len(maybe_set) - 1
            if group_size > pivots:
                group_size = pivots
            MultiPivot.call_count = MultiPivot.call_count + 1
            ordered_pivots = self.pivot_selection(pivots=pivots, query=query, documents=list(maybe_set), method=pivot_selection_method, k = top_k - len(definitely_set), embedding_path = embedding_path)
            organized_pivots = self.group_pivots(group_size=group_size, pivot_list=ordered_pivots)
            #instance_assignment = self.assign_pivot_instance(num_pivots_blocks= len(organized_pivots))
            maybe_set = list(maybe_set - set(ordered_pivots))
        return list(definitely_set)

    def embedding_filter(
        self,
        query: str,
        documents: list[str],
        cutoff: int,
        embedding_path: str,
    )->list[str]:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        from sentence_transformers import SentenceTransformer
        e_model = SentenceTransformer(embedding_path)
        document_embeddings = e_model.encode(documents)
        query_embedding = e_model.encode([query])
        similarities = e_model.similarity(query_embedding, document_embeddings)[0]
        temp, consider = zip(*sorted(zip(similarities, documents), reverse=True))
        return consider[:cutoff]
    
    def tournament_filter(
        self,
        query: str,
        documents: list[str],
        cutoff: int,
        leniancy: int,
        kill_loop: int,
    )->list[str]:
        import math
        kill_count = 0
        while kill_count < kill_loop  and (len(documents) < cutoff - leniancy or len(documents) > cutoff + leniancy):
            expected_doc_num = math.ceil(len(documents)/self.window_size)
            gather = 1
            if expected_doc_num < cutoff - leniancy:
                gather = math.ceil(cutoff/expected_doc_num)
            count = 0
            calls = []
            call = []
            for d in documents:
                if count == self.window_size:
                    calls.append(call.copy())
                    call = [d]
                    count = 1
                    MultiPivot.call_count = MultiPivot.call_count + 1
                else:
                    call.append(d)
                    count = count + 1
            calls.append(call)
            MultiPivot.call_count = MultiPivot.call_count + 1
            count = 0
            batches = []
            batch = []
            for c in calls:
                if count == self.batch_size*self.n_devices:
                    batches.append(batch.copy())
                    batch = [c]
                    count = 1
                else:
                    batch.append(c)
                    count = count + 1
            batches.append(batch)
            new_document_set = []
            for b in batches:
                batch_outputs = self.handle_batch_call(query=query, documents=b)
                for items in batch_outputs:
                    new_document_set = new_document_set + items[:gather]
            documents = new_document_set
            kill_count = kill_count + 1
        return documents


    def top_k(
        self,
        query: str,
        documents: list[str],
        top_k: int,
        pivot_selection_method: str = "",
        pivots: int = 1,
        group_size: int = 1,
        early_stop: bool = True,
        embedding_path: str = "",
        embedding_filter: list[bool, int] = [],
        tournament_filter: list[bool, int, int, int] = [],
        pivot_hint: bool = False,
    )->list[str]:
        MultiPivot.start_time_total = time.perf_counter()
        MultiPivot.maybe_set_size = [len(documents)]
        MultiPivot.definitely_set_size = [0]
        MultiPivot.intermediary_definitely_set = [[]]
        MultiPivot.latency = [0.0]
        MultiPivot.call_counts = [0]
        MultiPivot.call_count = 0
        if embedding_filter != [] and embedding_filter[0] and (embedding_filter[1] >= top_k or embedding_filter < len(documents)) and embedding_path != "":
            print("Running: embedding filter")
            documents = self.embedding_filter(query = query, documents=documents, cutoff=embedding_filter[1], embedding_path=embedding_path)
        if tournament_filter != [] and tournament_filter[0] and (tournament_filter[1] >= top_k or tournament_filter < len(documents)) and tournament_filter[2] >= 0 and tournament_filter[3] > 0:
            print("Running: tournament filter")
            documents = self.tournament_filter(query=query, documents=documents, cutoff= tournament_filter[1], leniancy= tournament_filter[2], kill_loop = tournament_filter[3])
        MultiPivot.maybe_set_size.append(len(documents))
        MultiPivot.latency.append(time.perf_counter() - MultiPivot.start_time_total)
        MultiPivot.call_counts.append(MultiPivot.call_count)
        MultiPivot.definitely_set_size.append(0)
        MultiPivot.intermediary_definitely_set.append([])
        if early_stop:
            print("Running: semtopk w/ early stopping")
            result = self.top_k_e(query = query, documents=documents, top_k=top_k, pivot_selection_method=pivot_selection_method, pivots=pivots, group_size=group_size, embedding_path = embedding_path, pivot_hint=pivot_hint)
            MultiPivot.maybe_set_size.append(0)
            MultiPivot.latency.append(time.perf_counter() - MultiPivot.start_time_total)
            MultiPivot.call_counts.append(MultiPivot.call_count)
            MultiPivot.definitely_set_size.append(len(result))
            MultiPivot.intermediary_definitely_set.append(result.copy())
            return result
        else:
            print("Running: semtopk w/o early stopping")
            result = self.top_k_ne(query = query, documents=documents, top_k=top_k, pivot_selection_method=pivot_selection_method, pivots=pivots, group_size=group_size, embedding_path = embedding_path, pivot_hint=pivot_hint)
            MultiPivot.maybe_set_size.append(0)
            MultiPivot.latency.append(time.perf_counter() - MultiPivot.start_time_total)
            MultiPivot.call_counts.append(MultiPivot.call_count)
            MultiPivot.definitely_set_size.append(len(result))
            MultiPivot.intermediary_definitely_set.append(result.copy())
            return result

    def stop_models(
        self
    )->None:
        self.instances.stop()