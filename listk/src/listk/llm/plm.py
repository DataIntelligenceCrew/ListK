import time
import torch
import multiprocessing
import os
import psutil
from lm import RankZephyrLM
from lm import GenericRankLM
from typing import Any

class MultiGenericRankLM:
    '''
        Allows for Multiple RankZephyr instances to run on multiple gpus.
        Requires and Assumes you have Cuda.
    '''
    process = []
    init_time: float
    num_cuda_devices = torch.cuda.device_count()

    def __init__(
        self, 
        model_path: str, 
        n_devices: int = 1,
        prompt_template_path: str = "",
        max_tokens: int = 4096,
        window_size: str = 20,
        batch_size: int = 32,
    ) -> None:
        '''
            Loads instances of rankzephyr
        '''
        if n_devices > MultiGenericRankLM.num_cuda_devices:
            self.n_devices = MultiGenericRankLM.num_cuda_devices
        elif n_devices == 0:
            self.n_devices = 1
        else:
            self.n_devices = n_devices
        self .model_path = model_path
        self.prompt_template_path = prompt_template_path
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.batch_size = batch_size
        self.task_queue = [multiprocessing.Queue() for i in range(self.n_devices)]
        self.result_queue = [multiprocessing.Queue() for i in range(self.n_devices)]
        self.in_use = [multiprocessing.Lock() for i in range(self.n_devices)]
        self.start()

    def start(
        self
    )-> None:
        '''
            Starts the number of rankzephyr processes you want (limited to number of gpus)
        '''
        for i in range(self.n_devices):
            p = multiprocessing.Process(target=self.open_process, args= (i,))
            MultiGenericRankLM.process.append(p)
            p.start()
    
    def stop(
        self
    )-> None:
        '''
            Stops all rankzephyr processes
        '''
        i = 0
        for p in MultiGenericRankLM.process:
            self.task_queue[i].put(None)
            p.join()
            i = i + 1

    def open_process(
        self,
        num: int
    )->None:
        '''
            Creates a rankzephyr process that will pread from a queue and then produce an output
        '''
        torch.cuda.set_device(num)
        this_gen = GenericRankLM(model_path=self.model_path,prompt_template_path=self.prompt_template_path,
                                        max_tokens=self.max_tokens, device="cuda", window_size=self.window_size,
                                        batch_size=self.batch_size)
        while True:
            task = self.task_queue[num].get()
            if task is None:
                break
            queries, task_id = task
            result = this_gen.call(queries)
            self.result_queue[num].put((task_id, result))
    
    def bin(
        self,
        items: list[Any],
        no_bins: int = 1,
    )->tuple[list[list[int]], list[list[Any]]]:
        '''
            Organizes outputs into bins
        '''
        item_bins = []
        index_bins = []
        for i in range(no_bins):
            item_bins.append([])
            index_bins.append([])
        current_bin_count = 0
        index = 0
        for i in items:
            if current_bin_count >= no_bins:
                current_bin_count = 0
            item_bins[current_bin_count].append(i)
            index_bins[current_bin_count].append(index)
            current_bin_count = current_bin_count + 1
            index = index + 1
        return index_bins, item_bins
    
    def process_handler(
        self,
        items: list[list[dict[str:Any]]],
        output_list: list[Any],
        index: int,
    )->None:
        '''
            Takes queries and a index + output location for call to process the queries
        '''
        self.in_use[index].acquire()
        self.task_queue[index].put((items, index))
        output_list[index] = self.result_queue[index].get()[1]
        self.in_use[index].release()

    def process_output(
        self,
        outputs: list[Any],
        index_bins = list[list[int]]
    )-> tuple[bool, float, list[tuple[str,list[str]]]]:
        '''
            Combines multiple outputs into one output
        '''
        texts = []
        index = []
        out_bool = True
        out_float = 0.0
        i = 0
        for item in outputs:
            if item != None:
                out_bool = out_bool and item[0]
                if out_float <= item[1]:
                    out_float = item[1]
                texts = texts + item[2]
                index = index + index_bins[i]
            i = i + 1
        out_index, out_texts = zip(*sorted(zip(index, texts)))
        return (out_bool, out_float, out_texts)

    def call_i(
        self, 
        queries: list[list[dict[str:Any]]],
        index: int = 0,
    )-> tuple[bool,float,list[tuple[str,list[str]]]]:
        '''
            Calls a specific vllm instance 
        '''
        self.in_use[index].acquire()
        self.task_queue[index].put((queries, index))
        output = self.result_queue[index].get()[1]
        self.in_use[index].release()
        return output
    
    def call(
        self, 
        queries: list[list[dict[str:Any]]],
    )-> tuple[bool,float,list[tuple[str,list[str]]]]:
        '''
            Calls all vllm instances splitting the queries evenly among them
        '''
        indexes, binned_queries = self.bin(queries, self.n_devices)
        outputs = multiprocessing.Manager().list([None]*self.n_devices)
        internal_process = []
        for i in range(self.n_devices):
            if binned_queries[i] != []:
                p = multiprocessing.Process(target = self.process_handler, args = (binned_queries[i], outputs, i))
                internal_process.append(p)
                p.start()
        for p in internal_process:
            p.join()
        return self.process_output(outputs, indexes)

class MultiGenericLM:
    '''
        Allows for Multiple VLLM instances to run on multiple gpus.
        Requires and Assumes you have Cuda.
    '''
    process = []
    init_time: float
    num_cuda_devices = torch.cuda.device_count()

    def __init__( 
        self, 
        model_path: str, 
        temeprature: float = 0.8, 
        top_p: float = 0.95, 
        max_tokens: int = 4096,
        gpu_memory_utilization: float = 0.85,
        n_devices: int = 1,
    ) -> None:
        '''
            Loads instances of vllm
        '''
        if n_devices > MultiGenericLM.num_cuda_devices:
            self.n_devices = MultiGenericLM.num_cuda_devices
        elif n_devices == 0:
            self.n_devices = 1
        else:
            self.n_devices = n_devices
        self .model_path = model_path
        self.temperature = temeprature
        self.top_p = top_p
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_tokens = max_tokens
        self.task_queue = [multiprocessing.Queue() for i in range(self.n_devices)]
        self.result_queue = [multiprocessing.Queue() for i in range(self.n_devices)]
        self.in_use = [multiprocessing.Lock() for i in range(self.n_devices)]
        self.start()

    def start(
        self
    )-> None:
        for i in range(self.n_devices):
            p = multiprocessing.Process(target=self.open_process, args= (i,))
            MultiGenericLM.process.append(p)
            p.start()
            time.sleep(5)
    
    #Fix vllm not actually ending when stop is called
    def stop(
        self
    )-> None:
        i = 0
        for p in MultiGenericLM.process:
            self.task_queue[i].put(None)
            try:
                parent = psutil.Process(p.pid)
                children = parent.children(recursive=True)
                for child in children:
                    child.kill()
                    child.wait(timeout=3)
            except:
                print("Issue Killing Child Processes Memory Leak May Occur")
                p.kill()
            p.join()
            self.task_queue[i].close()
            self.task_queue[i].join_thread()
            self.result_queue[i].close()
            self.result_queue[i].join_thread()
            i = i + 1
        

    def open_process(
        self,
        num: int
    )->None:
        from lm import GenericLM
        os.environ["CUDA_VISIBLE_DEVICES"] = str(num)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        this_vllm = GenericLM(model_path=self.model_path, temeprature = self.temperature, top_p = self.top_p,
                                        max_tokens=self.max_tokens, gpu_memory_utilization = self.gpu_memory_utilization)
        try:
            while True:
                task = self.task_queue[num].get()
                if task is None:
                    break
                else:
                    queries, task_id = task
                    result = this_vllm.call(queries)
                    self.result_queue[num].put((task_id, result))
        except KeyboardInterrupt:
            None

    def bin(
        self,
        items: list[Any],
        no_bins: int = 1,
    )->tuple[list[list[int]], list[list[Any]]]:
        item_bins = []
        index_bins = []
        for i in range(no_bins):
            item_bins.append([])
            index_bins.append([])
        current_bin_count = 0
        index = 0
        for i in items:
            if current_bin_count >= no_bins:
                current_bin_count = 0
            item_bins[current_bin_count].append(i)
            index_bins[current_bin_count].append(index)
            current_bin_count = current_bin_count + 1
            index = index + 1
        return index_bins, item_bins
    
    def process_handler(
        self,
        items: list[tuple[str,list[str]]],
        output_list: list[Any],
        index: int,
    )->None:
        self.in_use[index].acquire()
        self.task_queue[index].put((items, index))
        output_list[index] = self.result_queue[index].get()[1]
        self.in_use[index].release()

    def process_output(
        self,
        outputs: list[Any],
        index_bins = list[list[int]]
    )-> tuple[bool, float, list[tuple[str,list[str]]]]:
        texts = []
        index = []
        out_bool = True
        out_float = 0.0
        i = 0
        for item in outputs:
            if item != None:
                out_bool = out_bool and item[0]
                if out_float <= item[1]:
                    out_float = item[1]
                texts = texts + item[2]
                index = index + index_bins[i]
            i = i + 1
        out_index, out_texts = zip(*sorted(zip(index, texts)))
        return (out_bool, out_float, out_texts)

    def call_i(
        self, 
        queries: list[list[dict[str:Any]]],
        index: int = 0,
    )-> tuple[bool, float, list[(str,str)]]:
        self.in_use[index].acquire()
        self.task_queue[index].put((queries, index))
        output = self.result_queue[index].get()[1]
        self.in_use[index].release()
        return output
    
    def call(
        self, 
        queries: list[list[dict[str:Any]]],
    )-> tuple[bool, float, list[(str,str)]]:
        indexes, binned_queries = self.bin(queries, self.n_devices)
        outputs = multiprocessing.Manager().list([None]*self.n_devices)
        internal_process = []
        for i in range(self.n_devices):
            if binned_queries[i] != []:
                p = multiprocessing.Process(target = self.process_handler, args = (binned_queries[i], outputs, i))
                internal_process.append(p)
                p.start()
                time.sleep(5)
        for p in internal_process:
            p.join()
        return self.process_output(outputs, indexes)

class MultiRankZephyrLM:
    '''
        Allows for Multiple RankZephyr instances to run on multiple gpus.
        Requires and Assumes you have Cuda.
    '''
    process = []
    init_time: float
    num_cuda_devices = torch.cuda.device_count()

    def __init__(
        self, 
        model_path: str, 
        n_devices: int = 1,
        prompt_template_path: str = "",
        max_tokens: int = 4096,
        window_size: str = 20,
        batch_size: int = 32,
    ) -> None:
        '''
            Loads instances of rankzephyr
        '''
        if n_devices > MultiRankZephyrLM.num_cuda_devices:
            self.n_devices = MultiRankZephyrLM.num_cuda_devices
        elif n_devices == 0:
            self.n_devices = 1
        else:
            self.n_devices = n_devices
        self .model_path = model_path
        self.prompt_template_path = prompt_template_path
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.batch_size = batch_size
        self.task_queue = [multiprocessing.Queue() for i in range(self.n_devices)]
        self.result_queue = [multiprocessing.Queue() for i in range(self.n_devices)]
        self.in_use = [multiprocessing.Lock() for i in range(self.n_devices)]
        self.start()

    def start(
        self
    )-> None:
        '''
            Starts the number of rankzephyr processes you want (limited to number of gpus)
        '''
        for i in range(self.n_devices):
            p = multiprocessing.Process(target=self.open_process, args= (i,))
            MultiRankZephyrLM.process.append(p)
            p.start()
    
    def stop(
        self
    )-> None:
        '''
            Stops all rankzephyr processes
        '''
        i = 0
        for p in MultiRankZephyrLM.process:
            self.task_queue[i].put(None)
            p.join()
            i = i + 1

    def open_process(
        self,
        num: int
    )->None:
        '''
            Creates a rankzephyr process that will pread from a queue and then produce an output
        '''
        torch.cuda.set_device(num)
        this_zephyr = RankZephyrLM(model_path=self.model_path,prompt_template_path=self.prompt_template_path,
                                        max_tokens=self.max_tokens, device="cuda", window_size=self.window_size,
                                        batch_size=self.batch_size)
        while True:
            task = self.task_queue[num].get()
            if task is None:
                break
            queries, task_id = task
            result = this_zephyr.call(queries)
            self.result_queue[num].put((task_id, result))
    
    def bin(
        self,
        items: list[Any],
        no_bins: int = 1,
    )->tuple[list[list[int]], list[list[Any]]]:
        '''
            Organizes outputs into bins
        '''
        item_bins = []
        index_bins = []
        for i in range(no_bins):
            item_bins.append([])
            index_bins.append([])
        current_bin_count = 0
        index = 0
        for i in items:
            if current_bin_count >= no_bins:
                current_bin_count = 0
            item_bins[current_bin_count].append(i)
            index_bins[current_bin_count].append(index)
            current_bin_count = current_bin_count + 1
            index = index + 1
        return index_bins, item_bins
    
    def process_handler(
        self,
        items: list[list[dict[str:Any]]],
        output_list: list[Any],
        index: int,
    )->None:
        '''
            Takes queries and a index + output location for call to process the queries
        '''
        self.in_use[index].acquire()
        self.task_queue[index].put((items, index))
        output_list[index] = self.result_queue[index].get()[1]
        self.in_use[index].release()

    def process_output(
        self,
        outputs: list[Any],
        index_bins = list[list[int]]
    )-> tuple[bool, float, list[tuple[str,list[str]]]]:
        '''
            Combines multiple outputs into one output
        '''
        texts = []
        index = []
        out_bool = True
        out_float = 0.0
        i = 0
        for item in outputs:
            if item != None:
                out_bool = out_bool and item[0]
                if out_float <= item[1]:
                    out_float = item[1]
                texts = texts + item[2]
                index = index + index_bins[i]
            i = i + 1
        out_index, out_texts = zip(*sorted(zip(index, texts)))
        return (out_bool, out_float, out_texts)

    def call_i(
        self, 
        queries: list[list[dict[str:Any]]],
        index: int = 0,
    )-> tuple[bool,float,list[tuple[str,list[str]]]]:
        '''
            Calls a specific vllm instance 
        '''
        self.in_use[index].acquire()
        self.task_queue[index].put((queries, index))
        output = self.result_queue[index].get()[1]
        self.in_use[index].release()
        return output
    
    def call(
        self, 
        queries: list[list[dict[str:Any]]],
    )-> tuple[bool,float,list[tuple[str,list[str]]]]:
        '''
            Calls all vllm instances splitting the queries evenly among them
        '''
        indexes, binned_queries = self.bin(queries, self.n_devices)
        outputs = multiprocessing.Manager().list([None]*self.n_devices)
        internal_process = []
        for i in range(self.n_devices):
            if binned_queries[i] != []:
                p = multiprocessing.Process(target = self.process_handler, args = (binned_queries[i], outputs, i))
                internal_process.append(p)
                p.start()
        for p in internal_process:
            p.join()
        return self.process_output(outputs, indexes)