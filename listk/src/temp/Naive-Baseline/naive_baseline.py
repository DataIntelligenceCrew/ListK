import requests
import json
import pandas as pd
import multiprocessing

def read_row(row):
    output = []
    for r in row:
        output.append(r)
    return output

def LLM_call(call_string, keyfile, batch, loc):
    print(call_string)
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer " + json.load(open(keyfile, "r"))["key"],
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "nvidia/nemotron-nano-9b-v2:free",
        "messages": [
        {
            "role": "user",
            "content": call_string
        }
        ],
                   
    })
    )
    batch[loc] = response.json()['choices'][0]['message']['content'].strip() 

def test(call, keyfile, batch, j):
    if j == 1:
        print("l")
        batch[j] = "1"
    else:
        print("r")
        batch[j] = "0"

def batch_call(topk, i, b, item, scoring_criteria, keyfile):
    batch = multiprocessing.Manager().list(["0"]*b)
    processes = []
    for j in range(len(batch)):
        if j <= i:
            p = multiprocessing.Process(target=LLM_call, args = ("given i=" + str(item) + ", j=" + str(topk[i-j]) + ", and scoring criteria: [" + scoring_criteria + "] return 1 if i scores higher 0 otherwise (return only the number 1 or 0)", keyfile, batch, j))
            processes.append(p)
            p.start()
    for p in processes:
        p.join()
    return batch

def naivetopk(k, b, data, scoring_criteria):
    topk = []
    for n in range(len(data)):
        item = read_row(data.iloc[n])
        if topk == []:
            topk.append(item)
        else:
            filled = False
            end = False
            i = len(topk) - 1
            while i >= 0 and end == False:
                batch = batch_call(topk, i, b, item, scoring_criteria, "key.json")
                j = 0
                while j < len(batch) and end == False:
                    if i >= j and batch[j] == "0":
                        filled = True
                        if i-j == len(topk)-1:
                            topk.append(item)
                        else:
                            topk.insert(i - j + 1, item)
                        end = True
                    j = j + 1
                i = i - len(batch)
            if filled == False:
                topk.insert(0, item)
            if len(topk) > k:
                topk = topk[:k]
    return topk

def read_json(json_file):
    data = json.load(open(json_file, "r"))
    return data["K"], data["Criteria"], pd.read_parquet(data["Filename"], engine="pyarrow")

test_data = pd.DataFrame([["divergence", "coagulate", "contextualize"], ["divert", "conflict", "organize"], ["like", "as", "when"]])
test_data.to_parquet("test_data.parquet", engine="pyarrow", index=False)
params = read_json("input.json")
print(naivetopk(params[0], 2, params[2], params[1]))