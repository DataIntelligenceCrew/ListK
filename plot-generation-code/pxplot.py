import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def turn_to_list(input):
    input = input.replace("[", "")
    input = input.replace("]", "")
    input = input.replace(" ", "")
    input = input.replace("\'", "")
    input = input.split(",")
    return input

def get_organized_documents(input):
    data = pd.read_csv(input)
    q_ids = list(data['qid'])
    doc_ids = list(data['did'])
    output = []
    for i in range(len(doc_ids)):
        output.append((q_ids[i], turn_to_list(doc_ids[i]), list(data['time'])))
    return output

def get_rankings(id, path):
    path = path + f'{id}.parquet'
    return list(pd.read_parquet(path)['doc_id'])

def calc_recall(result, k, path):
    recalls = []
    for r in result:
        items = get_rankings(r[0], path)[:k]
        count = 0
        for i in items:
            if i in r[1]:
                count = count + 1
        recalls.append(count/len(r[1]))
    return recalls

def calc_stats_for_5000_e(path, parquet_path):
    nums = [1,2,4,8,16]
    datapoints = []
    for p in nums:
        for x in nums:
            if p >= x:
                temp = path + f'bier_result_unsorted_{p}_{x}_25.csv'
                current = get_organized_documents(temp)
                times = []
                for c in current:
                    times.append(c[2])
                recalls = calc_recall(current, 10, parquet_path)
                datapoints.append([p,x,np.mean(times), np.mean(recalls)])
    return pd.DataFrame(datapoints, columns=['p', 'x', 'time', 'recall'])

def calc_stats_for_5000_e_hgt(path, parquet_path):
    nums = [1,2,4,8,16]
    datapoints = []
    for p in nums:
        for x in nums:
            if p >= x:
                temp = path + f'bier_result_unsorted_{p}_{x}_25.csv'
                current = get_organized_documents(temp)
                times = []
                for c in current:
                    times.append(c[2])
                recalls = list(pd.read_csv(path + f"bier_metrics_{p}_{x}_25.csv")['Recall@10'])[0]
                datapoints.append([p,x,np.mean(times), np.mean(recalls)])
    return pd.DataFrame(datapoints, columns=['p', 'x', 'time', 'recall'])


def calc_stats_for_5000(path, e_path, parquet_path):
    nums = [1,2,4,8,16]
    datapoints = []
    for p in nums:
        for x in nums:
            if p > x:
                temp = path + f'bier_result_unsorted_{p}_{x}_25.csv'
                current = get_organized_documents(temp)
                times = []
                for c in current:
                    times.append(c[2])
                recalls = calc_recall(current, 10, parquet_path)
                datapoints.append([p,x,np.mean(times), np.mean(recalls)])
            elif p == x:
                temp = e_path + f'bier_result_unsorted_{p}_{x}_25.csv'
                current = get_organized_documents(temp)
                times = []
                for c in current:
                    times.append(c[2])
                recalls = calc_recall(current, 10, parquet_path)
                datapoints.append([p,x,np.mean(times), np.mean(recalls)])
    return pd.DataFrame(datapoints, columns=['p', 'x', 'time', 'recall'])

def calc_stats_for_5000_hgt(path, e_path, parquet_path):
    nums = [1,2,4,8,16]
    datapoints = []
    for p in nums:
        for x in nums:
            if p > x:
                temp = path + f'bier_result_unsorted_{p}_{x}_25.csv'
                current = get_organized_documents(temp)
                times = []
                for c in current:
                    times.append(c[2])
                try:
                    recalls = list(pd.read_csv(path + f"bier_metrics_{p}_{x}_25.csv")['Recall@10'])[0]
                except:
                    try:
                        recalls = list(pd.read_csv(path + f"4_1_combined/bier_metrics_{p}_{x}_25.csv")['Recall@10'])[0]
                    except:
                        try:
                            recalls = list(pd.read_csv(path + f"16_1_combined/bier_metrics_{p}_{x}_25.csv")['Recall@10'])[0]
                        except:
                            try:
                                recalls = list(pd.read_csv(path + f"16_2_combined/bier_metrics_{p}_{x}_25.csv")['Recall@10'])[0]
                            except:
                                None
                datapoints.append([p,x,np.mean(times), recalls])
            elif p == x:
                temp = e_path + f'bier_result_unsorted_{p}_{x}_25.csv'
                current = get_organized_documents(temp)
                times = []
                for c in current:
                    times.append(c[2])
                recalls = list(pd.read_csv(e_path + f"bier_metrics_{p}_{x}_25.csv")['Recall@10'])[0]
                datapoints.append([p,x,np.mean(times), recalls])
    return pd.DataFrame(datapoints, columns=['p', 'x', 'time', 'recall'])

def make_half_plot(var_name, dataframe, label):
    nums = [1,2,4,8,16]
    table = np.zeros((len(nums), len(nums)))
    points = []
    for p in range(len(nums)):
        for x in range(len(nums)):
            if nums[p] >= nums[x]:
                table[p][x] = dataframe[(dataframe['p'] == nums[p]) & (dataframe['x'] == nums[x])][var_name]
                points.append(dataframe[(dataframe['p'] == nums[p]) & (dataframe['x'] == nums[x])][var_name])
    mask = np.triu(np.ones_like(table, dtype=bool), k = 1)
    plt.figure(figsize=(10, 8))
    if var_name == 'time':
        sns.heatmap(table, mask=mask, cmap="viridis", vmax=6000, vmin=300,
                center=0, square=True, linewidths=.5, annot=True, fmt='.2f',
                cbar_kws={"shrink": .5}, xticklabels=nums, 
                yticklabels=nums)
    else:
        sns.heatmap(table, mask=mask, cmap="plasma", vmax=1, vmin=0,
                center=0, square=True, linewidths=.5, annot=True, fmt='.2f',
                cbar_kws={"shrink": .5}, xticklabels=nums, 
                yticklabels=nums)
    plt.xlabel("X")
    plt.ylabel("P")
    if var_name == 'time':
        plt.title(f'Latency (s) for P and X combinations with {label}')
    else:
        plt.title(f'Recall@10 for P and X combinations with {label}')
    plt.savefig(f'PvX_{var_name}_{label}.pdf', bbox_inches='tight')
    plt.savefig(f'PvX_{var_name}_{label}.png', bbox_inches='tight')
    plt.close()

def make_half_plot_hgt(var_name, dataframe, label):
    nums = [1,2,4,8,16]
    table = np.zeros((len(nums), len(nums)))
    points = []
    for p in range(len(nums)):
        for x in range(len(nums)):
            if nums[p] >= nums[x]:
                table[p][x] = dataframe[(dataframe['p'] == nums[p]) & (dataframe['x'] == nums[x])][var_name]
                points.append(dataframe[(dataframe['p'] == nums[p]) & (dataframe['x'] == nums[x])][var_name])
    mask = np.triu(np.ones_like(table, dtype=bool), k = 1)
    plt.figure(figsize=(10, 8))
    if var_name == 'time':
        sns.heatmap(table, mask=mask, cmap="viridis", vmax=6000, vmin=300,
                center=0, square=True, linewidths=.5, annot=True, fmt='.2f',
                cbar_kws={"shrink": .5}, xticklabels=nums, 
                yticklabels=nums)
    else:
        sns.heatmap(table, mask=mask, cmap="plasma", vmax=1, vmin=0,
                center=0, square=True, linewidths=.5, annot=True, fmt='.2f',
                cbar_kws={"shrink": .5}, xticklabels=nums, 
                yticklabels=nums)
    plt.xlabel("X")
    plt.ylabel("P")
    if var_name == 'time':
        plt.title(f'Latency (s) for P and X combinations with {label}')
    else:
        plt.title(f'Recall@10 for P and X combinations with {label}')
    plt.savefig(f'PvX_{var_name}_{label}_hgt.pdf', bbox_inches='tight')
    plt.savefig(f'PvX_{var_name}_{label}_hgt.png', bbox_inches='tight')
    plt.close()

def make_box_plots(dataframe1, dataframe2):
    weights = [list(dataframe1['time']), list(dataframe2['time'])]
    labels = ['embedding', 'no embedding']
    fig, ax = plt.subplots()
    ax.set_ylabel('Latency (s)')
    bplot = ax.boxplot(weights, tick_labels=labels)
    plt.title(f'Latency (s) for w/ and w/o Embedding Based Pivot Selection')
    plt.savefig(f'em_and_nem_comparison_.pdf', bbox_inches='tight')
    plt.savefig(f'em_and_nem_comparison_.png', bbox_inches='tight')
    plt.close()

def get_organized_tour_documents(input, input2):
    data1 = pd.read_csv(input)
    data = pd.read_csv(input2)
    q_ids = list(data['qid'])
    doc_ids = list(data1['ids'])
    output = []
    for i in range(len(doc_ids)):
        output.append((q_ids[i], turn_to_list(doc_ids[i]), list(data['time'])))
    return output

def calc_recall_tour(result, k, path):
    recalls = []
    for r in result:
        items = get_rankings(r[0], path)[:k]
        count = 0
        for i in items:
            if i in r[1]:
                count = count + 1
        recalls.append(count/k)
    return recalls

def make_tournament_plot(path, qid_path, vals, labels, k, gt_path):
    dataframes = []
    for v in vals:
        temp = path + f'bier_tfilter_result_{v}_25.csv'
        temp2 = qid_path + f'bier_result_unsorted_16_2_25.csv'
        dataframes.append(get_organized_tour_documents(temp, temp2))
    recalls = []
    for d in dataframes:
        recalls.append(calc_recall_tour(result=d, k=k, path=gt_path))
    ave_recalls = []
    for r in recalls:
        ave_recalls.append(np.mean(r))
    fig, ax = plt.subplots()
    ax.set_ylabel(f'Maximum Possible Recall@{k}')
    ax.set_xlabel(f'L')
    plt.plot(np.arange(len(ave_recalls)) + 1, ave_recalls, color='k')
    bplot = ax.boxplot(recalls, tick_labels=labels)
    plt.title(f'Maximum possible Recall@{k} for different L values')
    plt.savefig(f'tfilter_{k}_plot.pdf', bbox_inches='tight')
    plt.savefig(f'tfilter_{k}_plot.png', bbox_inches='tight')
    plt.close()

def make_k_plot(path, kvals, labels, gt_path):
    dataframes = []
    for v in kvals:
        temp = path + f'bier_result_unsorted_16_2_{v}_25.csv'
        dataframes.append(get_organized_documents(temp))
    vals = []
    for d in range(len(dataframes)):
        recalls = calc_recall(dataframes[d], kvals[d], gt_path)
        vals.append(np.mean(recalls))
    plt.plot(labels, vals, color='k')
    plt.ylim(0, 1)
    plt.xlabel('K')
    plt.ylabel('Recall@K')
    plt.title(f'Recall@K of Multipivot Quickselect for different K values')
    plt.savefig(f'krecallplot.pdf', bbox_inches='tight')
    plt.savefig(f'krecallplot.png', bbox_inches='tight')
    plt.close()

def calc_dcg(ordering, gt):
    count = 1
    dcg = 0.0
    for o in ordering:
        if o in gt:
            dcg = dcg + 1/(math.log2(1+count))
        count = count + 1
    return dcg

def calc_idcg(gt):
    count = 1
    idcg = 0.0
    for i in gt:
        idcg = idcg + 1/(math.log2(1+count))
        count = count + 1
    return idcg

def calc_ndcg(ordering, gt):
    return calc_dcg(ordering, gt)/calc_idcg(gt)

def get_organized_documents_sort(input):
    data = pd.read_csv(input)
    q_ids = list(data['qid'])
    doc_ids = list(data['ids'])
    output = []
    for i in range(len(doc_ids)):
        output.append((q_ids[i], turn_to_list(doc_ids[i]), list(data['time'])))
    return output

def sort_plots(path, gtpath, labels, k):
    vals = [20, 50, 100, 250, 500, 750, 1000]
    dataframes = []
    for v in vals:
        temp = path + f'bier_sort_result_{v}_25.csv'
        dataframes.append(get_organized_documents_sort(temp))
    to_plot_ndcg = []
    to_plot_time = []
    for d in dataframes:
        ndcg = []
        to_plot_time.append(np.mean(d[0][2]))
        for p in d:
            rankings = get_rankings(p[0], gtpath)[:k]
            ndcg.append(calc_ndcg(p[1][:k], rankings))
        to_plot_ndcg.append(ndcg)
    fig, ax = plt.subplots()
    ax.set_ylabel(f'NDCG@{k}')
    ax.set_xlabel('K')
    plt.ylim(0, 1)
    bplot = ax.boxplot(to_plot_ndcg, tick_labels=labels)
    plt.title(f'NDCG@{k} for Semantic Sort at different K')
    plt.savefig(f'sort_ndcg@{k}.pdf', bbox_inches='tight')
    plt.savefig(f'sort_ndcg@{k}.png', bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots()
    ax.set_ylabel('Latency (s)')
    ax.set_xlabel('K')
    plt.plot(labels, to_plot_time, color='k')
    plt.title(f'Latency (s) for Semantic Sort at different K')
    plt.savefig(f'sort_time.pdf', bbox_inches='tight')
    plt.savefig(f'sort_time.png', bbox_inches='tight')
    plt.close()

def sort_plots_hgt(path, gtpath, labels, k):
    vals = [20, 50, 100, 250, 500, 750, 1000]
    dataframes = []
    for v in vals:
        temp = path + f'bier_metrics_sorted_{v}_{v}_10_25.csv'
        dataframes.append(list(pd.read_csv(temp)['NDCG@10'])[0])
    fig, ax = plt.subplots()
    ax.set_ylabel('NDCG@K')
    ax.set_xlabel('K')
    plt.ylim(0, 1)
    plt.plot(labels, dataframes, color='k')
    plt.title(f'Average NDCG@K for Semantic Sort at different K')
    plt.savefig(f'sort_ndcg_hgt.pdf', bbox_inches='tight')
    plt.savefig(f'sort_ndcg_hgt.png', bbox_inches='tight')
    plt.close()

#./comparison_1_data/bier_result_unsorted_1_1_25.csv
#./5000e/bier_result_unsorted_16_2_25.csv
#./Tour_5/bier_result_unsorted_16_2_25.csv
#./Tour_10/bier_result_unsorted_16_2_25.csv
#./Tour_15/bier_result_unsorted_16_2_25.csv
def a_recall_plot(p_path, b_path, path5, path10, path15, path1, path2, pathlotusk, pathw128, gtpath):
    pair = get_organized_documents(p_path)
    base = get_organized_documents(b_path)
    t5 = get_organized_documents(path5)
    t10 = get_organized_documents(path10)
    t15 = get_organized_documents(path15)
    t1 = get_organized_documents(path1)
    t2 = get_organized_documents(path2)
    lotusk = get_organized_documents(pathlotusk)
    w128 = get_organized_documents(pathw128)
    precall = calc_recall(pair, 10, gtpath)
    brecall = calc_recall(base, 10, gtpath)
    t5recall = calc_recall(t5, 10, gtpath)
    t10recall = calc_recall(t10, 10, gtpath)
    t15recall = calc_recall(t15, 10, gtpath)
    t1recall = calc_recall(t1, 10, gtpath)
    t2recall = calc_recall(t2, 10, gtpath)
    lkrecall = calc_recall(lotusk, 10, gtpath)
    r128 = calc_recall(w128, 10, gtpath)
    print(get_rankings(3, gtpath)[:10])
    x = [np.mean(pair[0][2]), np.mean(base[0][2]), np.mean(t5[0][2]), np.mean(t10[0][2]), np.mean(t15[0][2]), np.mean(t1[0][2]), np.mean(t2[0][2]), 
        np.mean(lotusk[0][2]), np.mean(w128[0][2])]
    y = [np.mean(precall), np.mean(brecall), np.mean(t5recall), np.mean(t10recall), np.mean(t15recall), np.mean(t1recall), np.mean(t2recall),
        np.mean(lkrecall), np.mean(r128)]
    labels = ['pairwise', 'L=20', 'L=5', 'L=10', 'L=15', 'L=1', "L=2", "Lotus", 'W = 128']
    colors = ['black', 'red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'grey', 'violet']
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.scatter(x[i], y[i], c=colors[i], label = labels[i])
    #for i in range(len(labels)):
        #ax.annotate(labels[i], (x[i], y[i]))
    ax.legend()
    m, b = np.polyfit(x, y, 1)
    plt.xlim(0,max(x)+500)
    plt.ylim(0,1)
    plt.plot(x, m*np.array(x) + b, linestyle='dotted')
    plt.xlabel('Latency (s)')
    plt.ylabel('Recall@10')
    plt.title('Recall@10 vs Latency (s)')
    plt.savefig(f'main_plot_recall@10.pdf', bbox_inches='tight')
    plt.savefig(f'main_plot_recall@10.png', bbox_inches='tight')
    plt.close()

def window_size_plot(gtpath, ppath, path20, pathw):
    pair = get_organized_documents(ppath)
    base = get_organized_documents(path20)
    nums = [4,8,16,32,48,64,128]
    vals = [pair, base]
    times = [np.mean(pair[0][2]), np.mean(base[0][2]),]
    for n in nums:
        temp = pathw + f'bier_result_unsorted_E_{n}_25_A.csv'
        val = get_organized_documents(temp)
        vals.append(val)
        times.append(np.mean(val[0][2]))
    recalls = []
    xvals = [2,20,4,8,16,32,48,64,128]
    ticks = [2,4,8,16,20,32,48,64,128]
    for v in vals:
        recalls.append(np.mean(calc_recall(v, 10, gtpath)))
    labels = ['W=2', 'W=20', 'W=4', 'W=8', 'W=16', 'W=32', 'W=48', 'W=64', 'W=128']
    colors = ['red', 'black', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown']
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.scatter(xvals[i], recalls[i], c=colors[i], label = labels[i])
    #for i in range(len(labels)):
        #ax.annotate(labels[i], (times[i], recalls[i]))
    ax.legend()
    plt.ylim(0,1)
    plt.xlabel('W')
    plt.ylabel('Recall@10')
    plt.title('Recall@10 vs W')
    plt.savefig(f'w_recall@10.pdf', bbox_inches='tight')
    plt.savefig(f'w_recall@10.png', bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.scatter(xvals[i], times[i], c=colors[i], label = labels[i])
    ax.legend()
    plt.ylim(0,np.max(times) + 100)
    plt.xlabel('w')
    plt.ylabel('Latency (s)')
    plt.title('Latency (s) vs W')
    plt.savefig(f'w_latency.pdf', bbox_inches='tight')
    plt.savefig(f'w_latency.png', bbox_inches='tight')
    plt.close()
    fig, ax = plt.subplots()

def a_plots_hgt():
    pair = pd.read_csv('./hgt_data/pairwise_data/bier_metrics_sorted_1_1_2_2_10_25.csv')
    base = pd.read_csv('./hgt_data/16_2_data/bier_metrics_sorted_16_2_20_20_10_25.csv')
    t5 = pd.read_csv('./hgt_data/l5data/bier_metrics_sorted_16_2_5_20_10_25.csv')
    t10 = pd.read_csv('./hgt_data/l10data/bier_metrics_sorted_16_2_10_20_10_25.csv')
    t15 = pd.read_csv('./hgt_data/l15data/bier_metrics_sorted_16_2_15_20_10_25.csv')
    t1 = pd.read_csv('./hgt_data/l1data/bier_metrics_sorted_16_2_1_20_10_25.csv')
    t2 = pd.read_csv('./hgt_data/l2data/bier_metrics_sorted_16_2_2_20_10_25.csv')
    lotusk = pd.read_csv('./lotusk/combined_z7b/bier_metrics_lotus_10_25.csv')
    lotusk2 = pd.read_csv('./lotusk/qwen_data/bier_metrics_lotus_10_25.csv')
    w128 = pd.read_csv('./hgt_data/w128data/bier_metrics_sorted_1_1_1_128_10_25.csv')
    tour = pd.read_csv('./tourk5000/bier_metrics_tour_10_25.csv')
    x = [list(pair['time'])[0], list(base['time'])[0], list(t5['time'])[0], list(t10['time'])[0], list(t15['time'])[0], list(t1['time'])[0], list(t2['time'])[0], 
        list(lotusk['time'])[0], list(w128['time'])[0], list(lotusk2['time'])[0], list(tour['time'])[0]]
    y = [list(pair['Recall@10'])[0], list(base['Recall@10'])[0], list(t5['Recall@10'])[0], list(t10['Recall@10'])[0], list(t15['Recall@10'])[0], list(t1['Recall@10'])[0], list(t2['Recall@10'])[0], 
        list(lotusk['Recall@10'])[0], list(w128['Recall@10'])[0], list(lotusk2['Recall@10'])[0], list(tour['Recall@10'])[0]]
    labels = ['pairwise', 'L=20', 'L=5', 'L=10', 'L=15', 'L=1', "L=2", "LOTUS (Zephyr-7B)", 'W = 128', "LOTUS (QWEN3-8B)", 'Tournament Sort']
    colors = ['black', 'red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'grey', 'violet', 'pink', 'brown']
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.scatter(x[i], y[i], c=colors[i], label = labels[i])
    #for i in range(len(labels)):
        #ax.annotate(labels[i], (x[i], y[i]))
    ax.legend()
    m, b = np.polyfit(x, y, 1)
    plt.xlim(0,max(x)+500)
    plt.ylim(0,1)
    #plt.plot(x, m*np.array(x) + b, linestyle='dotted')
    plt.xlabel('Latency (s)')
    plt.ylabel('Recall@10')
    plt.title('Recall@10 vs Latency (s)')
    plt.savefig(f'hgt_main_plot_recall@10.pdf', bbox_inches='tight')
    plt.savefig(f'hgt_main_plot_recall@10.png', bbox_inches='tight')
    plt.close()
    x = [list(pair['time'])[0], list(base['time'])[0], list(t5['time'])[0], list(t10['time'])[0], list(t15['time'])[0], list(t1['time'])[0], list(t2['time'])[0], 
        list(lotusk['time'])[0], list(w128['time'])[0], list(lotusk2['time'])[0], list(tour['time'])[0]]
    y = [list(pair['NDCG@10'])[0], list(base['NDCG@10'])[0], list(t5['NDCG@10'])[0], list(t10['NDCG@10'])[0], list(t15['NDCG@10'])[0], list(t1['NDCG@10'])[0], list(t2['NDCG@10'])[0], 
        list(lotusk['NDCG@10'])[0], list(w128['NDCG@10'])[0], list(lotusk2['NDCG@10'])[0], list(tour['NDCG@10'])[0]]
    labels = ['pairwise', 'L=20', 'L=5', 'L=10', 'L=15', 'L=1', "L=2", "LOTUS (Zephyr-7B)", 'W = 128', "LOTUS (QWEN3-8B)", 'Tournament Sort']
    colors = ['black', 'red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'grey', 'violet', 'pink', 'brown']
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.scatter(x[i], y[i], c=colors[i], label = labels[i])
    #for i in range(len(labels)):
        #ax.annotate(labels[i], (x[i], y[i]))
    ax.legend()
    m, b = np.polyfit(x, y, 1)
    plt.xlim(0,max(x)+500)
    plt.ylim(0,1)
    #plt.plot(x, m*np.array(x) + b, linestyle='dotted')
    plt.xlabel('Latency (s)')
    plt.ylabel('NDCG@10')
    plt.title('NDCG@10 vs Latency (s)')
    plt.savefig(f'hgt_main_plot_ndcg@10.pdf', bbox_inches='tight')
    plt.savefig(f'hgt_main_plot_ndcg@10.png', bbox_inches='tight')
    plt.close()

def wsort_plot():
    datapoints = [2,4,8,9,10]
    d2 = pd.read_csv('./wsortdata/bier_metrics_sorted_16_2_20_2_10_25.csv')
    d4 = pd.read_csv('./wsortdata/bier_metrics_sorted_16_2_20_4_10_25.csv')
    d8 = pd.read_csv('./wsortdata/bier_metrics_sorted_16_2_20_8_10_25.csv')
    d9 = pd.read_csv('./wsortdata/bier_metrics_sorted_16_2_20_9_10_25.csv')
    d10 = pd.read_csv('./wsortdata/bier_metrics_sorted_16_2_20_10_10_25.csv')
    x = [list(d2['time'])[0], list(d4['time'])[0], list(d8['time'])[0], list(d9['time'])[0], list(d10['time'])[0]]
    y = [list(d2['NDCG@10'])[0], list(d4['NDCG@10'])[0], list(d8['NDCG@10'])[0], list(d9['NDCG@10'])[0], list(d10['NDCG@10'])[0]]
    labels = ['w=2', 'w=4', 'w=8', 'w=9', 'w=10']
    colors = ['red', 'orange', 'yellow', 'green', 'black']
    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.scatter(x[i], y[i], c=colors[i], label = labels[i])
    ax.legend()
    m, b = np.polyfit(x, y, 1)
    plt.xlim(0,max(x)+500)
    plt.ylim(0,1)
    #plt.plot(x, m*np.array(x) + b, linestyle='dotted')
    plt.xlabel('Latency (s)')
    plt.ylabel('NDCG@10')
    plt.title('NDCG@10 vs Latency (s)')
    plt.savefig(f'sortw_ndcg@10.pdf', bbox_inches='tight')
    plt.savefig(f'sortw_ndcg@10.png', bbox_inches='tight')
    plt.close()

    

e5000 = calc_stats_for_5000_e('./5000e/', './llm-topk-gt/data/phase7_combined_rankings/scifact/')
ne5000 = calc_stats_for_5000('./5000/', './5000e/', './llm-topk-gt/data/phase7_combined_rankings/scifact/')
make_half_plot('time', e5000, 'Early Stopping')
make_half_plot('recall', e5000, 'Early Stopping')
make_half_plot('time', ne5000, 'no Early Stopping')
make_half_plot('recall', ne5000, 'no Early Stopping')
make_box_plots(pd.read_csv('./5000e/bier_result_unsorted_16_2_25.csv'), pd.read_csv('./no-em-p/bier_result_unsorted_16_2_25.csv'))
make_tournament_plot('./tfilter/', './5000e/', [260,1295,2591,3887], ['1', '5', '10', '15'], 5, './llm-topk-gt/data/phase7_combined_rankings/scifact/')
make_tournament_plot('./tfilter/', './5000e/', [260,1295,2591,3887], ['1', '5', '10', '15'], 10, './llm-topk-gt/data/phase7_combined_rankings/scifact/')
make_tournament_plot('./tfilter/', './5000e/', [260,1295,2591,3887], ['1', '5', '10', '15'], 20, './llm-topk-gt/data/phase7_combined_rankings/scifact/')
make_tournament_plot('./tfilter/', './5000e/', [260,1295,2591,3887], ['1', '5', '10', '15'], 50, './llm-topk-gt/data/phase7_combined_rankings/scifact/')
make_tournament_plot('./tfilter/', './5000e/', [260,1295,2591,3887], ['1', '5', '10', '15'], 100, './llm-topk-gt/data/phase7_combined_rankings/scifact/')
make_k_plot('./k5000/', [10, 20, 50, 100], ['10', '20', '50', '100'], './llm-topk-gt/data/phase7_combined_rankings/scifact/')
sort_plots('./sort/', './llm-topk-gt/data/phase7_combined_rankings/scifact/', ['20', '50', '100', '250', '500', '750', '1000'], 5)
sort_plots('./sort/', './llm-topk-gt/data/phase7_combined_rankings/scifact/', ['20', '50', '100', '250', '500', '750', '1000'], 10)
sort_plots('./sort/', './llm-topk-gt/data/phase7_combined_rankings/scifact/', ['20', '50', '100', '250', '500', '750', '1000'], 20)
#./comparison_1_data/bier_result_unsorted_1_1_25.csv
#./5000e/bier_result_unsorted_16_2_25.csv
#./Tour_5/bier_result_unsorted_16_2_25.csv
#./Tour_10/bier_result_unsorted_16_2_25.csv
#./Tour_15/bier_result_unsorted_16_2_25.csv
a_recall_plot('./comparison_1_data/bier_result_unsorted_1_1_25.csv', './5000e/bier_result_unsorted_16_2_25.csv', './tour5-b/bier_result_unsorted_16_2_25.csv', 
              './Tour_10-b/bier_result_unsorted_16_2_25.csv', './tour_15-b/bier_result_unsorted_16_2_25.csv', './Tour_1/bier_result_unsorted_16_2_25.csv', './Tour_2/bier_result_unsorted_16_2_25.csv',
               './lotusk/qwen_data/bier_lotus_semtopk_result.csv', './5000w/bier_result_unsorted_E_128_25_A.csv','./llm-topk-gt/data/phase7_combined_rankings/scifact/')
window_size_plot('./llm-topk-gt/data/phase7_combined_rankings/scifact/', './comparison_1_data/bier_result_unsorted_1_1_25.csv', './5000e/bier_result_unsorted_1_1_25.csv', './5000w/')
a_plots_hgt()
wsort_plot()

e5000_hgt = calc_stats_for_5000_e_hgt('./5000e/', './llm-topk-gt/data/phase7_combined_rankings/scifact/')
ne5000_hgt = calc_stats_for_5000_hgt('./5000/', './5000e/', './llm-topk-gt/data/phase7_combined_rankings/scifact/')
make_half_plot_hgt('recall', e5000_hgt, 'Early Stopping')
make_half_plot_hgt('recall', ne5000_hgt, 'no Early Stopping')
sort_plots_hgt('./sort/hgt_metrics/', './llm-topk-gt/data/phase7_combined_rankings/scifact/', ['20', '50', '100', '250', '500', '750', '1000'], 5)
