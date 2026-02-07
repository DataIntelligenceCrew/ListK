Note: This is experimental code and not currently a drop-in library. Also the code base is concurrently not cleaned up as of yet.

All code from the three repositories added. Will clean later.

# Reproducibility Instructions
## 1. Setup
From the main folder set up the python enviorment with:
> uv sync && source .venv/bin/activate

and

> uv pip install -e .

Run the download_models.py to download all of the models associated with reproducing the code of the publication:
> python download_models.py

This will make the folder titled models_and_benchmarks. Within the folder we want to make a temp folder for the scifact dataset:
> cd ./models_and_benchmarks && mkdir scifact

We also want to get the movies dataset from sembench (https://sembench.ngrok.io/). Upon aquiring the zipped file run the following:
> unzip \<path to zipped file> -d ./movies

We want to take the Q9.sql file from the sembench github (https://github.com/SemBench/SemBench/) in files/movie/query/gold_sql of the repository and place it within the movies folder:
> cp \<path to movie Q9.sql> ./movies

Return to the main directory:
> cd ..

## 2. Scifact Tests
All of the reproducibility scripts are within the reproducibility_scripts/data/raw folder:
> cd ./reproducibility_scripts/data/raw

Let's run the test case for our implementation of pairwise comparison **(Note: one must specify the number of available GPUs they have)**:
> cd ./comparison_1_data && python generate_data.py **\<number of GPUs>** && cd ..

Afterwards we can produce the sorted result:
> cd ./hgt_data/pairwise_data && python clean_data.py **\<number of GPUs>** && cd ../..

The test case for no embedding base pivot selection:
> cd ./no-em-p && python generate_data.py **\<number of GPUs>** && cd ..

The test case for qwen3-8B running LMPQSelect+Sort:
> cd ./dllmdata/qwen && python generate_data.py **\<number of GPUs>** && python clean_data.py **\<number of GPUs>** && cd ../..

The test case for zephyr-7B running LMPQSelect+Sort:
> cd ./dllmdata/zephyr && python generate_data.py **\<number of GPUs>** && python clean_data.py **\<number of GPUs>** && cd ../..

The test cases for running LMPQSelect+Sort with different $L$ (TFilter):
> cd ./Tour_1 && python generate_data.py **\<number of GPUs>** && cd ..

> cd ./Tour_2 && python generate_data.py **\<number of GPUs>** && cd ..

> cd ./tour5-b && python generate_data.py **\<number of GPUs>** && cd ..

> cd ./Tour_10-b && python generate_data.py **\<number of GPUs>** && cd ..

> cd ./tour_15-b && python generate_data.py **\<number of GPUs>** && cd ..

> cd ./hgt_pair/l1 && python clean_data.py **\<number of GPUs>** && cd ../..

> cd ./hgt_pair/l2 && python clean_data.py **\<number of GPUs>** && cd ../..

> cd ./hgt_pair/l5 && python clean_data.py **\<number of GPUs>** && cd ../..

> cd ./hgt_pair/l10 && python clean_data.py **\<number of GPUs>** && cd ../..

> cd ./hgt_pair/l15 && python clean_data.py **\<number of GPUs>** && cd ../..

The test case for TFilter:
> cd ./tfilter && python generate_data.py **\<number of GPUs>** 260 && cd ..

>cd ./tfilter && python generate_data.py **\<number of GPUs>** 1295 && cd ..

> cd ./tfilter && python generate_data.py **\<number of GPUs>** 2591 && cd ..

> cd ./tfilter && python generate_data.py **\<number of GPUs>** 3887 && cd ..

The test case for TFilter+TSort:
> cd ./tfilter-tsort-data && python generate_data.py **\<number of GPUs>** 260 1 && cd ..

> cd ./tfilter-tsort-data && python generate_data.py **\<number of GPUs>** 1295 5 && cd ..

> cd ./tfilter-tsort-data && python generate_data.py **\<number of GPUs>** 2591 10 && cd ..

> cd ./tfilter-tsort-data && python generate_data.py **\<number of GPUs>** 3887 15 && cd ..

The test case for TSort:
> cd ./tourk5000 && python clean_data.py **\<number of GPUs>** && cd ..

The test case for LMPQSelect with different $K$:

>cd ./k5000 && python generate_data.py **\<number of GPUs>** 10 && cd ..

>cd ./k5000 && python generate_data.py **\<number of GPUs>** 20 && cd ..

>cd ./k5000 && python generate_data.py **\<number of GPUs>** 50 && cd ..

>cd ./k5000 && python generate_data.py **\<number of GPUs>** 100 && cd ..

The test case for LMPQSort:

>cd ./sort && python generate_data.py **\<number of GPUs>** 20 && cd ..

>cd ./sort && python generate_data.py **\<number of GPUs>** 50 && cd ..

>cd ./sort && python generate_data.py **\<number of GPUs>** 100 && cd ..

>cd ./sort && python generate_data.py **\<number of GPUs>** 250 && cd ..

>cd ./sort && python generate_data.py **\<number of GPUs>** 500 && cd ..

>cd ./sort && python generate_data.py **\<number of GPUs>** 750 && cd ..

>cd ./sort && python generate_data.py **\<number of GPUs>** 1000 && cd ..

>cd ./sort/hgt_metrics && python clean_data.py **\<number of GPUs>** 20 && cd ../..

>cd ./sort/hgt_metrics && python clean_data.py **\<number of GPUs>** 50 && cd ../..

>cd ./sort/hgt_metrics && python clean_data.py **\<number of GPUs>** 100 && cd ../..

>cd ./sort/hgt_metrics && python clean_data.py **\<number of GPUs>** 250 && cd ../..

>cd ./sort/hgt_metrics && python clean_data.py **\<number of GPUs>** 500 && cd ../..

>cd ./sort/hgt_metrics && python clean_data.py **\<number of GPUs>** 750 && cd ../..

>cd ./sort/hgt_metrics && python clean_data.py **\<number of GPUs>** 1000 && cd ../..

The case for LMPQSelect+Sort with different list sizes:
>cd ./5000w && python generate_data.py **\<number of GPUs>** 4 && cd ..

>cd ./5000w && python generate_data.py **\<number of GPUs>** 8 && cd ..

>cd ./5000w && python generate_data.py **\<number of GPUs>** 16 && cd ..

>cd ./5000w && python generate_data.py **\<number of GPUs>** 32 && cd ..

>cd ./5000w && python generate_data.py **\<number of GPUs>** 48 && cd ..

>cd ./5000w && python generate_data.py **\<number of GPUs>** 64 && cd ..

>cd ./5000w && python generate_data.py **\<number of GPUs>** 128 && cd ..

>cd ./hgt_pair/w128 && python clean_data.py **\<number of GPUs>** && cd ../..

The case for LMPQSelect with differnet $P$ and $P'$ values with early stopping:

>cd ./5000e && python generate_data.py **\<number of GPUs>** 1 1 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 2 1 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 2 2 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 4 1 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 4 2 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 4 4 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 8 1 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 8 2 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 8 4 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 8 8 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 16 1 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 16 2 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 16 4 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 16 8 && cd ..

>cd ./5000e && python generate_data.py **\<number of GPUs>** 16 16 && cd ..

Trying different List Sizes for Top Few Sort:
> cd ./wsortdata && python clean_data.py **\<number of GPUs>** 2 && cd ..

> cd ./wsortdata && python clean_data.py **\<number of GPUs>** 4 && cd ..

> cd ./wsortdata && python clean_data.py **\<number of GPUs>** 8 && cd ..

> cd ./wsortdata && python clean_data.py **\<number of GPUs>** 9 && cd ..

> cd ./wsortdata && python clean_data.py **\<number of GPUs>** 10 && cd ..

The test case for LMPQSelect with differnet $P$ and $P'$ values with no early stopping **(Note: This can take substantial ammount of time a copy of all the test data generated for the publication is also included at [location])**:

>cd ./5000 && python generate_data.py **\<number of GPUs>** 2 1 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 4 1 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 4 2 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 8 1 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 8 2 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 8 4 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 16 1 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 16 2 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 16 4 && cd ..

>cd ./5000 && python generate_data.py **\<number of GPUs>** 16 8 && cd ..

The next few tests utilize LOTUS which uses vllm serve for local LLMs and example of running vllm serve is as follows **(Note: It is recommended that this be run in another terminal instance within the same python enviorment)**:
>  vllm serve **\<model path>** --gpu-memory-utilization 0.85 --max-model-len 4096 --host localhost --data-parallel-size **\<number of GPUs>**

Start the instance for qwen3-8B and run the following test cases:
> cd ./lotusk/qwen_data && python generate_data.py **\<path to model>** && cd ../..

> cd ./lotusk/qwen_data && python clean_data.py && cd ../..

> cd ./bier_pointwise/qwen && python generate_data.py **\<path to model>** && cd ../..

> cd ./bier_pointwise/qwen && python clean_data.py && cd ../..

Start the instance for zephyr-7B and run the following test cases:
> cd ./lotusk/combined_z7b && python generate_data.py **\<path to model>** && cd ../..

> cd ./lotusk/combined_z7b && python clean_data.py && cd ../..

> cd ./bier_pointwise/zephyr && python generate_data.py **\<path to model>** && cd ../..

> cd ./bier_pointwise/zephyr && python clean_data.py && cd ../..


## 3. Sembench Tests
Run the setup for sembench and note the path to the **reviews.csv** and **movies.csv** files generated by the script:
> cd ./sembench_data && python sembench_test.py && cd ..

Start the instance for zephyr-7B and run the following test cases:
> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 1 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 2 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 3 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 4 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 5 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 6 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 7 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 8 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 9 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** z 10 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

Start the instance for qwen3-8B and run the following test cases:
> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 1 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 2 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 3 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 4 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 5 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 6 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 7 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 8 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 9 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/pointwise && python sembench_test.py **\<path to model>** qw 10 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

The test for Sembench with LMPQSort and Pairwise
> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 1 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 2 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 3 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 4 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 5 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 6 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 7 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 8 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 9 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 2 10 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 1 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 2 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 3 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 4 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 5 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 6 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 7 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 8 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 9 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/sort && python sembench_test.py **\<number of GPUs>** 20 10 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

The test case for Sembench with TSort:
> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 1 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 2 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 3 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 4 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 5 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 6 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 7 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 8 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 9 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

> cd ./sembench_data/tour && python sembench_test.py **\<number of GPUs>** 10 **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../..

Top few recomputation:
> cd ./sembench/recomputed/rz20 && python sembench_test.py **\<number of GPUs>** **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../../..

> cd ./sembench/recomputed/tour && python sembench_test.py **\<number of GPUs>** **\<path to reviews.csv>** **\<path to movies.csv>** && cd ../../..

## 4. Wrapup and Plotting