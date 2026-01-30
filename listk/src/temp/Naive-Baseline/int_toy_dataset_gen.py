import random
import sys
import pandas as pd

def create_random_row(row_size, low, high):
    output = []
    for i in range(row_size):
        output.append(random.randint(low, high))
    return output

def create_random_table(col_size, row_size, low, high):
    output = []
    for i in range(col_size):
        output.append(create_random_row(row_size, low, high))
    return output

#inputs are the row size, the column size, the minimum integer, the maximum integer, and the name of the parquet file (do not include .parquet)
pd.DataFrame(create_random_table(int(sys.argv[2]), int(sys.argv[1]), int(sys.argv[3]), int(sys.argv[4]))).to_parquet(sys.argv[5] + ".parquet", engine="pyarrow", index=False)