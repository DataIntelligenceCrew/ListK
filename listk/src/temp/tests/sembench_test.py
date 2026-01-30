from semtopk import MultiPivot
from semsort import MultiPivot_sort
from semtour import MultiPivot_tour
from gensemtopk import GenMultiPivot
from gensemtour import GenMultiPivot_tour
from gensemsort import GenMultiPivot_sort
import time
import random
import pandas as pd
import json
import torch
import gc
import numpy as np
import argparse
from pathlib import Path
from collections import Counter
import re
import os
import duckdb

#Coppied from Sembench generate_data.py from Movies Scenario
def load_data(data_path):
    """Load and clean the raw data."""
    print("Loading data...")
    movies_df = pd.read_csv(f"{data_path}/rotten_tomatoes_movies.csv")
    reviews_df = pd.read_csv(f"{data_path}/rotten_tomatoes_movie_reviews.csv")
    
    print(f"Loaded {len(movies_df)} movies and {len(reviews_df)} reviews")
    
    # Filter out reviews with null reviewText
    reviews_df = reviews_df.dropna(subset=['reviewText'])
    print(f"After filtering null reviewText: {len(reviews_df)} reviews")
    
    return movies_df, reviews_df

def find_pattern_movie(reviews_df):
    """Find the movie with largest number of reviews following same score pattern."""
    print("Finding movie with consistent score pattern...")
    
    # Pre-filter for /5 and /10 patterns only
    pattern_5_reviews = reviews_df[reviews_df['originalScore'].str.contains('/5', na=False)]
    pattern_10_reviews = reviews_df[reviews_df['originalScore'].str.contains('/10', na=False)]
    
    best_movie = None
    best_pattern = None
    best_score = 0
    
    # Check /5 pattern
    if len(pattern_5_reviews) > 0:
        movie_counts = pattern_5_reviews.groupby('id').size()
        for movie_id, count in movie_counts.items():
            if count >= 50:  # Need at least 50 reviews
                unique_scores = pattern_5_reviews[pattern_5_reviews['id'] == movie_id]['originalScore'].nunique()
                score = count * unique_scores  # Prefer more reviews with more diversity
                if score > best_score:
                    best_movie = movie_id
                    best_pattern = '/5'
                    best_score = score
    
    # Check /10 pattern
    if len(pattern_10_reviews) > 0:
        movie_counts = pattern_10_reviews.groupby('id').size()
        for movie_id, count in movie_counts.items():
            if count >= 50:  # Need at least 50 reviews
                unique_scores = pattern_10_reviews[pattern_10_reviews['id'] == movie_id]['originalScore'].nunique()
                score = count * unique_scores
                if score > best_score:
                    best_movie = movie_id
                    best_pattern = '/10'
                    best_score = score
    
    print(f"Selected pattern movie: {best_movie} with {best_pattern} pattern")
    return best_movie, best_pattern


def get_negative_movie():
    """Use hardcoded negative movie."""
    print("Using hardcoded negative movie: taken_3")
    return 'taken_3'

def get_top_movies_fast(reviews_df, top_n=200):
    """Get top movies by review count quickly."""
    print(f"Getting top {top_n} movies by review count...")
    movie_counts = reviews_df.groupby('id').size().sort_values(ascending=False)
    return movie_counts.head(top_n).index.tolist()


def sample_reviews(reviews_df, pattern_movie, pattern_pattern, negative_movie, top_movies, scale_factor):
    """Sample reviews using movie-first strategy."""
    print(f"Sampling {scale_factor} reviews using movie-first strategy...")
    
    selected_reviews = []
    used_movies = set()
    
    # Step 1: Add pattern movie reviews (ONLY those matching the pattern)
    print(f"Step 1: Adding pattern movie {pattern_movie} (only {pattern_pattern} reviews)")
    pattern_movie_reviews = reviews_df[
        (reviews_df['id'] == pattern_movie) & 
        (reviews_df['originalScore'].str.contains(pattern_pattern, na=False))
    ]
    selected_reviews.append(pattern_movie_reviews)
    used_movies.add(pattern_movie)
    remaining = scale_factor - len(pattern_movie_reviews)
    print(f"Added {len(pattern_movie_reviews)} pattern reviews, remaining: {remaining}")
    
    # Step 2: Add negative movie reviews (ALL reviews, no pattern filtering)
    if negative_movie != pattern_movie and remaining > 0:
        print(f"Step 2: Adding negative movie {negative_movie} (all reviews)")
        negative_movie_reviews = reviews_df[reviews_df['id'] == negative_movie]
        if len(negative_movie_reviews) > 0:
            sample_size = min(len(negative_movie_reviews), remaining // 3)  # Use up to 1/3 of remaining
            sampled = negative_movie_reviews.sample(n=sample_size, random_state=42)
            selected_reviews.append(sampled)
            used_movies.add(negative_movie)
            remaining -= sample_size
            print(f"Added {sample_size} negative reviews, remaining: {remaining}")
    
    # Step 3: Add reviews from top movies (ALL reviews, no pattern filtering)
    print("Step 3: Adding reviews from top movies (all reviews)")
    for movie_id in top_movies:
        if remaining <= 0:
            break
        if movie_id in used_movies:
            continue
            
        movie_reviews = reviews_df[reviews_df['id'] == movie_id]
        if len(movie_reviews) == 0:
            continue
            
        # Sample 5-15 reviews per movie for diversity
        sample_size = min(len(movie_reviews), max(5, remaining // 30), remaining)
        sampled = movie_reviews.sample(n=sample_size, random_state=42)
        selected_reviews.append(sampled)
        used_movies.add(movie_id)
        remaining -= sample_size
    
    print(f"Final step: added reviews from {len(used_movies)} total movies")
    
    # Combine and shuffle
    final_reviews = pd.concat(selected_reviews, ignore_index=True)
    final_reviews = final_reviews.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Trim to exact scale factor
    final_reviews = final_reviews.head(scale_factor)
    
    return final_reviews


def generate_movies_table(movies_df, reviews_df):
    """Generate movies table for movies that have reviews."""
    movie_ids_with_reviews = reviews_df['id'].unique()
    selected_movies = movies_df[movies_df['id'].isin(movie_ids_with_reviews)].copy()
    return selected_movies

def print_statistics(movies_df, reviews_df, pattern_movie, pattern_pattern, negative_movie):
    """Print comprehensive statistics to verify requirements."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"Total movies: {len(movies_df)}")
    print(f"Total reviews: {len(reviews_df)}")
    
    # Top-5 movies with most reviews in dataset
    movie_review_counts = reviews_df.groupby('id').size().sort_values(ascending=False)
    print(f"\nTop-5 movies with most reviews in dataset:")
    for i, (movie_id, count) in enumerate(movie_review_counts.head(5).items()):
        print(f"  {i+1}. {movie_id}: {count} reviews")
    
    # Pattern movie statistics
    pattern_movie_reviews = reviews_df[reviews_df['id'] == pattern_movie]
    pattern_movie_pattern_reviews = pattern_movie_reviews[
        pattern_movie_reviews['originalScore'].str.contains(pattern_pattern, na=False)
    ]
    pattern_scores = pattern_movie_pattern_reviews['originalScore'].value_counts()
    
    print(f"\nPattern movie ({pattern_movie}):")
    print(f"  Total reviews in dataset: {len(pattern_movie_reviews)}")
    print(f"  Reviews with {pattern_pattern} pattern: {len(pattern_movie_pattern_reviews)}")
    if len(pattern_movie_pattern_reviews) > 0:
        print(f"  Unique {pattern_pattern} scores: {pattern_movie_pattern_reviews['originalScore'].nunique()}")
        print(f"  Score distribution: {dict(pattern_scores.head())}")
    
    # Negative movie statistics
    negative_movie_reviews = reviews_df[reviews_df['id'] == negative_movie]
    if len(negative_movie_reviews) > 0:
        negative_ratio = (negative_movie_reviews['scoreSentiment'] == 'NEGATIVE').mean()
        
        print(f"\nNegative movie ({negative_movie}):")
        print(f"  Total reviews in dataset: {len(negative_movie_reviews)}")
        print(f"  Negative reviews: {(negative_movie_reviews['scoreSentiment'] == 'NEGATIVE').sum()}")
        print(f"  Negative ratio: {negative_ratio:.1%}")
    
    # Overall dataset statistics
    print(f"\nOverall dataset:")
    print(f"  Null reviewText: {reviews_df['reviewText'].isnull().sum()} (should be 0)")
    print(f"  Average review length: {reviews_df['reviewText'].str.len().mean():.1f} characters")
    
    # Score pattern distribution
    for pattern in ['/5', '/10', '/4']:
        count = reviews_df['originalScore'].str.contains(pattern, na=False).sum()
        pct = count / len(reviews_df) * 100
        print(f"  Reviews with {pattern} pattern: {count} ({pct:.1f}%)")
    
    # Sentiment distribution
    sentiment_counts = reviews_df['scoreSentiment'].value_counts()
    print(f"  Sentiment distribution: {dict(sentiment_counts)}")

#Roughly coppied from Sembench generate_data.py from Movies Scenario
def generate_data():
    scale_factor = 2000
    if scale_factor is None:
        parser.error("scale_factor is required (either positional or --scale-factor)")

    # Load data
    movies_df, reviews_df = load_data('/localdisk/shin,jason-HonorsThesis/datasets/movie')
    
    # Validate scale_factor bounds
    max_reviews = len(reviews_df)  # Already filtered for nulls in load_data
    if scale_factor > max_reviews:
        print(f"Warning: scale_factor ({scale_factor}) exceeds available reviews ({max_reviews})")
        print(f"Using maximum available reviews: {max_reviews}")
        scale_factor = max_reviews
    
    # Find special movies
    pattern_movie, pattern_pattern = find_pattern_movie(reviews_df)
    negative_movie = get_negative_movie()
    
    # Get top movies by review count (much faster than ranking all movies)
    top_movies = get_top_movies_fast(reviews_df)
    
    # Sample reviews
    selected_reviews = sample_reviews(
        reviews_df, pattern_movie, pattern_pattern, negative_movie,
        top_movies, scale_factor
    )
    
    # Generate movies table
    selected_movies = generate_movies_table(movies_df, selected_reviews)
    
    # Print statistics
    print_statistics(selected_movies, selected_reviews, pattern_movie, pattern_pattern, negative_movie)
    
    # Save files to data/sf_{scale_factor}/ directory
    base_output_dir = Path(__file__).resolve().parents[4] / "files" / "movie" / "data"
    output_dir = base_output_dir / f"sf_{scale_factor}"
    output_dir.mkdir(parents=True, exist_ok=True)

    movies_file = output_dir / "Movies.csv"
    reviews_file = output_dir / "Reviews.csv"

    # Clean reviewText to prevent CSV formatting issues
    selected_reviews = selected_reviews.copy()
    selected_reviews['reviewText'] = selected_reviews['reviewText'].str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False)

    selected_movies.to_csv(movies_file, index=False)
    selected_reviews.to_csv(reviews_file, index=False)

    print(f"\nFiles saved:")
    print(f"  {movies_file}")
    print(f"  {reviews_file}")

    print(f"\n=== Generated Tables Summary ===")
    print(f"Movies.csv: {len(selected_movies)} rows")
    print(f"Reviews.csv: {len(selected_reviews)} rows")
    print(f"Maximum table size: {max(len(selected_movies), len(selected_reviews))} rows")

def run_q9_rz_sort(w, label):
    reviews = pd.read_csv('/localdisk/shin,jason-HonorsThesis/files/movie/data/sf_2000/Reviews.csv')
    filtered_reviews = reviews[
            reviews["id"] == "ant_man_and_the_wasp_quantumania"
        ]
    if len(filtered_reviews) == 0:
        print(
            "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
        )
        return pd.DataFrame(columns=["reviewId", "reviewScore"])
    ids = list(filtered_reviews['reviewId'])
    text = list(filtered_reviews['reviewText'])
    formatted_documents = []
    for i in range(len(ids)):
        formatted_documents.append(f"{ids[i]}:{text[i]}:{i}")
    test = MultiPivot_sort(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4, window_size = w)
    start_time = time.perf_counter()
    result = test.sem_sort(query="Which review shows the most positive sentiment about the movie?", documents=formatted_documents, pivots=2, group_size=2, pivot_selection_method= 'embedding', embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model")
    end_time = time.perf_counter()
    test.stop_models()
    results = []
    for r in range(len(result)):
        total_reviews = len(result)
        if total_reviews == 1:
            assigned_score = 3.0
        else:
            score_range = 4.0
            normalized_position = r / (
                total_reviews - 1
            ) 
            assigned_score = 5.0 - (normalized_position * score_range)
            assigned_score = round(
                assigned_score, 1
            )
        current_id = result[r].split(":", 1)[0]
        results.append(
            {"reviewId": float(current_id), "reviewScore": assigned_score}
        )
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{label}.csv", index=False)
    return result_df, end_time-start_time

#Coppied from evaluate.py in movies scenario
def gen_ground_truth():
    sql_path = '/localdisk/shin,jason-HonorsThesis/datasets/movie/Q9.sql'
    
    # Read the SQL query
    with open(sql_path, 'r') as f:
        sql_query = f.read().strip()
    
    # Create DuckDB connection and load data
    conn = duckdb.connect()
    
    # Register DataFrames as tables
    conn.register('Movies', pd.read_csv('/localdisk/shin,jason-HonorsThesis/files/movie/data/sf_2000/Movies.csv'))
    conn.register('Reviews', pd.read_csv('/localdisk/shin,jason-HonorsThesis/files/movie/data/sf_2000/Reviews.csv'))
    
    # Execute the query and get results
    result = conn.execute(sql_query).fetchdf()
    
    # Save ground truth
    result.to_csv("Q9.csv", index=False)
    
    conn.close()
    return result

def evaluate_results(gt, result, time, label):
    from scipy.stats import spearmanr, kendalltau
    if len(gt) == 0 or len(result) == 0:
        met = pd.DataFrame([[0.0,0.0,time]], columns = ['spearman', 'kendalltau', 'time'])
        met.to_csv(f"{label}.csv", index=False)
        return
    sys_id_col = result.columns[0]
    sys_score_col = result.columns[1]
    gt_id_col = gt.columns[0]
    gt_score_col = gt.columns[1]
    sys_scores = {}
    for _, row in result.iterrows():
        id_val = row[sys_id_col]
        score_val = row[sys_score_col]
        if pd.notna(id_val) and pd.notna(score_val):
            try:
                sys_scores[id_val] = float(score_val)
            except (ValueError, TypeError):
                continue

    gt_scores = {}
    for _, row in gt.iterrows():
        id_val = row[gt_id_col]
        score_val = row[gt_score_col]
        if pd.notna(id_val) and pd.notna(score_val):
            try:
                gt_scores[id_val] = float(score_val)
            except (ValueError, TypeError):
                continue
    common_ids = set(sys_scores.keys()) & set(gt_scores.keys())
    if len(common_ids) < 2:
        met = pd.DataFrame([[0.0,0.0,time]], columns = ['spearman', 'kendalltau', 'time'])
        met.to_csv(f"{label}.csv", index=False)
        return
    sys_values = [sys_scores[id_val] for id_val in common_ids]
    gt_values = [gt_scores[id_val] for id_val in common_ids]

    # Calculate correlations
    spearman_corr = 0.0
    kendall_corr = 0.0

    try:
        spearman_result = spearmanr(sys_values, gt_values)
        spearman_corr = (
            spearman_result.correlation
            if not pd.isna(spearman_result.correlation)
            else 0.0
        )
    except Exception:
        spearman_corr = 0.0

    try:
        kendall_result = kendalltau(sys_values, gt_values)
        kendall_corr = (
            kendall_result.correlation
            if not pd.isna(kendall_result.correlation)
            else 0.0
        )
    except Exception:
        kendall_corr = 0.0
    
    met = pd.DataFrame([[spearman_corr,kendall_corr,time]], columns = ['spearman', 'kendalltau', 'time'])
    met.to_csv(f"{label}.csv", index=False)
    return

def run_q9_rz_tour(w, label):
    reviews = pd.read_csv('/localdisk/shin,jason-HonorsThesis/files/movie/data/sf_2000/Reviews.csv')
    filtered_reviews = reviews[
            reviews["id"] == "ant_man_and_the_wasp_quantumania"
        ]
    if len(filtered_reviews) == 0:
        print(
            "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
        )
        return pd.DataFrame(columns=["reviewId", "reviewScore"])
    ids = list(filtered_reviews['reviewId'])
    text = list(filtered_reviews['reviewText'])
    formatted_documents = []
    for i in range(len(ids)):
        formatted_documents.append(f"{ids[i]}:{text[i]}:{i}")
    test = MultiPivot_tour(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4, window_size = w)
    start_time = time.perf_counter()
    result = test.tournament_top_k(query="Which review shows the most positive sentiment about the movie?", documents=formatted_documents, k = len(formatted_documents))
    end_time = time.perf_counter()
    test.stop_models()
    results = []
    for r in range(len(result)):
        total_reviews = len(result)
        if total_reviews == 1:
            assigned_score = 3.0
        else:
            score_range = 4.0
            normalized_position = r / (
                total_reviews - 1
            ) 
            assigned_score = 5.0 - (normalized_position * score_range)
            assigned_score = round(
                assigned_score, 1
            )
        current_id = result[r].split(":", 1)[0]
        results.append(
            {"reviewId": float(current_id), "reviewScore": assigned_score}
        )
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{label}.csv", index=False)
    return result_df, end_time-start_time

#This is most coppied from sembench Lotus Movies runner
def run_q9_map(name, label):
    import lotus
    from lotus.models import LM
    lm = LM(model=f'hosted_vllm/{name}',
        api_base='http://localhost:8000/v1',
        max_ctx_len=4096,
        max_tokens=1000)
    lotus.settings.configure(lm=lm)

    reviews = pd.read_csv('/localdisk/shin,jason-HonorsThesis/files/movie/data/sf_2000/Reviews.csv')
    filtered_reviews = reviews[
            reviews["id"] == "ant_man_and_the_wasp_quantumania"
        ]
    if len(filtered_reviews) == 0:
        print(
            "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
        )
        return pd.DataFrame(columns=["reviewId", "reviewScore"])
    scoring_prompt = """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments."""

    # Use sem_map for scoring - returns scores directly
    start_time = time.perf_counter()
    scored_reviews = filtered_reviews.sem_map(scoring_prompt)
    end_time = time.perf_counter()
    # Extract scores from the _map column
    results = []
    for _, row in scored_reviews.iterrows():
        score = row["_map"]
        # Ensure score is numeric and within range 1-5
        try:
            numeric_score = float(score)
            if 1 <= numeric_score <= 5:
                results.append(
                    {
                        "reviewId": row["reviewId"],
                        "reviewScore": numeric_score,
                    }
                )
            else:
                # Default to 3 if score is out of range
                results.append(
                    {"reviewId": row["reviewId"], "reviewScore": 3.0}
                )
        except (ValueError, TypeError):
            # Default to 3 if score is not numeric
            results.append(
                {"reviewId": row["reviewId"], "reviewScore": 3.0}
            )
    result_df = pd.DataFrame(results)
    result_df.to_csv(f"{label}.csv", index=False)
    return result_df, end_time-start_time

def run_q9_top10_rerank(csv_path_list, result_csv_path_list):
    reviews = pd.read_csv('/localdisk/shin,jason-HonorsThesis/files/movie/data/sf_2000/Reviews.csv')
    filtered_reviews = reviews[
            reviews["id"] == "ant_man_and_the_wasp_quantumania"
        ]
    if len(filtered_reviews) == 0:
        print(
            "  Warning: No reviews found for movie 'ant_man_and_the_wasp_quantumania'"
        )
        return pd.DataFrame(columns=["reviewId", "reviewScore"])
    test = MultiPivot_sort(model_path="/localdisk/shin,jason-HonorsThesis/rankzephyr", n_devices=4, window_size = 2)
    ids = list(filtered_reviews['reviewId'])
    text = list(filtered_reviews['reviewText'])
    label_id = 1
    result_path = 0
    for csv in csv_path_list:
        data = pd.read_csv(csv)
        data_ids = list(data['reviewId'])
        top_10 = data_ids[:10]
        formatted_reviews = []
        i = 0
        for t in top_10:
            index = 0
            while t != float(ids[index]):
                index = index + 1
            formatted_reviews.append(f"{ids[index]}:{text[index]}:{i}")
            i = i + 1
        start_time = time.perf_counter()
        result = test.sem_sort(query="Which review shows the most positive sentiment about the movie?", documents=formatted_reviews, pivots=1, group_size=1, pivot_selection_method= 'embedding', embedding_path= "/localdisk/shin,jason-HonorsThesis/st_model")
        end_time = time.perf_counter()
        formatted_other = []
        for i in data_ids:
            formatted_other.append(f"{i}:")
        result = result + formatted_other[len(result):]
        results = []
        for r in range(len(result)):
            total_reviews = len(result)
            if total_reviews == 1:
                assigned_score = 3.0
            else:
                score_range = 4.0
                normalized_position = r / (
                    total_reviews - 1
                ) 
                assigned_score = 5.0 - (normalized_position * score_range)
                assigned_score = round(
                    assigned_score, 1
                )
            current_id = result[r].split(":", 1)[0]
            results.append(
                {"reviewId": float(current_id), "reviewScore": assigned_score}
            )
        result_df = pd.DataFrame(results)
        result_df.to_csv(f"{label_id}_recomputed.csv", index=False)
        times = pd.read_csv(result_csv_path_list[result_path])
        gt = gen_ground_truth()
        metric = list(times['time'])[0] + (end_time-start_time)
        evaluate_results(gt, result_df, metric, f"{label_id}_score_recomputed")
        label_id = label_id + 1
        result_path = result_path + 1
    test.stop_models()
    return

def run_recompute(path, val, tour):
    csv_path = []
    result_path = []
    i = 1
    while i <= 10:
        if tour:
            temp = path + f"rzt20_{i}.csv"
            csv_path.append(temp)
            temp = path + f"q9t_20_{i}.csv"
            result_path.append(temp)
        else:
            temp = path + f"rz{val}_{i}.csv"
            csv_path.append(temp)
            temp = path + f"q9_{val}_{i}.csv"
            result_path.append(temp)
        i = i + 1
    run_q9_top10_rerank(csv_path, result_path)
    return

#/localdisk/shin,jason-HonorsThesis/zephyr7b
import sys
#generate_data()
#df, metric = run_q9_rz_sort(int(sys.argv[1]), sys.argv[2])
#df, metric = run_q9_rz_tour(int(sys.argv[1]), sys.argv[2])
#df, metric = run_q9_map("/localdisk/shin,jason-HonorsThesis/zephyr7b", sys.argv[1])
#gt = gen_ground_truth()
#evaluate_results(gt, df, metric, sys.argv[3])
#evaluate_results(gt, df, metric, sys.argv[2])
run_recompute('/localdisk/shin,jason-HonorsThesis/solicedb/src/solicedb/llm/sembench_data/tour/', 20, True)