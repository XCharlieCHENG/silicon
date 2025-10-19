import sys

import openai
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import time
import os
import csv

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import ipdb
import krippendorff
from datetime import datetime
from itertools import combinations
import scipy.stats as st
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import confusion_matrix
import inspect
from scipy.stats import norm
import json
import re
# from google import genai  # Not used in current functions
import pandas as pd

def cohen_kappa_score_weighted(y1, y2,
                               labels=['fearspeech', 'hatespeech', 'normal', 'confused', 'hatespeech,fearspeech'],
                               disagree_weight={('fearspeech', 'hatespeech,fearspeech'): 0.5, ('hatespeech', 'hatespeech,fearspeech'): 0.5}):
    """
    Computes Cohen's kappa with a custom disagreement weight matrix.

    Parameters:
    y1, y2 : array-like of shape (n_samples,)
        Labels assigned by the first and second annotators respectively.
    labels : list of str
        The labels to index the matrix. This must include all labels used by the annotators.
    disagree_weight : dict
        A dictionary with key as tuple of labels and value as the custom weight for these labels.

    Returns:
    kappa : float
        The calculated Cohen's kappa statistic.
    """
    # Create confusion matrix
    confusion = confusion_matrix(y1, y2, labels=labels)
    n_classes = confusion.shape[0]

    # Create the weight matrix
    w_mat = np.ones([n_classes, n_classes], dtype=float)
    w_mat.flat[::n_classes + 1] = 0  # Set diagonal to 0 (perfect agreement has no weight)

    # Apply custom weights for specified disagreements
    for (label1, label2), weight in disagree_weight.items():
        index1 = labels.index(label1)
        index2 = labels.index(label2)
        w_mat[index1, index2] = weight
        w_mat[index2, index1] = weight  # Symmetry in disagreement weight

    # Calculate Cohen's Kappa
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    kappa = 1-k
    # ipdb.set_trace()
    return kappa

def calculate_kappa_and_se_bootstrap(df, col1, col2, confidence=0.90, num_bootstrap_samples=1):
    """
    Calculate Cohen's Kappa and its confidence interval using bootstrap resampling.

    Parameters:
    - df: DataFrame containing the ratings.
    - col1, col2: Columns in df containing the two sets of ratings to compare.
    - confidence: Confidence level for the interval (default 0.90).
    - num_bootstrap_samples: Number of bootstrap samples to generate (default 1000).

    Returns:
    - kappa: Cohen's Kappa score.
    - se: Standard error of the Kappa score.
    """
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(df[col1], df[col2])

    # Bootstrap to get S.E.
    kappas = []
    for _ in range(num_bootstrap_samples):
        bootstrap_sample = df.sample(frac=1, replace=True)
        kappa_bootstrap = cohen_kappa_score(bootstrap_sample[col1], bootstrap_sample[col2])
        kappas.append(kappa_bootstrap)

    # Calculate standard error (S.E.)
    se = np.std(kappas)

    return kappa, se

def calculate_kappa_and_se_weighted_bootstrap(df, col1, col2, label_list, weight_dict, confidence=0.90, num_bootstrap_samples=1):
    """
    Calculate Cohen's Kappa and its confidence interval using bootstrap resampling with custom weights.

    Parameters:
    - df: DataFrame containing the ratings.
    - col1, col2: Columns in df containing the two sets of ratings to compare.
    - label_list: List of unique labels.
    - weight_dict: Dictionary with custom weights for label disagreements.
    - confidence: Confidence level for the interval (default 0.90).
    - num_bootstrap_samples: Number of bootstrap samples to generate (default 1000).

    Returns:
    - kappa: Cohen's Kappa score.
    - se: Standard error of the Kappa score.
    """
    # Calculate Cohen's Kappa with custom weights
    kappa = cohen_kappa_score_weighted(df[col1], df[col2], labels=label_list, disagree_weight=weight_dict)

    # Bootstrap to get S.E.
    kappas = []
    for _ in range(num_bootstrap_samples):
        bootstrap_sample = df.sample(frac=1, replace=True)
        kappa_bootstrap = cohen_kappa_score_weighted(bootstrap_sample[col1], bootstrap_sample[col2], labels=label_list, disagree_weight=weight_dict)
        kappas.append(kappa_bootstrap)

    # Calculate standard error (S.E.)
    se = np.std(kappas)

    return kappa, se

def cohen_kappa_score_weighted(y1, y2, labels, disagree_weight):
    """
    Computes Cohen's kappa with a custom disagreement weight matrix.

    Parameters:
    - y1, y2: Labels assigned by the first and second annotators respectively.
    - labels: List of unique labels.
    - disagree_weight: Dictionary with custom weights for label disagreements.

    Returns:
    - kappa: The calculated Cohen's kappa statistic.
    """
    # Create confusion matrix
    confusion = confusion_matrix(y1, y2, labels=labels)
    n_classes = confusion.shape[0]

    # Create the weight matrix
    w_mat = np.ones([n_classes, n_classes], dtype=float)
    w_mat.flat[::n_classes + 1] = 0  # Set diagonal to 0 (perfect agreement has no weight)

    # Apply custom weights for specified disagreements
    for (label1, label2), weight in disagree_weight.items():
        index1 = labels.index(label1)
        index2 = labels.index(label2)
        w_mat[index1, index2] = weight
        w_mat[index2, index1] = weight  # Symmetry in disagreement weight

    # Calculate Cohen's Kappa
    sum0 = np.sum(confusion, axis=0)
    sum1 = np.sum(confusion, axis=1)
    expected = np.outer(sum0, sum1) / np.sum(sum0)

    k = np.sum(w_mat * confusion) / np.sum(w_mat * expected)
    kappa = 1 - k

    return kappa

def get_label_list(df, col_list):
    """
    Given a DataFrame and a list of columns, return a list of unique values that appear
    at least once in any one of the specified columns.

    Parameters:
    - df: The DataFrame to process.
    - col_list: The list of column names to look for unique values.

    Returns:
    - list: A list of unique values.
    """
    unique_values = set()
    for col in col_list:
        if col in df.columns:
            unique_values.update(df[col].dropna().unique())
    return list(unique_values)

def get_weight_dict(label_list):
    """
    Given a list of labels, returns a dictionary with unique dyads of labels as keys,
    and values calculated as 1 - J*M, where J is the Jaccard similarity and M is a
    monotonicity penalization metric based on set relationships.
    See Passonneau: Measuring Agreement on Set-valued Items (MASI) for Semantic and Pragmatic Annotation

    Parameters:
    - label_list: The list of label strings, where each label can contain multiple comma-separated values.

    Returns:
    - dict: A dictionary with tuple keys representing label dyads, and their associated weight values.
    """
    def calculate_M(set1, set2):
        if set1 == set2:
            return 1
        elif set1.issubset(set2) or set2.issubset(set1):
            return 2/3
        elif set1 & set2 and set1 - set2 and set2 - set1:
            return 1/3
        else:
            return 0

    label_sets = [set(labels.split(',')) for labels in label_list]
    weight_dict = {}
    for label_string1, label_string2 in combinations(label_list, 2):
        set1 = set(label_string1.split(','))
        set2 = set(label_string2.split(','))
        intersection = set1 & set2
        union = set1 | set2
        J = len(intersection) / len(union)
        M = calculate_M(set1, set2)
        weight_dict[(label_string1, label_string2)] = 1 - J * M
    return weight_dict

def calculate_mean_kappa_same_labelers(df, columns, print_detail=False, method='Unweighted', confidence=0.9):
    """
    Computes the average Cohen's Kappa score for each pair of columns in the given list from the DataFrame.

    Parameters:
    - df: The DataFrame containing the data.
    - columns: A list of column names for which to compute the Cohen's Kappa scores.
    - print_detail: Whether to print detailed kappa scores.
    - method: The method to use ('Unweighted-bootstrap' or 'Weighted-bootstrap').
    - confidence: Confidence level for the interval.

    Returns:
    - tuple: The average Cohen's Kappa score and its confidence interval.
    """
    # Replace NaN values with the most common value in each column
    for column in columns:
        mode_values = df[column].mode()
        if len(mode_values) == 0:
            print(f"Warning: Column '{column}' has no mode. Skipping NaN replacement for this column.")
            continue
        most_common_value = mode_values.iloc[0]
        df[column] = df[column].fillna(most_common_value)

    kappa_scores = []
    kappa_ses = []
    n_columns = len(columns)

    # Prepare inputs for weighted kappa
    if method in ['Unweighted-bootstrap', "Weighted-bootstrap"]:
        label_list = get_label_list(df, columns)
        weight_dict = get_weight_dict(label_list)

    # Iterate over each unique pair of columns (i.e., annotators) to calculate Cohen's Kappa score
    for i in range(n_columns):
        for j in range(i + 1, n_columns):
            if method == 'Unweighted-bootstrap':
                score, se = calculate_kappa_and_se_bootstrap(df, columns[i], columns[j], confidence, num_bootstrap_samples=1)
                kappa_scores.append(score)
                kappa_ses.append(se)
            elif method == 'Weighted-bootstrap':
                score, se = calculate_kappa_and_se_weighted_bootstrap(df, columns[i], columns[j], label_list, weight_dict, confidence, num_bootstrap_samples=1)
                kappa_scores.append(score)
                kappa_ses.append(se)
            else:
                print("Method should be one of the following: 'Unweighted-bootstrap' or 'Weighted-bootstrap'.")

    # Calculate the average of all Cohen's Kappa scores
    average_score = np.mean(kappa_scores)
    average_se = np.mean(kappa_ses)
    if print_detail:
        formatted_kappa_scores = [f"{score:.3f}" for score in kappa_scores]
        print(f"Pairwise Cohen's Kappa: {formatted_kappa_scores}")

    # Determine the z-score for the specified confidence level
    alpha = 1 - confidence
    z = st.norm.ppf(1 - alpha / 2)

    # Calculate confidence interval (CI)
    lower = average_score - z * average_se
    upper = average_score + z * average_se

    ci = (round(lower, 3), round(upper, 3))

    return round(average_score, 3), ci



from collections import Counter

def get_majority_vote_tag(df, columns, new_col):
    majority_votes = []

    for _, row in df.iterrows():
        combined_tags = []
        valid_cols = 0

        # collect tags and count non-missing columns
        for col in columns:
            raw = row[col]
            if pd.notnull(raw):
                valid_cols += 1
                cleaned = re.sub(r'\s+', '', str(raw))
                tags = cleaned.split(',') if cleaned else []
            else:
                tags = []
            combined_tags.extend(tags)

        # dynamic threshold: half of the non-missing columns
        threshold = valid_cols / 2

        tag_counts = Counter(combined_tags)
        majority = [tag for tag, cnt in tag_counts.items() if cnt >= threshold]

        majority_votes.append(','.join(majority) if majority else np.nan)

    df[new_col] = majority_votes
    return df

def get_majority_vote_across_runs(df, mapping_dict, cols_for_majority, new_col="majority_5llms"):
    """
    Given a DataFrame, a dict mapping column-keys to human labels, and a list of those labels 
    you want to include in your majority-vote, this will:
      1. Invert the dict to find the column-keys corresponding to each label in cols_for_majority.
      2. Call get_majority_vote_tag on those columns and write the result into new_col.

    Args:
        df (pd.DataFrame): DataFrame containing your raw tag columns.
        mapping_dict (dict): e.g. {"label_llm4": "GPT-4 Turbo", ...}
        cols_for_majority (list of str): list of *values* from mapping_dict, e.g. ["GPT-4 Turbo", "GPT-4o", …]
        new_col (str): name of the output column to hold majority-vote tags.

    Returns:
        pd.DataFrame: the same df, with new_col added.
    """
    # build reverse lookup: label → column-key
    reverse_map = {v: k for k, v in mapping_dict.items()}
    
    # for each label in cols_for_majority, find its column-key
    columns = []
    for label in cols_for_majority:
        if label not in reverse_map:
            raise KeyError(f"Label '{label}' not found in mapping_dict")
        columns.append(reverse_map[label])
    
    # delegate to your existing majority-vote function
    return get_majority_vote_tag(df, columns=columns, new_col=new_col)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




def output_llm_gt_kappas(df, gt_col, llm_col_list, llm_col_labels, weighted=False):
    """
    Compare a ground truth column with multiple LLM-generated columns and output a table of kappa values.

    Args:
        df (pd.DataFrame): DataFrame containing the columns for comparison.
        gt_col (str): Ground truth column name.
        llm_col_list (list of str): List of LLM-generated column names to compare against the ground truth.
        llm_col_labels (list of str): List of labels for each LLM column.
        weighted (bool): Whether to use weighted Cohen's Kappa.

    Returns:
        pd.DataFrame: Table of kappa values and LLM labels.
    """
    if gt_col not in df.columns:
        raise ValueError(f"Ground truth column '{gt_col}' not found in DataFrame.")

    # Ensure the lengths of column list and labels match
    if len(llm_col_list) != len(llm_col_labels):
        raise ValueError("The length of llm_col_list and llm_col_labels must be the same.")

    # Clean the data by removing underscores, slashes, spaces and converting to lowercase
    for col in llm_col_list + [gt_col]:
        df[col] = df[col].apply(
            lambda x: x.replace("_", "").replace("/", "").replace(" ", "").strip().lower() if isinstance(x, str) else x
        )

    results = []

    for llm_col, label in zip(llm_col_list, llm_col_labels):
        if weighted:
            score, _ = calculate_mean_kappa_same_labelers(df, [gt_col, llm_col], print_detail=False, method='Weighted-bootstrap')
        else:
            score, _ = calculate_mean_kappa_same_labelers(df, [gt_col, llm_col], print_detail=False, method='Unweighted-bootstrap')


        results.append({'LLM': label, 'Kappa Score': score})

    # Convert results to a DataFrame and display
    result_df = pd.DataFrame(results)
    # print(result_df)
    return result_df
    
def output_llm_gt_metrics(df, gt_col, llm_col_list, llm_col_labels, average='macro'):
    """
    Compare a ground truth column with multiple LLM-generated columns and output a table of classification metrics.

    Args:
        df (pd.DataFrame): DataFrame containing the columns for comparison.
        gt_col (str): Ground truth column name.
        llm_col_list (list of str): List of LLM-generated column names to compare against the ground truth.
        llm_col_labels (list of str): List of labels for each LLM column (for display).
        average (str): Averaging method for precision, recall, and F1. 
                       Options include 'binary', 'micro', 'macro', 'weighted'. Default is 'macro'.

    Returns:
        pd.DataFrame: Table of accuracy, precision, recall, F1 for each LLM.
    """
    if gt_col not in df.columns:
        raise ValueError(f"Ground truth column '{gt_col}' not found in DataFrame.")
    if len(llm_col_list) != len(llm_col_labels):
        raise ValueError("The lengths of llm_col_list and llm_col_labels must match.")

    # Clean the data by removing underscores, slashes, spaces and converting to lowercase
    for col in llm_col_list + [gt_col]:
        df[col] = df[col].apply(
            lambda x: x.replace("_", "").replace("/", "").replace(" ", "").strip().lower() if isinstance(x, str) else x
        )

    results = []
    y_true = df[gt_col]

    for col, label in zip(llm_col_list, llm_col_labels):
        y_pred = df[col]

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        rec  = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1   = f1_score(y_true, y_pred, average=average, zero_division=0)

        results.append({
            'LLM':       label,
            'Accuracy':  round(acc, 3),
            'Precision': round(prec, 3),
            'Recall':    round(rec, 3),
            'F1':        round(f1, 3)
        })

    result_df = pd.DataFrame(results)
    # print(result_df)
    return result_df