import re
import pandas as pd
import numpy as np
import math
from collections import defaultdict
from typing import Dict, List, Union


def add_fsd_column(df: pd.DataFrame, key_token: str) -> pd.DataFrame:
    """
    Add FSD column to DataFrame by applying extract_topk and calculate_fsd to each row.

    Args:
        df: DataFrame with raw logprob data in second column
        key_token: Token to look for in logprobs (e.g., "category" or "label")

    Returns:
        Original DataFrame with additional 'fsd' column
    """
    result_df = df.copy()
    fsds = []
    for _, row in df.iterrows():
        topk_dict = extract_topk(row.iloc[1], key_token, k=5)
        if not topk_dict:
            fsds.append(0.0)
            continue
        topk_df = next(iter(topk_dict.values()))
        fsd = calculate_fsd(topk_df)
        fsds.append(fsd)
    result_df['fsd'] = fsds
    return result_df



def calculate_fsd(topk_df: pd.DataFrame) -> float:
    """Calculate FSD from DataFrame with candidate tokens and logprobs"""
    if len(topk_df) < 2:
        return 0.0
    token_probs = defaultdict(float)
    for _, row in topk_df.iterrows():
        normalized_token = row['candidate_token'].lower()
        prob = math.exp(row['candidate_logprob'])
        token_probs[normalized_token] += prob
    sorted_tokens = sorted(token_probs.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_tokens) < 2:
        return 0.0
    first_prob = sorted_tokens[0][1]
    second_prob = sorted_tokens[1][1]
    prob_diff = first_prob - second_prob
    return prob_diff


