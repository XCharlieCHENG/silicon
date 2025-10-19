import pandas as pd
from typing import Dict, List, Union


def calculate_fsd(
    df: pd.DataFrame,
    label_list: List[str],
    *,
    return_row_scores: bool = False,
    row_score_col: str = 'FSD_row',
) -> Union[float, pd.DataFrame]:
    """
    Calculate the FSD score (sampling-based, exact mirror of the original API).

    Backward-compatible default returns a single float (mean FSD across rows).
    If return_row_scores=True, returns a DataFrame with an extra column containing
    each row's FSD value.

    Args:
        df: Data containing sampled outputs per row.
        label_list: Column names corresponding to sampled outputs for each row.
        return_row_scores: If True, return df with per-row FSD in `row_score_col`.
        row_score_col: Name of the per-row FSD column when returning a DataFrame.

    Returns:
        float if return_row_scores=False, otherwise a DataFrame with per-row FSD.
    """
    fsd_scores: List[float] = []

    for _, row in df.iterrows():
        answers = [str(row[col]) for col in label_list if col in df.columns and pd.notnull(row.get(col))]

        if not answers:
            fsd_value = 0.0
        else:
            frequency: Dict[str, int] = {}
            for answer in answers:
                frequency[answer] = frequency.get(answer, 0) + 1

            if "[invalid]" in frequency and len(frequency) > 1:
                del frequency["[invalid]"]

            counts = sorted(frequency.values(), reverse=True)
            if len(counts) > 1:
                top_gap = counts[0] - counts[1]
            else:
                top_gap = len(answers)

            fsd_value = top_gap / max(len(answers), 1)

        fsd_scores.append(fsd_value)

    if return_row_scores:
        out_df = df.copy()
        out_df[row_score_col] = fsd_scores
        return out_df

    mean_fsd = float(sum(fsd_scores) / len(fsd_scores)) if fsd_scores else 0.0
    return mean_fsd



