import os
from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .agreement_func import output_llm_gt_kappas
from .fsd_sampling import calculate_fsd



# ----------------------- FSD-gated variant (single task) -----------------------


def _match_model_cols(df: pd.DataFrame, family_names: List[str]) -> List[str]:
    """Return label_llm* columns whose suffix exactly matches any given model name or its '-repeat*' variants."""
    names_set = set(family_names)
    cols: List[str] = []
    for c in df.columns:
        if not (c.startswith('label_llm') and c != 'label_llm'):
            continue
        suffix = c[len('label_llm'):].lstrip('_')
        for name in names_set:
            if suffix == name or suffix.startswith(f"{name}-repeat") or suffix.startswith(f"{name}_repeat"):
                cols.append(c)
                break
    return cols


def _threshold_dependent_majority_vote(df: pd.DataFrame, *, main_label_col: str, aux_label_cols: List[str], fsd_series: pd.Series, threshold: float, out_col: str) -> str:
    """
    If fsd >= threshold, use main model's label; else majority vote across [main_label_col] + aux_label_cols.
    """
    # Prepare a simple majority vote column from aux_label_cols
    # Normalize multi-label strings to tag sets, then pick most frequent tag set
    def _normalize_tags(val: str) -> str:
        s = str(val).strip().replace('_', '').replace('/', '').replace(' ', '').lower()
        tags = sorted([t for t in s.split(',') if t])
        return ','.join(tags) if tags else ''

    # Precompute per-row majority among vote columns (main + aux)
    vote_cols = [main_label_col] + list(aux_label_cols)

    def _majority_vote(row: pd.Series) -> str:
        counts = {}
        for col in vote_cols:
            v = row.get(col)
            if pd.isna(v):
                continue
            norm = _normalize_tags(v)
            if norm:
                counts[norm] = counts.get(norm, 0) + 1
        if not counts:
            return np.nan
        # Highest count; break ties deterministically by string
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return best

    mv_all = df.apply(_majority_vote, axis=1)

    # Build output by gating with fsd
    main_norm = df[main_label_col].apply(_normalize_tags)
    out_values = []
    for i in range(len(df)):
        fsd_val = fsd_series.iloc[i] if i < len(fsd_series) else np.nan
        if pd.notna(fsd_val) and float(fsd_val) >= float(threshold):
            out_values.append(main_norm.iloc[i])
        else:
            out_values.append(mv_all.iloc[i])
    df[out_col] = out_values
    return out_col


def _compute_fsd_series_from_samples(df: pd.DataFrame, main_cols: List[str]) -> pd.Series:
    """
    Compute rowwise FSD using multiple sampled labels from the main model family.
    """
    if len(main_cols) < 2:
        raise ValueError('Need at least two sampled columns to compute sampling-based FSD.')
    df_with_scores = calculate_fsd(df, main_cols, return_row_scores=True, row_score_col='fsd')
    return df_with_scores['fsd']


def plot_kappa_vs_threshold_fearspeech_with_fsd(
    base_path: str,
    save_path: str,
    main_aliases: List[str] = ('gpt-4o', 'gpt_4o', 'gpt4o'),
    families_for_mv: List[List[str]] = (['gpt-4.1', 'gpt_4.1', 'gpt41'], ['o3-mini', 'o3mini']),
    gt_col: str = 'majority_label',
    thresholds: List[float] = None,
    weighted_kappa: bool = True,
) -> pd.DataFrame:
    """
    Single-task FSD-gated threshold plot for fearspeech.
    - If FSD >= t: use main model's label
    - Else: majority vote across specified auxiliary families
    Returns summary DataFrame with columns: Threshold, Kappa.
    """
    src = os.path.join(base_path, 'outputs/Combined_Files/fearspeech.csv')
    df = pd.read_csv(src)

    if thresholds is None:
        thresholds = [round(x, 2) for x in np.linspace(0.0, 1.0, 11)]

    # Identify model columns
    main_cols = _match_model_cols(df, list(main_aliases))
    if not main_cols:
        raise ValueError('Main model columns not found for fearspeech.')
    main_label_col = main_cols[0]

    # Normalize families_for_mv to accept a flat list of exact names
    fam_input = families_for_mv
    try:
        if fam_input and len(fam_input) > 0 and isinstance(fam_input[0], str):
            fam_input = [[name] for name in fam_input]  # type: ignore
    except Exception:
        pass

    aux_cols: List[str] = []
    for fam_aliases in fam_input:  # type: ignore
        fam_cols = _match_model_cols(df, list(fam_aliases))
        if fam_cols:
            aux_cols.append(fam_cols[0])
    if len(aux_cols) < 1:
        raise ValueError('No auxiliary model columns found for majority vote.')

    # Compute FSD series for main model from sampled labels (sampling-based FSD)
    fsd_series = _compute_fsd_series_from_samples(df, main_cols)

    # Compute kappa over thresholds
    rows = []
    for t in thresholds:
        mv_col = f'mv_fsd_{t:.2f}'
        _threshold_dependent_majority_vote(
            df,
            main_label_col=main_label_col,
            aux_label_cols=aux_cols,
            fsd_series=fsd_series,
            threshold=float(t),
            out_col=mv_col,
        )
        tbl = output_llm_gt_kappas(
            df.copy(),
            gt_col=gt_col,
            llm_col_list=[mv_col],
            llm_col_labels=[f't={t:.2f}'],
            weighted=weighted_kappa,
        )
        kappa_val = float(tbl.loc[0, 'Kappa Score']) if not tbl.empty else np.nan
        rows.append({'Threshold': t, 'Kappa': kappa_val})

    summary = pd.DataFrame(rows)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(summary['Threshold'], summary['Kappa'], marker='o', linewidth=2)
    plt.xlabel('FSD threshold (t)')
    plt.ylabel('Weighted Kappa vs. ground truth')
    plt.title('FSD-gated Multi-LLM Threshold (Fearspeech)')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return summary

