import os
import pandas as pd


def _detect_label_columns(df: pd.DataFrame) -> list:
    """Auto-detect model label columns from the combined file.

    Use the columns produced by the current merge logic (suffixing
    "label_llm" with the iteration token derived from the filename),
    e.g., "label_llmgpt-4o", "label_llmo3-mini", "label_llmgpt-4o_repeat1".
    """
    return [c for c in df.columns if c.startswith('label_llm') and c != 'label_llm']


def create_regression_csv(source_file, gt_col,output_path):
    """Create regression CSV with binary accuracy indicators using detected model columns.

    - 1 if model prediction matches ground truth exactly, else 0.
    - Columns are auto-detected from the combined file as any column starting with
      "label_llm" (excluding the bare "label_llm").
    - The output columns are named after the detected model suffixes (e.g.,
      "gpt-4o", "o3-mini", "gpt-4o_repeat1"), plus an 'id' column.
    """
    df = pd.read_csv(source_file)

    # Auto-detect model label columns produced by the merge step
    model_cols = _detect_label_columns(df)

    # If nothing detected, no-op with only an id column
    if not model_cols:
        df_final = pd.DataFrame({'id': range(len(df))})
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        return df_final

    # Restrict to rows with ground truth available
    keep_cols = model_cols + [gt_col]
    df_filtered = df[keep_cols].copy()
    df_filtered = df_filtered.dropna(subset=[gt_col])

    # Convert to binary accuracy per model column
    for col in model_cols:
        if col in df_filtered.columns:
            df_filtered[col] = (df_filtered[col] == df_filtered[gt_col]).astype(int)

    # Use model names derived from column suffixes as headers
    model_names = [c[len('label_llm'):] for c in model_cols]
    rename_dict = {orig: new for orig, new in zip(model_cols, model_names)}
    df_filtered.rename(columns=rename_dict, inplace=True)

    df_final = df_filtered[model_names].copy()
    df_final['id'] = range(len(df_final))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"Created regression CSV: ({len(df_final)} rows).")
    return df_final


