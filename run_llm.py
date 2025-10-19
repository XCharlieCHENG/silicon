import os
import json
import subprocess
import pandas as pd

# Resolve paths relative to this demo folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# Constrain all I/O within silicon_demo
BASE_PATH = os.getenv('BASE_PATH', SCRIPT_DIR)
OUTPUT_DIR = os.getenv('OUTPUT_DIR', 'outputs/Combined_Files')

#
# SECTION 0: Paths and inputs
#

DATA_PATH = os.getenv('FEARSPEECH_DATA_PATH', os.path.join(SCRIPT_DIR, 'inputs', 'pnas_2023_fearspeech_sample.csv'))
df_task = pd.read_csv(DATA_PATH).iloc[:10,:]
pattern_label = r'"id":\s*"([^"]+)"\s*,\s*"label":\s*"([^"]+)"'
with open(os.path.join(SCRIPT_DIR, 'inputs', 'prompt_fearspeech.json'), 'r', encoding='utf-8') as json_file:
    task_prompts = json.load(json_file)


#
# SECTION 1: LLM inference for fearspeech
#

from openai import OpenAI
from google import genai
import anthropic
from scr.run_llm_func import run_llm

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

client_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
client_deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com") if DEEPSEEK_API_KEY else None
client_gemini = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
client_claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

if not any([client_openai, client_deepseek, client_gemini, client_claude]):
        print("No API clients configured. Skipping LLM inference (set OPENAI_API_KEY or API_KEYS_JSON_PATH).")


# Run LLMs for Fearspeech task
if client_openai is not None:
        _ = run_llm(client_openai, df_task, model='gpt-4o',
                instruction_prompt=task_prompts['instruction_prompt'],
                user_prompt_beginning=task_prompts['user_prompt_beginning'],
                base_path=BASE_PATH,
                pattern=pattern_label, model_series='gpt')
        _ = run_llm(client_openai, df_task, model='gpt-4.1-2025-04-14',
                instruction_prompt=task_prompts['instruction_prompt'],
                user_prompt_beginning=task_prompts['user_prompt_beginning'],
                base_path=BASE_PATH,
                pattern=pattern_label, model_series='gpt')
        _ = run_llm(client_openai, df_task, model='o3-mini',
                instruction_prompt=task_prompts['instruction_prompt'],
                user_prompt_beginning=task_prompts['user_prompt_beginning'],
                base_path=BASE_PATH,
                pattern=pattern_label, model_series='gpt')
if client_claude is not None:
        _ = run_llm(client_claude, df_task, model='claude-3-7-sonnet-20250219',
                instruction_prompt=task_prompts['instruction_prompt'],
                user_prompt_beginning=task_prompts['user_prompt_beginning'],
                base_path=BASE_PATH,
                pattern=pattern_label, model_series='claude')
if client_gemini is not None:
        _ = run_llm(client_gemini, df_task, model='gemini-2.5-pro-preview-03-25',
                instruction_prompt=task_prompts['instruction_prompt'],
                user_prompt_beginning=task_prompts['user_prompt_beginning'],
                base_path=BASE_PATH,
                pattern=pattern_label, model_series='gemini')


#
# SECTION 2: Merge multiple iteration outputs for fearspeech
#
from scr.run_llm_func import merge_multiple_runs
df = merge_multiple_runs(
        base_path=BASE_PATH,
        relative_path="outputs/fearspeech",
        file_pattern=r"iteration_(.+?)\.csv$",
        iteration_cols=['label_llm'],
        key="fearspeech",
        output_dir=OUTPUT_DIR,
        keep_cols=['id', 'text', 'majority_label']
)
_combined_after_merge = os.path.join(BASE_PATH, 'outputs/Combined_Files', 'fearspeech.csv')
if os.path.exists(_combined_after_merge):
        try:
            _dfc = pd.read_csv(_combined_after_merge)
            print(f"Combined CSV ready: {_combined_after_merge} ({len(_dfc)} rows)")
        except Exception as e:
            print(f"Note: combined CSV exists but could not be read: {e}")
else:
        print("No combined CSV yet (run inference or place combined file under outputs/Combined_Files).")


#
# SECTION 3: Agreement analysis (Kappa) for fearspeech
#
from scr.agreement_func import output_llm_gt_kappas
from scr.datasets_config import fearspeech_dict
combined_path = os.path.join(BASE_PATH, 'outputs/Combined_Files', 'fearspeech.csv')

is_multi_label_task=True

if os.path.exists(combined_path):
        df_task_combined = pd.read_csv(combined_path)
        has_gt = ('majority_label' in df_task_combined.columns) and df_task_combined['majority_label'].notna().any()
        if len(df_task_combined) > 0 and has_gt:
            # Use demo-generated columns only (label_llm*)
            demo_cols = [c for c in df_task_combined.columns if c.startswith('label_llm') and c != 'label_llm']
            if demo_cols:
                def pretty(name: str) -> str:
                    suff = name[len('label_llm'):].lstrip('_')
                    mapping = {
                        'gpt_4o': 'GPT-4o',
                        'o3_mini': 'o3-mini',
                    }
                    return mapping.get(suff, suff.replace('_', '-'))
                demo_labels = [pretty(c) for c in demo_cols]
                result_task = output_llm_gt_kappas(
                        df_task_combined,
                        gt_col='majority_label',
                        llm_col_list=demo_cols,
                        llm_col_labels=demo_labels,
                        weighted=is_multi_label_task,
                )
                print(f"Kappa analysis results:\n {result_task}")
            else:
                print("Warning: No demo LLM columns (label_llm*) found. Skipping kappa analysis.")
        elif len(df_task_combined) > 0 and not has_gt:
            print("Warning: Ground truth 'majority_label' not found or empty. Skipping kappa analysis.")
        else:
            print("Warning: Combined fearspeech CSV is empty. Skipping kappa analysis.")
else:
        print("Warning: Combined fearspeech CSV not found in silicon_demo/outputs/Combined_Files. Skipping kappa analysis.")


#
# SECTION 4: Regression CSV for fearspeech (model equivalence test input)
#
from scr.regression_utils import create_regression_csv
source_path = os.path.join(BASE_PATH, 'outputs/Combined_Files', 'fearspeech.csv')
output_path = os.path.join(BASE_PATH, 'outputs/Regression', 'fearspeech_regression.csv')
if os.path.exists(source_path):
        try:
            df_src = pd.read_csv(source_path)
            if ('majority_label' in df_src.columns) and df_src['majority_label'].notna().any():
                create_regression_csv(source_path, 'majority_label', fearspeech_dict, output_path)
                # Run regression analysis (binary accuracy) for the generated fearspeech CSV
                try:
                    script_sh = os.path.join(SCRIPT_DIR, 'scr', 'regression_analysis', 'run_model_comparison.sh')
                    # Execute via bash to avoid relying on executable bit; run from demo root for relative paths
                    subprocess.run(['bash', script_sh], cwd=SCRIPT_DIR, check=True)
                except Exception as e:
                    print(f"Warning: Regression analysis script failed: {e}")
            else:
                print("Warning: No ground truth in combined CSV. Skipping regression CSV generation.")
        except Exception as e:
            print(f"Warning: Failed to create regression CSV: {e}")
else:
        print("Warning: Combined fearspeech CSV not found. Skipping regression CSV generation.")

#
# SECTION 5: Multi-LLM Labeling Threshold Plot (single-task: fearspeech)
#
from scr.threshold_plot import plot_kappa_vs_threshold_fearspeech_with_fsd
from scr.fsd_sampling import calculate_fsd
plot_save = os.path.join(SCRIPT_DIR, 'outputs', 'kappa_vs_threshold_fearspeech.png')
plot_save_fsd = os.path.join(SCRIPT_DIR, 'outputs', 'kappa_vs_threshold_fearspeech_fsd.png')
combined_exists = os.path.exists(os.path.join(BASE_PATH, 'outputs/Combined_Files', 'fearspeech.csv'))
if combined_exists:
        try:
            df_plot_src = pd.read_csv(os.path.join(BASE_PATH, 'outputs/Combined_Files', 'fearspeech.csv'))
            has_gt_plot = ('majority_label' in df_plot_src.columns) and df_plot_src['majority_label'].notna().any()
            # Require demo label_llm* columns
            has_demo_cols = any(c.startswith('label_llm') and c != 'label_llm' for c in df_plot_src.columns)
            if has_gt_plot and has_demo_cols:
                # Step A: Generate FSD dataset for 'gpt-4o' models (sampling-based) into silicon_demo/outputs/fsd_per_task
                fsd_out_dir = os.path.join(SCRIPT_DIR, 'outputs', 'fsd_per_task')
                os.makedirs(fsd_out_dir, exist_ok=True)
                fsd_out_path = os.path.join(fsd_out_dir, 'fearspeech_fsd.csv')

                # Identify 'gpt-4o*' sampled columns from combined CSV
                def _normalize_token(s: str) -> str:
                    return ''.join(ch for ch in s.lower() if ch.isalnum())

                def _match_model_cols(df_any: pd.DataFrame, family_aliases: list) -> list:
                    norm_aliases = {_normalize_token(a) for a in family_aliases}
                    cols = []
                    for c in df_any.columns:
                        if not (c.startswith('label_llm') and c != 'label_llm'):
                            continue
                        suffix = c[len('label_llm'):].lstrip('_')
                        norm_suffix = _normalize_token(suffix)
                        if any(a in norm_suffix for a in norm_aliases):
                            cols.append(c)
                    return cols

                main_cols = _match_model_cols(df_plot_src, ['gpt-4o', 'gpt_4o', 'gpt4o'])
                if len(main_cols) < 2:
                    raise ValueError("Need at least two 'gpt-4o' sampled columns in Combined_Files to compute FSD.")

                df_with_scores = calculate_fsd(df_plot_src, main_cols, return_row_scores=True, row_score_col='fsd')
                fsd_df = pd.DataFrame({'fsd': df_with_scores['fsd']})
                fsd_df.to_csv(fsd_out_path, index=False)
                print(f"Saved sampling-based FSD series to {fsd_out_path}")

                # Step B: Run FSD-gated threshold plot (no fallback)
                summary_fsd = plot_kappa_vs_threshold_fearspeech_with_fsd(
                        base_path=BASE_PATH,
                        save_path=plot_save_fsd,
                        main_aliases=['gpt_4o'],
                        families_for_mv=['gpt_4_1_2025_04_14', 'o3_mini'],
                        weighted_kappa=is_multi_label_task,
                )
                print(f"FSD-gated threshold plot saved: {plot_save_fsd}")
                if summary_fsd is not None and not summary_fsd.empty:
                    print("Kappa vs FSD threshold:")
                    print(summary_fsd.to_string(index=False))
            else:
                print("Warning: Insufficient data (ground truth or model columns) for threshold plot. Skipping.")
        except Exception as e:
            print(f"Warning: Failed to generate threshold plot: {e}")
else:
        print("Warning: Combined fearspeech CSV not found. Skipping threshold plot.")
