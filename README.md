## silicon_demo

Brief demo for running multiple LLMs on the fearspeech task and producing merged outputs and basic analyses.

### What this does
- Runs several models (OpenAI GPT, Claude, Gemini; optional DeepSeek) on a small sample dataset using `scr/run_llm_func.py`.
- Writes per-model iteration CSVs to `outputs/fearspeech/` and interim rows to `wip/`.
- Merges iterations into `outputs/Combined_Files/fearspeech.csv`.
- Optionally computes agreement (Kappa) and generates a threshold plot and a regression-ready CSV.

### Key files
- `run_llm.py`: End-to-end driver that loads inputs, calls `run_llm(...)`, merges outputs, runs agreement and plotting.
- `scr/run_llm_func.py`: Core helpers for model calls and output handling (GPT/DeepSeek/Gemini/Claude, merging utilities).
- `inputs/`: Sample fearspeech CSV and prompt JSON.

### Install
```bash
pip install -r requirements.txt
```

### Configure API keys (set any you plan to use)
```bash
export OPENAI_API_KEY=...      # OpenAI (GPT/o3)
export DEEPSEEK_API_KEY=...    # DeepSeek via OpenAI client (optional)
export GEMINI_API_KEY=...      # Google Gemini
export ANTHROPIC_API_KEY=...   # Claude
```

### Optional environment variables
- `FEARSPEECH_DATA_PATH`: path to input CSV (defaults to `inputs/pnas_2023_fearspeech_sample.csv`).
- `BASE_PATH`: where outputs should be written (defaults to this `silicon_demo` folder).
- `OUTPUT_DIR`: combined output subfolder (default: `outputs/Combined_Files`).

### Run
From inside the `silicon_demo` folder:
```bash
python run_llm.py
```

### Outputs (by default)
- Per-iteration model files: `outputs/fearspeech/iteration_<model>.csv` and `..._repeatN.csv`.
- Combined file: `outputs/Combined_Files/fearspeech.csv`.
- Agreement plot(s): `outputs/kappa_vs_threshold_fearspeech*.png`.
- Regression input: `outputs/Regression/fearspeech_regression.csv`.

Repository: https://github.com/XCharlieCHENG/silicon


