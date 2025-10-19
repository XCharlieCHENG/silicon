# SILICON Workflow for LLM Annotation

Code for running LLM annotation tasks and conducting analyses. 

Reference: Xiang Cheng, Raveesh Mayya, and João Sedoc: “To Err Is Human; To Annotate, SILICON? Reducing Measurement Error in LLM Annotation.” arXiv preprint: [link](https://arxiv.org/abs/2412.14461).

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

### Run
From inside the `silicon_demo` folder:
```bash
python run_llm.py
```

Alternatively, you can run the script using `run_llm.ipynb`.