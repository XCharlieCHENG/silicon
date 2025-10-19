"""
Centralized lists and dictionaries for dataset/model configurations.

This module extracts commonly used lists and dicts from notebooks so they can be
imported and reused without duplication.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Label-task pair definitions per model family
# Each tuple: (label_list, task_name, file_path, filter_cols)
# ---------------------------------------------------------------------------

label_task_pairs_llama33: List[Tuple[List[str], str, str, List[str]]] = [
    (['label_llm39', 'label_llm40', 'label_llm41', 'label_llm49', 'label_llm50'], "Business Proposal Evaluation", "output/Combined_Files/dao.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['label_llm7', 'label_llm8', 'label_llm9', 'label_llm10', 'label_llm11'], "Review Attribute Detection", "output/Combined_Files/review.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['tag1_llm11', 'tag1_llm17', 'tag1_llm18', 'tag1_llm19', 'tag1_llm20'], "Dialog Intent Classification", "output/Combined_Files/dialog.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm11', 'tag2_llm17', 'tag2_llm18', 'tag2_llm19', 'tag2_llm20'], "Dialog Breakdown Analysis", "output/Combined_Files/dialog.csv", ['tag2_A', 'tag2_B', 'tag2_C']),
    (['tag1_llm7', 'tag1_llm8', 'tag1_llm9', 'tag1_llm10', 'tag1_llm11'], "Connectedness Presence", "output/Combined_Files/connected.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm7', 'tag2_llm8', 'tag2_llm9', 'tag2_llm10', 'tag2_llm11'], "Connectedness Classification", "output/Combined_Files/connected.csv", ['tag2_A', 'tag2_B', 'tag2_C'])
]

label_task_pairs_gpt4o: List[Tuple[List[str], str, str, List[str]]] = [
    (['label_llm16', 'label_llm66', 'label_llm67', 'label_llm68', 'label_llm69'], "Business Proposal Evaluation", "output/Combined_Files/dao.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['label_llm2', 'label_llm20', 'label_llm21', 'label_llm22', 'label_llm23'], "Review Attribute Detection", "output/Combined_Files/review.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['tag1_llm10', 'tag1_llm24', 'tag1_llm25', 'tag1_llm26', 'tag1_llm27'], "Dialog Intent Classification", "output/Combined_Files/dialog.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm10', 'tag2_llm24', 'tag2_llm25', 'tag2_llm26', 'tag2_llm27'], "Dialog Breakdown Analysis", "output/Combined_Files/dialog.csv", ['tag2_A', 'tag2_B', 'tag2_C']),
    (['tag1_llm2', 'tag1_llm19', 'tag1_llm20', 'tag1_llm21', 'tag1_llm22'], "Connectedness Presence", "output/Combined_Files/connected.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm2', 'tag2_llm19', 'tag2_llm20', 'tag2_llm21', 'tag2_llm22'], "Connectedness Classification", "output/Combined_Files/connected.csv", ['tag2_A', 'tag2_B', 'tag2_C'])
]

label_task_pairs_o3mini: List[Tuple[List[str], str, str, List[str]]] = [
    (['label_llm38', 'label_llm70', 'label_llm71', 'label_llm72', 'label_llm73'], "Business Proposal Evaluation", "output/Combined_Files/dao.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['label_llm12', 'label_llm24', 'label_llm25', 'label_llm26', 'label_llm27'], "Review Attribute Detection", "output/Combined_Files/review.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['tag1_llm12', 'tag1_llm28', 'tag1_llm29', 'tag1_llm30', 'tag1_llm31'], "Dialog Intent Classification", "output/Combined_Files/dialog.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm12', 'tag2_llm28', 'tag2_llm29', 'tag2_llm30', 'tag2_llm31'], "Dialog Breakdown Analysis", "output/Combined_Files/dialog.csv", ['tag2_A', 'tag2_B', 'tag2_C']),
    (['tag1_llm5', 'tag1_llm23', 'tag1_llm24', 'tag1_llm25', 'tag1_llm26'], "Connectedness Presence", "output/Combined_Files/connected.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm5', 'tag2_llm23', 'tag2_llm24', 'tag2_llm25', 'tag2_llm26'], "Connectedness Classification", "output/Combined_Files/connected.csv", ['tag2_A', 'tag2_B', 'tag2_C'])
]

label_task_pairs_gemini15pro: List[Tuple[List[str], str, str, List[str]]] = [
    (['label_llm8', 'label_llm78', 'label_llm79', 'label_llm80', 'label_llm81'], "Business Proposal Evaluation", "output/Combined_Files/dao.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['label_llm4', 'label_llm32', 'label_llm33', 'label_llm34', 'label_llm35'], "Review Attribute Detection", "output/Combined_Files/review.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['tag1_llm9', 'tag1_llm36', 'tag1_llm37', 'tag1_llm38', 'tag1_llm39'], "Dialog Intent Classification", "output/Combined_Files/dialog.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm9', 'tag2_llm36', 'tag2_llm37', 'tag2_llm38', 'tag2_llm39'], "Dialog Breakdown Analysis", "output/Combined_Files/dialog.csv", ['tag2_A', 'tag2_B', 'tag2_C']),
    (['tag1_llm13', 'tag1_llm31', 'tag1_llm32', 'tag1_llm33', 'tag1_llm34'], "Connectedness Presence", "output/Combined_Files/connected.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm13', 'tag2_llm31', 'tag2_llm32', 'tag2_llm33', 'tag2_llm34'], "Connectedness Classification", "output/Combined_Files/connected.csv", ['tag2_A', 'tag2_B', 'tag2_C'])
]

label_task_pairs_deepseekr1: List[Tuple[List[str], str, str, List[str]]] = [
    (['label_llm57', 'label_llm82', 'label_llm83', 'label_llm84', 'label_llm85'], "Business Proposal Evaluation", "output/Combined_Files/dao.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['label_llm14', 'label_llm36', 'label_llm37', 'label_llm38', 'label_llm39'], "Review Attribute Detection", "output/Combined_Files/review.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['tag1_llm14', 'tag1_llm40', 'tag1_llm41', 'tag1_llm42', 'tag1_llm43'], "Dialog Intent Classification", "output/Combined_Files/dialog.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm14', 'tag2_llm40', 'tag2_llm41', 'tag2_llm42', 'tag2_llm43'], "Dialog Breakdown Analysis", "output/Combined_Files/dialog.csv", ['tag2_A', 'tag2_B', 'tag2_C']),
    (['tag1_llm12', 'tag1_llm35', 'tag1_llm36', 'tag1_llm37', 'tag1_llm38'], "Connectedness Presence", "output/Combined_Files/connected.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm12', 'tag2_llm35', 'tag2_llm36', 'tag2_llm37', 'tag2_llm38'], "Connectedness Classification", "output/Combined_Files/connected.csv", ['tag2_A', 'tag2_B', 'tag2_C'])
]

label_task_pairs_claude35sonnet: List[Tuple[List[str], str, str, List[str]]] = [
    (['label_llm9', 'label_llm74', 'label_llm75', 'label_llm76', 'label_llm77'], "Business Proposal Evaluation", "output/Combined_Files/dao.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['label_llm5', 'label_llm28', 'label_llm29', 'label_llm30', 'label_llm31'], "Review Attribute Detection", "output/Combined_Files/review.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['tag1_llm8', 'tag1_llm32', 'tag1_llm33', 'tag1_llm34', 'tag1_llm35'], "Dialog Intent Classification", "output/Combined_Files/dialog.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm8', 'tag2_llm32', 'tag2_llm33', 'tag2_llm34', 'tag2_llm35'], "Dialog Breakdown Analysis", "output/Combined_Files/dialog.csv", ['tag2_A', 'tag2_B', 'tag2_C']),
    (['tag1_llm14', 'tag1_llm27', 'tag1_llm28', 'tag1_llm29', 'tag1_llm30'], "Connectedness Presence", "output/Combined_Files/connected.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm14', 'tag2_llm27', 'tag2_llm28', 'tag2_llm29', 'tag2_llm30'], "Connectedness Classification", "output/Combined_Files/connected.csv", ['tag2_A', 'tag2_B', 'tag2_C'])
]

label_task_pairs_gptoss120b: List[Tuple[List[str], str, str, List[str]]] = [
    (['label_llm86', 'label_llm87', 'label_llm88', 'label_llm89', 'label_llm90'], "Business Proposal Evaluation", "output/Combined_Files/dao.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['label_llm40', 'label_llm41', 'label_llm42', 'label_llm43', 'label_llm44'], "Review Attribute Detection", "output/Combined_Files/review.csv", ['tag_A', 'tag_B', 'tag_C']),
    (['tag1_llm44', 'tag1_llm45', 'tag1_llm46', 'tag1_llm47', 'tag1_llm48'], "Dialog Intent Classification", "output/Combined_Files/dialog.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm44', 'tag2_llm45', 'tag2_llm46', 'tag2_llm47', 'tag2_llm48'], "Dialog Breakdown Analysis", "output/Combined_Files/dialog.csv", ['tag2_A', 'tag2_B', 'tag2_C']),
    (['tag1_llm39', 'tag1_llm40', 'tag1_llm41', 'tag1_llm42', 'tag1_llm43'], "Connectedness Presence", "output/Combined_Files/connected.csv", ['tag1_A', 'tag1_B', 'tag1_C']),
    (['tag2_llm39', 'tag2_llm40', 'tag2_llm41', 'tag2_llm42', 'tag2_llm43'], "Connectedness Classification", "output/Combined_Files/connected.csv", ['tag2_A', 'tag2_B', 'tag2_C'])
]


# ---------------------------------------------------------------------------
# Model label to display name mappings per dataset
# ---------------------------------------------------------------------------

dao_prompt_onlyllm_dict: Dict[str, str] = {
    "label_llm16": "GPT4o;Base;Sys",
    "label_llm19": "GPT4o;Persona;Sys",
    "label_llm20": "GPT4o;CoT;Sys",
    "label_llm18": "GPT4o;Base;User",
    "label_llm27": "GPT4o;Persona;User",
    "label_llm28": "GPT4o;CoT;User",

    "label_llm8": "Gemini1.5;Base;Sys",
    "label_llm22": "Gemini1.5;Persona;Sys",
    "label_llm23": "Gemini1.5;CoT;Sys",
    "label_llm21": "Gemini1.5;Base;User",
    "label_llm29": "Gemini1.5;Persona;User",
    "label_llm30": "Gemini1.5;CoT;User",

    "label_llm9": "Claude3.5;Base;Sys",
    "label_llm25": "Claude3.5;Persona;Sys",
    "label_llm26": "Claude3.5;CoT;Sys",
    "label_llm24": "Claude3.5;Base;User",
    "label_llm31": "Claude3.5;Persona;User",
    "label_llm32": "Claude3.5;CoT;User",

    "label_llm11": "LLaMA3.3;Base;Sys",
    "label_llm36": "LLaMA3.3;Persona;Sys",
    "label_llm37": "LLaMA3.3;CoT;Sys",
    "label_llm33": "LLaMA3.3;Base;User",
    "label_llm34": "LLaMA3.3;Persona;User",
    "label_llm35": "LLaMA3.3;CoT;User",
}

dao_model_onlyllm_dict: Dict[str, str] = {
    "label_llm4": "GPT-4 Turbo",
    "label_llm16": "GPT-4o",
    'label_llm58': 'GPT-4.5',
    "label_llm8": "Gemini 1.5 Pro",
    "label_llm10": "LLaMA 3 70B",
    # "label_llm11": "LLaMA 3.1 70B",
    'label_llm39': 'LLaMA 3.3 70B',
    "label_llm9": "Claude 3.5 Sonnet",
    'label_llm38': 'o3-mini',
    'label_llm57': 'DeepSeek-R1',
    "label_llm59": "DeepSeek-R1-LLaMA",
    "label_llm61": "Claude 3.7 Sonnet",
    "label_llm62": "Gemini 2.5 Pro",
    'label_llm63': 'GPT-4.1',
    'label_llm86': 'GPT-OSS 120B',
    "label_llm60": "Initial Guidelines GPT-4o",
    'tag_crowd_majority': 'Crowdsourced',
    # "label_majority_5llms": "Combined_5LLMs",
}

review_model_onlyllm_dict: Dict[str, str] = {
    "label_llm3": "GPT-4 Turbo",
    "label_llm2": "GPT-4o",
    'label_llm13': 'GPT-4.5',
    "label_llm4": "Gemini 1.5 Pro",
    "label_llm6": "LLaMA 3 70B",
    'label_llm7': 'LLaMA 3.3 70B',
    "label_llm5": "Claude 3.5 Sonnet",
    'label_llm12': 'o3-mini',
    'label_llm14': 'DeepSeek-R1',
    "label_llm17": "Claude 3.7 Sonnet",
    "label_llm18": "Gemini 2.5 Pro",
    'label_llm19': 'GPT-4.1',
    'label_llm40': 'GPT-OSS 120B',
    "label_llm16": "Initial Guidelines GPT-4o",
    'tag_crowd_majority': 'Crowdsourced',
    # "label_majority_5llms": "Combined_5LLMs",
}

dialog_tag1_model_onlyllm_dict: Dict[str, str] = {
    "tag1_llm1": "GPT-4 Turbo",
    "tag1_llm10": "GPT-4o",
    'tag1_llm13': 'GPT-4.5',
    "tag1_llm9": "Gemini 1.5 Pro",
    "tag1_llm3": "LLaMA 3 70B",
    'tag1_llm11': 'LLaMA 3.3 70B',
    "tag1_llm8": "Claude 3.5 Sonnet",
    'tag1_llm12': 'o3-mini',
    'tag1_llm14': 'DeepSeek-R1',
    "tag1_llm21": "Claude 3.7 Sonnet",
    "tag1_llm22": "Gemini 2.5 Pro",
    "tag1_llm15": "DeepSeek-R1-LLaMA",
    'tag1_llm23': 'GPT-4.1',
    'tag1_llm44': 'GPT-OSS 120B',
    "tag1_llm16": "Initial Guidelines GPT-4o",
    'tag1_crowd_majority': 'Crowdsourced',
    # "tag1_majority_5llms": "Combined_5LLMs",
}

dialog_tag2_model_onlyllm_dict: Dict[str, str] = {
    "tag2_llm1": "GPT-4 Turbo",
    "tag2_llm10": "GPT-4o",
    'tag2_llm13': 'GPT-4.5',
    "tag2_llm9": "Gemini 1.5 Pro",
    "tag2_llm3": "LLaMA 3 70B",
    'tag2_llm11': 'LLaMA 3.3 70B',
    "tag2_llm8": "Claude 3.5 Sonnet",
    'tag2_llm12': 'o3-mini',
    'tag2_llm14': 'DeepSeek-R1',
    "tag2_llm21": "Claude 3.7 Sonnet",
    "tag2_llm22": "Gemini 2.5 Pro",
    'tag2_llm23': 'GPT-4.1',
    'tag2_llm44': 'GPT-OSS 120B',
    "tag2_llm16": "Initial Guidelines GPT-4o",
    'tag2_crowd_majority': 'Crowdsourced',
    # "tag2_majority_5llms": "Combined_5LLMs",
}

connected_tag1_dict: Dict[str, str] = {
    # "tag1_llm1": "GPT-4o; Base",
    "tag1_llm2": "GPT-4o",  # ; Persona
    "tag1_llm3": "GPT-4 Turbo",
    "tag1_llm4": "GPT-4.5",
    "tag1_llm5": "o3-mini",
    'tag1_llm12': 'DeepSeek-R1',
    'tag1_llm13': 'Gemini 1.5 Pro',
    'tag1_llm14': 'Claude 3.5 Sonnet',
    'tag1_llm17': 'LLaMA 3 70B',
    'tag1_llm7': 'LLaMA 3.3 70B',
    "tag1_llm15": "Claude 3.7 Sonnet",
    "tag1_llm16": "Gemini 2.5 Pro",
    'tag1_llm18': 'GPT-4.1',
    'tag1_llm40': 'GPT-OSS 120B',
    "tag1_llm6": "Initial Guidelines GPT-4o",
    'tag1_crowd_majority': 'Crowdsourced',
    # "tag1_majority_5llms": "Combined_5LLMs",
}

connected_tag2_dict: Dict[str, str] = {
    # "tag2_llm1": "GPT-4o; Base",
    "tag2_llm2": "GPT-4o",  # ; Persona
    "tag2_llm3": "GPT-4 Turbo",
    "tag2_llm4": "GPT-4.5",
    "tag2_llm5": "o3-mini",
    'tag2_llm12': 'DeepSeek-R1',
    'tag2_llm13': 'Gemini 1.5 Pro',
    'tag2_llm14': 'Claude 3.5 Sonnet',
    'tag2_llm17': 'LLaMA 3 70B',
    'tag2_llm7': 'LLaMA 3.3 70B',
    "tag2_llm15": "Claude 3.7 Sonnet",
    "tag2_llm16": "Gemini 2.5 Pro",
    'tag2_llm18': 'GPT-4.1',
    'tag2_llm40': 'GPT-OSS 120B',
    "tag2_llm6": "Initial Guidelines GPT-4o",
    'tag2_crowd_majority': 'Crowdsourced',
    # "tag2_majority_5llms": "Combined_5LLMs",
}

fearspeech_dict: Dict[str, str] = {
    "label_llm18": "GPT-4 Turbo",
    "label_llm23": "GPT-4o",
    "label_llm34": "GPT-4.5",
    "label_llm25": "Gemini 1.5 Pro",
    "label_llm20": "LLaMA 3 70B",
    'label_llm28': 'LLaMA 3.3 70B',
    "label_llm24": "Claude 3.5 Sonnet",
    'label_llm30': 'o3-mini',
    'label_llm31': 'DeepSeek-R1',
    "label_llm32": "Claude 3.7 Sonnet",
    "label_llm33": "Gemini 2.5 Pro",
    "label_llm35": "GPT-4.1",
    'label_llm36': 'GPT-OSS 120B',
    # "label_majority_5llms": "Combined_5LLMs",
}

tweet_dict: Dict[str, str] = {
    "label_llm3": "GPT-4 Turbo",
    "label_llm8": "GPT-4o",
    "label_llm17": "GPT-4.5",
    "label_llm6": "Gemini 1.5 Pro",
    'label_llm10': 'LLaMA 3 70B',
    'label_llm11': 'LLaMA 3.3 70B',
    "label_llm7": "Claude 3.5 Sonnet",
    'label_llm13': 'o3-mini',
    'label_llm14': 'DeepSeek-R1',
    "label_llm15": "Claude 3.7 Sonnet",
    "label_llm16": "Gemini 2.5 Pro",
    "label_llm18": "GPT-4.1",
    'label_llm19': 'GPT-OSS 120B',
    # "label_majority_5llms": "Combined_5LLMs",
}

sentiment_dict: Dict[str, str] = {
    "label_llm1": "GPT-4 Turbo",
    "label_llm4": "GPT-4o",
    "label_llm14": "GPT-4.5",
    "label_llm5": "Gemini 1.5 Pro",
    'label_llm3': 'LLaMA 3 70B',
    'label_llm8': 'LLaMA 3.3 70B',
    "label_llm6": "Claude 3.5 Sonnet",
    'label_llm10': 'o3-mini',
    'label_llm11': 'DeepSeek-R1',
    "label_llm12": "Claude 3.7 Sonnet",
    "label_llm13": "Gemini 2.5 Pro",
    "label_llm15": "GPT-4.1",
    'label_llm16': 'GPT-OSS 120B',
    # "label_majority_5llms": "Combined_5LLMs",
}


__all__ = [
    # label-task pairs
    'label_task_pairs_llama33',
    'label_task_pairs_gpt4o',
    'label_task_pairs_o3mini',
    'label_task_pairs_gemini15pro',
    'label_task_pairs_deepseekr1',
    'label_task_pairs_claude35sonnet',
    'label_task_pairs_gptoss120b',
    # mappings
    'dao_prompt_onlyllm_dict',
    'dao_model_onlyllm_dict',
    'review_model_onlyllm_dict',
    'dialog_tag1_model_onlyllm_dict',
    'dialog_tag2_model_onlyllm_dict',
    'connected_tag1_dict',
    'connected_tag2_dict',
    'fearspeech_dict',
    'tweet_dict',
    'sentiment_dict',
]


