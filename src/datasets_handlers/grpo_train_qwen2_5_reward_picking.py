# 导入内置包 
import re
import argparse
import importlib
import os
import sys
import random

from typing import (
    Optional
)

from dataclasses import (
    dataclass, field
)

# 导入torch
import torch.distributed as dist


# 导入HF toolkit
import datasets

from datasets import (
    load_dataset
)

from transformers import (
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer
)

from qwen_vl_utils import process_vision_info

from trl import (
    GRPOConfig,
    GRPOTrainer, 
    ModelConfig, 
    ScriptArguments, 
    TrlParser, 
    get_peft_config
)

default_fps = 1. 

max_pixels = 14 * 14 * 80

total_pixels = 1024 * 28 * 28

question_template = \
"""\
    Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos.\
    Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption),\
    temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail),\
    coordination of human movement(with emphasis on unrealistic limbs movements and distortions),and any other factors you deem relevant.\
    For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score.\
    Calculate the total score for each video by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags.\
    Then, in the <answer> tag, output exactly one of the following strings: 'Video 1 is better' or 'Video 2 is better' based on the total scores. No additional text is allowed in the <answer> section.\
    \n\nExample output format:\n<think>1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...\n2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...\n3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) \
    - ...\n4. Coordination of human movement: Video 1 (6/10) - ...; Video 2 (8/10) - ... \n[Additional dimensions if any]: Video 1 (6/10) - ...;  Video 2 (8/10) - ...\nTotal score:\nVideo 1: 9+8+7+6+6=36\nVideo 2: 7+6+5+8+8=34</think><answer>Video 1 is better</answer>\
    Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.*\
    \n\nYour task is provided as follows:\nText Caption: [{prompt}]\n\
"""

def selection_identify(selction: list = None, default_selection: list = None):
    if default_selection is None:
        default_selection = ["chosen_video_path", "rejected_video_path"]
    return int(selction == default_selection)

def generate_prompts(processor, sample, video_paths, curr_fps, system_prompt = None):
        left_video, right_video = video_paths
        message = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question_template.format(prompt=sample["caption"])
                    },
                    {
                        "type": "text",
                        "text": "This is the start of Video 1:\n"
                    },
                    {
                        "type": "video",
                        "video": f"{sample[left_video]}",
                        "max_pixels": max_pixels,
                        "total_pixels": total_pixels,
                        "fps": curr_fps,
                    },
                    {
                        "type": "text",
                        "text": "\nThis is the end of Video 1.\n\nThis is the start of Video 2:\n"
                    },
                    {
                        "type": "video",
                        "video": f"{sample[right_video]}",
                        "max_pixels": max_pixels,
                        "total_pixels": total_pixels,
                        "fps": curr_fps,
                    },
                    {
                        "type": "text", 
                        "text": "\nThis is the end of Video 2.\n\n"
                    },
                ],
            }
        ]
        prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return prompt, message


def load_human_body(data_path:str, args = None):

    if hasattr(args, "fps"):
        global default_fps
        default_fps = args.fps
    
    dataset = load_dataset(
        path = data_path,
        )["train"]
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    train_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset) * args.data_select_ratio)))
    val_dataset = val_dataset.shuffle(seed=42).select(range(int(len(val_dataset) * args.data_select_ratio)))
    train_dataset = train_dataset.select_columns(["chosen_video_path", "rejected_video_path", "caption"])
    val_dataset = val_dataset.select_columns(["chosen_video_path", "rejected_video_path", "caption"])

    return train_dataset, val_dataset


def _human_body_preprocess_handler(sample, processor=None, system_prompt = None):
    try: 
        model_inputs = {}

        curr_fps = default_fps
        video_paths = ["chosen_video_path", "rejected_video_path"]
        random.shuffle(video_paths)
        selection = selection_identify(video_paths)

        prompt, message = generate_prompts(processor, sample, video_paths, curr_fps, system_prompt)
            
        model_inputs["selections"] = selection
        model_inputs["prompts_text"] = prompt
        model_inputs["message"] = message
        return model_inputs
    except Exception as exp:
        print(exp)
        return None

def human_body_preprocess_handler(dataset: datasets.Dataset = None, processor = None, system_prompt = None):
    dataset = dataset.map(
        function = _human_body_preprocess_handler,
        fn_kwargs = {
            "processor" : processor,
            "system_prompt": system_prompt,
        },
        batched=False,
        load_from_cache_file = False,
    )
    return dataset