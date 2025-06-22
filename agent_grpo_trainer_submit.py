# 导入内置包 
import re
import argparse
import importlib
import os
import sys

from typing import (
    Optional
)

from dataclasses import (
    dataclass, field
)

# 导入torch
import torch.distributed as dist


# 导入HF toolkit
from datasets import (
    load_dataset
)

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
)

from trl import (
    GRPOConfig,
    GRPOTrainer,
    GRPOTrainer_qwen, 
    GRPOTrainer_agent_qwen,
    ModelConfig, 
    ScriptArguments, 
    TrlParser, 
    get_peft_config
)

# 导入自定义方法
from src.datasets_handlers.grpo_train_qwen2_5_reward_picking import (
    load_human_body, 
    human_body_preprocess_handler
)


def dist_debug():
    if dist.get_rank() == 0 :
        breakpoint()
    dist.barrier()


def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["caption"]},
        ],
    }


def select_data_pipeline(preprocess_handler):
    if preprocess_handler == "qwen2.5-humanbody-grpo":
        return load_human_body, human_body_preprocess_handler
    else:
        raise NotImplementedError("not implemented")


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    # breakpoint()
    completion_contents = [completion for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]

system_prompt = None
reward_funcs_registry = None
def get_asset():

    global system_prompt, reward_funcs_registry

    system_prompt = (
        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
        "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
        "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
        "<think> reasoning process here </think><answer> answer here </answer>"
    )

    reward_funcs_registry = {
        "think_format_reward": format_reward,
    }

def update_image_or_video_reward(args):

    get_asset()

    global system_prompt, reward_funcs_registry

    if not hasattr(args, "data_source"):
        args.data_source = "image"
    if args.data_source == "image":
        reward_funcs_registry["accuracy_reward"] = pick_correct_image_reward
    else:
        reward_funcs_registry["accuracy_reward"] = pick_correct_video_reward

def pick_correct_video_reward(completions, **kwargs):

    image_left_is_better =  kwargs["selections"]
    completion_contents = [completion for completion in completions]
    rewards = []
    predictons = []
    for content, selection in zip(completion_contents, image_left_is_better):
        
        if "video 1 is better" in content.lower():
            predictons.append(1)
        elif "video 2 is better" in content.lower():
            predictons.append(0)
        else:
            predictons.append(-1)

        if predictons[-1] == selection:
            rewards.append(1.)
        else:
            rewards.append(0.)
            
    return rewards

def pick_correct_image_reward(completions, **kwargs):
    image_left_is_better =  kwargs["selections"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    predictons = []
    for content, selection in zip(completion_contents, image_left_is_better):
        
        if "image 1 is better" in content.lower():
            predictons.append(1)
        elif "image 2 is better" in content.lower():
            predictons.append(0)
        else:
            predictons.append(-1)

        if predictons[-1] == selection:
            rewards.append(1.)
        else:
            rewards.append(0.)
            
    return rewards



@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_name_or_path (`str` or `None`, *optional*, defaults to `None`):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
        reward_funcs (`list[str]` or `None`, *optional*, defaults to `None`):
            Reward functions to use. It can be either one of  `"think_format_reward"`; or a dotted import path "
            (e.g., `'my_lib.rewards.custom_reward'`).
    """

    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: Optional[list[str]] = field(
        default_factory= lambda :["think_format_reward", "accuracy_reward"],
        metadata={
            "help": "Reward functions to use. It can be either one of  'think_format_reward'; or a dotted "
            "import path. (e.g., 'my_lib.rewards.custom_reward')."
        },
    )

    save_last_checkpoint: bool = False

    debug_entry_point: bool = False

    data_source: str = "image"

    data_pipeline:str = "qwen2.5-humanbody-grpo"

    data_select_ratio:float = 0.1

    cache_dir: str = None

    fps: float = 1.

    prompt_data: str = "test"

    eval_data: str = "test"


def main(script_args, training_args, model_args):
    
    update_image_or_video_reward(script_args)

    # Get the reward models and functions
    reward_funcs = []

    if script_args.reward_model_name_or_path:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            script_args.reward_model_name_or_path, trust_remote_code=model_args.trust_remote_code, num_labels=1
        )
        reward_funcs.append(reward_model)

    if script_args.reward_funcs:
        for func_name in script_args.reward_funcs:
            if func_name in reward_funcs_registry:
                reward_funcs.append(reward_funcs_registry[func_name])
            elif "." in func_name:
                # 
                module_path, func_name = func_name.rsplit(".", 1)
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(module_path)
                reward_func = getattr(module, func_name)
                reward_funcs.append(reward_func)
            else:
                raise ValueError(
                    f"Could not load reward function '{func_name}'. Expected one of "
                    f"{list(reward_funcs_registry.keys())} or a valid import path."
                )

    # Load the dataset
    dataloader_handler, preprocess_handler = select_data_pipeline(script_args.data_pipeline)
    
    train_dataset, test_dataset = dataloader_handler(
        script_args.dataset_name,
        script_args
    )

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    train_dataset = preprocess_handler(train_dataset, processor, system_prompt)
    test_dataset = preprocess_handler(test_dataset, processor, system_prompt)
    del (
        processor
    )

    # Initialize the GRPO trainer
    global trainer 
    
    trainer = GRPOTrainer_agent_qwen(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    if script_args.debug_entry_point:
        dist_debug()

    if training_args.do_train:
        trainer.train()

    if training_args.do_eval:
        trainer.evaluate()

    # Save and push to hub
    if script_args.save_last_checkpoint:
        trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args = parser.parse_args_and_config()
    model_init_kwargs = {}
    for name in ["torch_dtype", "cache_dir"]:
        if hasattr(script_args, name):
            model_init_kwargs[name] = getattr(script_args, name)
        elif hasattr(training_args, name):
            model_init_kwargs[name] = getattr(training_args, name)
    training_args.model_init_kwargs = model_init_kwargs
    training_args.prompt_data = script_args.prompt_data
    training_args.eval_data = script_args.eval_data
    training_args.zero_stage = 2 #script_args.zero_stage
    training_args.advantage_estimator = "group"
    
    main(script_args, training_args, model_args)
