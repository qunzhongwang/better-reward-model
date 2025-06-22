import os
import time
import json
import re 
import pickle as pkl
from abc import ABC
from copy import deepcopy, copy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict

from tqdm import tqdm
from PIL import Image
import numpy as np

import ray
import torch
import torch.nn as nn
import torch.distributed as dist
from vllm import SamplingParams

import datasets
from datasets import interleave_datasets, load_dataset

from openrlhf.models.actor import Actor
from openrlhf.models.utils import (
    compute_approx_kl, 
    compute_reward, 
    masked_mean, 
    unpacking_samples
)

from openrlhf.utils.logging_utils import init_logger

# from openrlhf.trainer.ppo_utils.data_processor import add_pixel_bounds

from qwen_vl_utils import (
    smart_resize,
    process_vision_info, 
    extract_vision_info, 
    fetch_image
)

from collections import defaultdict


# pip install math-verify

from math_verify import (
    parse, 
    verify
)

from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.fncglobal_prompts.nous_fncglobal_prompt import (
    NousFnCallPrompt,
    Message,
    ContentItem,
)


logger = init_logger(__name__)

def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")

@register_tool("select_frames_null")
class SelectFrames(BaseTool):
    @property
    def description(self):
        return """
Select frames from a video.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "target_frames": {
                "type": "array",
                "description": "List of frame indices to select from the video (no more than 8 frames in total).",
                "items": {
                    "type": "integer",
                    "description": "Frame index from 1 to 240."
                }
            }
        },
        "required": ["target_frames"]
    }

    def call(self, images, target_frames):
        return [images[tgt] for tgt in target_frames]


@register_tool("crop_image_normalized_null")
class CropImageNormalized(BaseTool):
    @property
    def description(self):
        return """
Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.
""".strip()

    parameters = {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "description": "coordinates for bounding box of the area you want to zoom in. Values should be within [0.0,1.0].",
                "items": {
                    "type": "number",
                }
            },
            "target_image":{
                "type": "number",
                "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."
            }
        },
        "required": ["bbox_2d", "target_image"]
    }
    
    def call(self, image, bbox_2d,  padding=0.1):
        """
        Crop the image based on the bounding box coordinates.
        """
        image_x, image_y = image.size
        if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
            normalized_bbox_2d = (float(bbox_2d[0])-padding, float(bbox_2d[1])-padding, float(bbox_2d[2])+padding, float(bbox_2d[3])+padding)
        else:
            normalized_bbox_2d = (float(bbox_2d[0])/image_x-padding, float(bbox_2d[1])/image_y-padding, float(bbox_2d[2])/image_x+padding, float(bbox_2d[3])/image_y+padding)
        normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
        normalized_x1 =min(max(0, normalized_x1), 1)
        normalized_y1 =min(max(0, normalized_y1), 1)
        normalized_x2 =min(max(0, normalized_x2), 1)
        normalized_y2 =min(max(0, normalized_y2), 1)
        cropped_image = image.crop((normalized_x1*image_x, normalized_y1*image_y, normalized_x2*image_x, normalized_y2*image_y))
        w, h = cropped_image.size
        assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"



        return cropped_image 
    

def extract_qwen_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("<|im_start|>assistant\n")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = "".join(parts[1:])
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("<|im_start|>user\n")[1].split('<|im_end|>')[0].split('<|vision_end|>')[-1]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response


def extract_dsmath_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("Assistant:")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("User:")[1].strip()
    
    # Return the user query and the assistant's response
    return user_query, assistant_response


def extract_dpsk_query_and_response(input_text):
    # Split the input text by the assistant's start token
    # print(input_text)
    parts = input_text.split("<|Assistant|>")
    
    # The first part contains the system and user messages
    if len(parts)==0:
        print('!!!! warning extraction', input_text)
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("<|User|>")[1]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response

def extract_llama_query_and_response(input_text):
    # Split the input text by the assistant's start token
    parts = input_text.split("assistant<|end_header_id|>\n\n")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("user<|end_header_id|>\n\n")[1].split('<|eot_id|><|start_header_id|>')[0]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response

def extract_autocode_query_and_response(input_text):
    # print('!!!! example input', input_text)
    # Split the input text by the assistant's start token
    parts = input_text.split("Response:")
    
    # The first part contains the system and user messages
    user_part = parts[0]
    
    # The second part contains the assistant's response
    if len(parts)==1: assistant_response = ""
    else: assistant_response = parts[1]
    
    # Extract the user query by splitting the user part
    user_query = user_part.split("### Instruction:\n")[1].split('\n\n### ')[0]
    
    # Return the user query and the assistant's response
    return user_query, assistant_response
    
def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor

