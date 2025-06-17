import torch
import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image
import ray
from vllm import SamplingParams
from dataclasses import dataclass
from copy import copy, deepcopy

# ==================== 工具相关辅助函数 ====================

@register_tool("select_frames")
class SelectFrames(BaseTool):

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

    @property
    def description(self):
        return """Select frames from a video.""".strip()

    def call(self, images, target_frames):
        return [images[tgt] for tgt in target_frames]

@register_tool("crop_image_normalized")
class CropImageNormalized(BaseTool):

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
    
    @property
    def description(self):
        return """Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.""".strip()

    
    def call(self, image, bbox_2d,  padding=0.1):
        """
        Crop the image based on the bounding box coordinates.
        """
        img_x, img_y = image.size
        if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
            normalized_bbox_2d = (float(bbox_2d[0])-padding, float(bbox_2d[1])-padding, float(bbox_2d[2])+padding, float(bbox_2d[3])+padding)
        else:
            normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding, float(bbox_2d[1])/img_y-padding, float(bbox_2d[2])/img_x+padding, float(bbox_2d[3])/img_y+padding)
        normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
        normalized_x1 =min(max(0, normalized_x1), 1)
        normalized_y1 =min(max(0, normalized_y1), 1)
        normalized_x2 =min(max(0, normalized_x2), 1)
        normalized_y2 =min(max(0, normalized_y2), 1)
        cropped_img = image.crop((normalized_x1*img_x, normalized_y1*img_y, normalized_x2*img_x, normalized_y2*img_y))
        w, h = cropped_img.size
        assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"



        return cropped_img 


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """解析最后一个工具调用"""
    tool_end = "</tool_call>"
    if not text.endswith(tool_end):
        return None
    
    try:
        # 这里需要实现具体的解析逻辑
        # 假设工具调用格式为 <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
        import re
        pattern = r'<tool_call>(.*?)</tool_call>'
        matches = re.findall(pattern, text)
        if matches:
            tool_info = json.loads(matches[-1])
            return tool_info
    except:
        return None
    return None

def check_termination_conditions(
    response: str,
    num_tool_calls: int,
    num_images: int,
    total_tokens: int,
    max_tools: int = 3,
    max_images: int = 16,
    max_tokens: int = 12000
) -> Tuple[bool, bool]:
    """检查是否需要终止生成
    返回: (是否需要工具调用, 是否强制终止)
    """
    tool_end = "</tool_call>"
    require_tool = response.endswith(tool_end)
    
    force_terminate = (
        num_tool_calls > max_tools or 
        num_images > max_images or 
        total_tokens > max_tokens - 200
    )
    
    return require_tool, force_terminate

def process_tool_result(
    tool_name: str,
    tool_args: Dict[str, Any],
    images: List[Image.Image],
    raw_images: List[Image.Image],
    is_video: bool,
    operations: Dict[str, Any],
    image_size_config: Dict[str, int]
) -> Tuple[List[Image.Image], str, bool]:
    """处理工具执行结果
    返回: (新增图像列表, 消息文本, 是否出错)
    """
    error_flag = False
    added_images = []
    message = ""
    
    try:
        if tool_name == 'select_frames':
            if not is_video:
                message = "Execution error:\nYou attempted to select frames from an **image**, but this operation is only designed for analyzing videos. Think again.\n"
                return [], message, True
            
            selected_frames, info = execute_tool(
                images, raw_images, tool_args, tool_name, 
                is_video=is_video, 
                function=operations[tool_name].call
            )
            
            if isinstance(info, str):
                message = f"\n{info}"
                if len(selected_frames) == 0:
                    return [], message, False
            
            # 调整图像大小
            added_images = [
                resize_cropped(
                    frame, 
                    min_pixels=image_size_config['select_min_pixels'],
                    max_pixels=image_size_config['select_max_pixels']
                ) for frame in selected_frames
            ]
            
            if added_images:
                size = added_images[0].size
                message = f"\nHere are the selected frames (Frame Size: {size[0]}x{size[1]}, Numbered {len(images)} to {len(selected_frames)+len(images)-1}):"
                
        else:  # crop_image
            cropped = execute_tool(
                images, raw_images, tool_args, tool_name,
                is_video=is_video,
                function=operations[tool_name].call
            )
            
            processed_img = resize_cropped(
                cropped,
                min_pixels=image_size_config['crop_min_pixels'],
                max_pixels=image_size_config['crop_max_pixels']
            )
            
            added_images = [processed_img]
            size = processed_img.size
            message = f"\nHere is the cropped image (Image Size: {size[0]}x{size[1]}):"
            
    except Exception as e:
        error_flag = True
        message = f"\nExecution error:\n{str(e)}\n"
        
    return added_images, message, error_flag

def create_tool_response_message(
    message: str,
    images: List[Image.Image]
) -> Dict[str, Any]:
    """创建工具响应消息"""
    content = [{'type': 'text', 'text': message}]
    content.extend([{'type': 'image', 'image': img} for img in images])
    
    return {
        'role': 'user',
        'content': content
    }

# ==================== VLLM 版本 ====================

def generate_with_tools_vllm(
    prompts: List[str],
    vllm_engines: List[Any],
    tokenizer: Any,
    data_processor: Any,
    operations: Dict[str, Any],
    strategy_args: Any,
    is_eval: bool = False,
    **kwargs
) -> List[Any]:
    """使用VLLM进行多轮对话工具链生成"""
    
    # 配置参数
    config = {
        'max_turns': getattr(strategy_args, "maxturn", 2) - 1,
        'max_images': 16,
        'max_new_tokens': getattr(strategy_args, "max_out_tokens", 2048),
        'temperature': 0.0 if is_eval else getattr(strategy_args, "temperature", 0.85),
        'top_p': 1.0 if is_eval else kwargs.get("top_p", 0.95),
        'stop_tokens': ['<|im_end|>', '<|eot_id|>', '<|endoftext|>'],
        'image_sizes': {
            'raw_max': 2000,
            'zoom_max': 1000,
            'select_max': 400,
            'eval_min': 256 if is_eval else 4,
            'eval_max': 8000 if is_eval else 5120,
        }
    }
    
    # 初始化数据结构
    all_conversations = {}
    all_images = {}
    all_raw_images = {}
    all_outputs = []
    num_tool_calls = []
    num_tool_fails = []
    video_flags = []
    
    # 准备初始输入
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    
    # 扩展prompts以支持多个样本
    n_samples = 1 if is_eval else strategy_args.n_samples_per_prompt
    expanded_prompts = []
    expanded_qids = []
    
    for prompt in prompts:
        info = json.loads(prompt)
        qid = info[-1].get('qid', 'unknown')
        clean_prompt = json.dumps(info[:-1])
        
        for _ in range(n_samples):
            expanded_prompts.append(clean_prompt)
            expanded_qids.append(qid)
            num_tool_calls.append(0)
            num_tool_fails.append(0)
    
    # 统一的生成循环
    turn = 0
    active_indices = list(range(len(expanded_prompts)))
    
    while active_indices and turn <= config['max_turns']:
        # 准备当前轮次的输入
        current_vllm_inputs = []
        current_indices = []
        
        for idx in active_indices:
            uid = f"{expanded_qids[idx]}-{idx}"
            
            if turn == 0:
                # 初始轮次：处理原始输入
                message = expanded_prompts[idx]
                prompt_text, conversations = get_prompt_from_messages(
                    [message], prompt_maker, tools, data_processor.processor
                )
                
                conversations, images, is_video = data_processor.obtain_conv_images_from_conversations(
                    conversations,
                    batch_min_pixels=[config['image_sizes']['eval_min'] * 28 * 28],
                    batch_max_pixels=[config['image_sizes']['eval_max'] * 28 * 28]
                )
                
                all_conversations[uid] = conversations[0]
                all_images[uid] = images[0] if is_video else images[0]
                all_raw_images[uid] = images[0]
                video_flags.append(is_video)
                
                # 处理视频帧选择
                if is_video and len(images[0]) > 8:
                    step = max(1, len(images[0]) // 8)
                    selected_frames = images[0][::step][:8]
                    all_images[uid] = [selected_frames]
                    # 更新conversation中的视频引用
                
            else:
                # 后续轮次：检查是否需要工具调用
                last_output = all_outputs[idx]
                response_text = tokenizer.decode(
                    last_output.outputs[0].token_ids,
                    skip_special_tokens=False
                )
                
                require_tool, force_terminate = check_termination_conditions(
                    response_text,
                    num_tool_calls[idx],
                    len(all_images[uid]),
                    len(last_output.prompt_token_ids) + len(last_output.outputs[0].token_ids),
                    config['max_turns'],
                    config['max_images']
                )
                
                if not require_tool or force_terminate:
                    # 不需要继续生成
                    continue
                
                # 解析并执行工具
                tool_info = parse_tool_call(response_text)
                if not tool_info:
                    num_tool_fails[idx] += 1
                    continue
                
                added_images, message, error_flag = process_tool_result(
                    tool_info['name'],
                    tool_info['arguments'],
                    all_images[uid],
                    all_raw_images[uid],
                    video_flags[idx],
                    operations,
                    {
                        'select_min_pixels': config['image_sizes']['eval_min'] * 28 * 28,
                        'select_max_pixels': config['image_sizes']['select_max'] * 28 * 28,
                        'crop_min_pixels': config['image_sizes']['eval_min'] * 28 * 28,
                        'crop_max_pixels': config['image_sizes']['zoom_max'] * 28 * 28,
                    }
                )
                
                if error_flag:
                    num_tool_fails[idx] += 1
                
                # 更新对话和图像
                assistant_msg = {
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': response_text}]
                }
                user_msg = create_tool_response_message(message, added_images)
                
                all_conversations[uid].extend([assistant_msg, user_msg])
                all_images[uid].extend(added_images)
                num_tool_calls[idx] += 1
            
            # 准备VLLM输入
            prompt_text = data_processor.processor.apply_chat_template(
                [all_conversations[uid]],
                tokenize=False,
                add_generation_prompt=True
            )[0]
            
            vllm_input = {
                "prompt": prompt_text,
                "multi_modal_data": {
                    "video" if video_flags[idx] else "image": all_images[uid]
                }
            }
            
            current_vllm_inputs.append(vllm_input)
            current_indices.append(idx)
        
        if not current_vllm_inputs:
            break
        
        # 使用VLLM生成
        sampling_params = SamplingParams(
            temperature=config['temperature'] if turn == 0 else 0.9,
            top_p=config['top_p'],
            max_tokens=config['max_new_tokens'],
            stop=config['stop_tokens'],
            include_stop_str_in_output=False
        )
        
        # 分配到不同的VLLM引擎
        batch_size = (len(current_vllm_inputs) + len(vllm_engines) - 1) // len(vllm_engines)
        refs = []
        
        for i, llm in enumerate(vllm_engines):
            batch_inputs = current_vllm_inputs[i * batch_size : (i + 1) * batch_size]
            if batch_inputs:
                refs.append(
                    llm.add_requests_vlm.remote(
                        rank,
                        sampling_params=sampling_params,
                        vllm_vision_input=batch_inputs
                    )
                )
        
        ray.get(refs)
        torch.distributed.barrier()
        
        # 获取结果
        output_refs = []
        for llm in vllm_engines:
            output_refs.append(llm.get_responses.remote(rank))
        
        batch_outputs = sum(ray.get(output_refs), [])
        
        # 更新输出
        for i, idx in enumerate(current_indices):
            if turn == 0:
                all_outputs.append(batch_outputs[i])
            else:
                all_outputs[idx] = batch_outputs[i]
        
        # 更新活跃索引
        active_indices = current_indices
        turn += 1
    
    # 后处理和构建返回结果
    return build_samples_from_outputs(
        all_outputs, all_conversations, all_images,
        expanded_qids, tokenizer, data_processor,
        strategy_args
    )

# ==================== Transformer 版本 ====================

def generate_with_tools_transformer(
    prompts: List[str],
    model: Any,
    tokenizer: Any,
    data_processor: Any,
    operations: Dict[str, Any],
    strategy_args: Any,
    is_eval: bool = False,
    device: str = 'cuda',
    **kwargs
) -> List[Any]:
    """使用普通Transformer进行多轮对话工具链生成"""
    
    # 配置参数
    config = {
        'max_turns': getattr(strategy_args, "maxturn", 2) - 1,
        'max_images': 16,
        'max_new_tokens': getattr(strategy_args, "max_out_tokens", 2048),
        'temperature': 0.0 if is_eval else getattr(strategy_args, "temperature", 0.85),
        'top_p': 1.0 if is_eval else kwargs.get("top_p", 0.95),
        'stop_tokens': ['<|im_end|>', '<|eot_id|>', '<|endoftext|>'],
        'image_sizes': {
            'raw_max': 2000,
            'zoom_max': 1000,
            'select_max': 400,
            'eval_min': 256 if is_eval else 4,
            'eval_max': 8000 if is_eval else 5120,
        }
    }
    
    # 初始化数据结构
    all_conversations = {}
    all_images = {}
    all_raw_images = {}
    all_outputs = []
    all_generated_texts = []
    num_tool_calls = []
    num_tool_fails = []
    video_flags = []
    
    # 扩展prompts
    n_samples = 1 if is_eval else strategy_args.n_samples_per_prompt
    expanded_prompts = []
    expanded_qids = []
    
    for prompt in prompts:
        info = json.loads(prompt)
        qid = info[-1].get('qid', 'unknown')
        clean_prompt = json.dumps(info[:-1])
        
        for _ in range(n_samples):
            expanded_prompts.append(clean_prompt)
            expanded_qids.append(qid)
            num_tool_calls.append(0)
            num_tool_fails.append(0)
    
    # 统一的生成循环
    turn = 0
    active_indices = list(range(len(expanded_prompts)))
    
    while active_indices and turn <= config['max_turns']:
        # 准备批次输入
        batch_inputs = []
        batch_visual_inputs = []
        current_indices = []
        
        for idx in active_indices:
            uid = f"{expanded_qids[idx]}-{idx}"
            
            if turn == 0:
                # 初始轮次处理
                message = expanded_prompts[idx]
                conversations = json.loads(message)
                
                # 处理图像和视频
                processed_conv, images, is_video = data_processor.obtain_conv_images_from_conversations(
                    [conversations],
                    batch_min_pixels=[config['image_sizes']['eval_min'] * 28 * 28],
                    batch_max_pixels=[config['image_sizes']['eval_max'] * 28 * 28]
                )
                
                all_conversations[uid] = processed_conv[0]
                all_images[uid] = images[0]
                all_raw_images[uid] = images[0]
                video_flags.append(is_video)
                
                # 处理视频帧
                if is_video and len(images[0]) > 8:
                    step = max(1, len(images[0]) // 8)
                    selected_frames = images[0][::step][:8]
                    all_images[uid] = [selected_frames]
                
            else:
                # 检查上一轮输出
                last_response = all_generated_texts[idx]
                
                require_tool, force_terminate = check_termination_conditions(
                    last_response,
                    num_tool_calls[idx],
                    len(all_images[uid]),
                    len(tokenizer.encode(last_response)),
                    config['max_turns'],
                    config['max_images']
                )
                
                if not require_tool or force_terminate:
                    continue
                
                # 执行工具调用
                tool_info = parse_tool_call(last_response)
                if not tool_info:
                    num_tool_fails[idx] += 1
                    continue
                
                added_images, message, error_flag = process_tool_result(
                    tool_info['name'],
                    tool_info['arguments'],
                    all_images[uid],
                    all_raw_images[uid],
                    video_flags[idx],
                    operations,
                    {
                        'select_min_pixels': config['image_sizes']['eval_min'] * 28 * 28,
                        'select_max_pixels': config['image_sizes']['select_max'] * 28 * 28,
                        'crop_min_pixels': config['image_sizes']['eval_min'] * 28 * 28,
                        'crop_max_pixels': config['image_sizes']['zoom_max'] * 28 * 28,
                    }
                )
                
                if error_flag:
                    num_tool_fails[idx] += 1
                
                # 更新对话
                assistant_msg = {
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': last_response}]
                }
                user_msg = create_tool_response_message(message, added_images)
                
                all_conversations[uid].extend([assistant_msg, user_msg])
                all_images[uid].extend(added_images)
                num_tool_calls[idx] += 1
            
            # 准备模型输入
            text_input = data_processor.processor.apply_chat_template(
                [all_conversations[uid]],
                tokenize=False,
                add_generation_prompt=True
            )[0]
            
            # 处理多模态输入
            if video_flags[idx]:
                visual_inputs = data_processor.processor(
                    text=[text_input],
                    videos=[all_images[uid][0]] if isinstance(all_images[uid][0], list) else [all_images[uid]],
                    images=all_images[uid][1:] if len(all_images[uid]) > 1 else None,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=20000
                )
            else:
                visual_inputs = data_processor.processor(
                    text=[text_input],
                    images=all_images[uid],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=20000
                )
            
            batch_inputs.append(visual_inputs['input_ids'])
            batch_visual_inputs.append({
                k: v for k, v in visual_inputs.items() 
                if k not in ['input_ids', 'attention_mask']
            })
            current_indices.append(idx)
        
        if not batch_inputs:
            break
        
        # 批量生成
        input_ids = torch.nn.utils.rnn.pad_sequence(
            batch_inputs, 
            batch_first=True, 
            padding_value=tokenizer.pad_token_id
        ).to(device)
        
        attention_mask = input_ids.ne(tokenizer.pad_token_id).to(device)
        
        # 合并visual inputs
        combined_visual_inputs = {}
        for key in batch_visual_inputs[0].keys():
            if all(key in vi for vi in batch_visual_inputs):
                combined_visual_inputs[key] = torch.cat(
                    [vi[key] for vi in batch_visual_inputs], 
                    dim=0
                ).to(device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **combined_visual_inputs,
                max_new_tokens=config['max_new_tokens'],
                temperature=config['temperature'] if turn == 0 else 0.9,
                top_p=config['top_p'],
                do_sample=not is_eval,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 解码输出
        generated_ids = outputs[:, input_ids.shape[1]:]
        generated_texts = tokenizer.batch_decode(
            generated_ids, 
            skip_special_tokens=False
        )
        
        # 更新结果
        for i, idx in enumerate(current_indices):
            if turn == 0:
                all_outputs.append(outputs[i])
                all_generated_texts.append(generated_texts[i])
            else:
                all_outputs[idx] = outputs[i]
                all_generated_texts[idx] = generated_texts[i]
        
        active_indices = current_indices
        turn += 1
    
    # 构建返回结果
    return build_samples_from_transformer_outputs(
        all_outputs, all_generated_texts, all_conversations, 
        all_images, expanded_qids, tokenizer, data_processor,
        strategy_args
    )

# ==================== 辅助函数 ====================

def resize_cropped(image: Image.Image, min_pixels: int, max_pixels: int) -> Image.Image:
    """调整裁剪后图像的大小"""
    w, h = image.size
    total_pixels = w * h
    
    if total_pixels < min_pixels:
        scale = np.sqrt(min_pixels / total_pixels)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    elif total_pixels > max_pixels:
        scale = np.sqrt(max_pixels / total_pixels)
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)
    
    return image

def build_samples_from_outputs(
    outputs: List[Any],
    conversations: Dict[str, List],
    images: Dict[str, List],
    qids: List[str],
    tokenizer: Any,
    data_processor: Any,
    args: Any
) -> List[Any]:
    """从输出构建Samples对象列表"""
    # 这里需要实现将输出转换为Samples对象的逻辑
    # 与原代码中的后处理部分类似
    pass

def build_samples_from_transformer_outputs(
    outputs: List[torch.Tensor],
    generated_texts: List[str],
    conversations: Dict[str, List],
    images: Dict[str, List],
    qids: List[str],
    tokenizer: Any,
    data_processor: Any,
    args: Any
) -> List[Any]:
    """从Transformer输出构建Samples对象列表"""
    # 类似于build_samples_from_outputs，但处理Transformer特定的输出格式
    pass