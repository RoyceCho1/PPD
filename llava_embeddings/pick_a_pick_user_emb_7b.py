from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import math

import sys
import warnings
warnings.filterwarnings("ignore")

import datasets
import os
import io
from absl import app, flags
from textwrap import dedent
import time
from collections import defaultdict
import tqdm
import pandas as pd
from transformers import BitsAndBytesConfig

FLAGS = flags.FLAGS
flags.DEFINE_integer('num_shots', 4, 'Number of shots to use for each user')
flags.DEFINE_string('output_dir', '/home/roycecho/Personalized-Text-To-Image-Diffusion/emb_data', 'Output directory to save the embeddings')
flags.DEFINE_string('pretrained', 'lmms-lab/llava-onevision-qwen2-7b-ov-chat', 'Pretrained model to use')
flags.DEFINE_string('model_name', "llava_qwen", 'Model name to use')
flags.DEFINE_string('device', 'cuda:0', 'Device to use for inference')
flags.DEFINE_string('device_map', 'none', 'Device map to use for inference')
flags.DEFINE_boolean('load_4bit', False, 'Load model in 4-bit quantization (requires bitsandbytes)')
flags.DEFINE_integer('num_chunks', 8, 'Number of chunks to split the data into')
flags.DEFINE_integer('which_chunk', 0, 'Which chunk to process')
flags.DEFINE_float('temperature', 0.7, 'Temperature to use for inference')
flags.DEFINE_integer('max_new_tokens', 1024, 'Max new tokens to generate')
flags.DEFINE_integer('save_every', 100, 'Save every n examples')
flags.DEFINE_integer('per_user', -1, 'Number of examples(number of chuncks) per user to generate') # 한 유저에서 나올 embedding

def get_image_from_bytes(byte_string):
    if isinstance(byte_string, list):
        byte_string = byte_string[0]
    image_stream = io.BytesIO(byte_string)
    try:
        image = Image.open(image_stream)
    except Exception as e:
        print("Error: ", e)
        raise e
    return image

def get_all_shots(user_ds, num_shots, per_user):
    user_df = user_ds.to_pandas()
    shuffled_user_df = user_df.sample(frac=1)
    
    user_df_chunks = [shuffled_user_df[i:i+num_shots] for i in range(0, len(shuffled_user_df), num_shots)]
    if len(user_df_chunks[-1]) < num_shots:
        user_df_chunks.pop()
        
    if per_user > 0:
        curr_num = len(user_df_chunks)
        per_user = min(per_user, curr_num)
        user_df_chunks = user_df_chunks[:per_user]
    
    all_shots = []
    for user_df_chunks in user_df_chunks:
        shot_images = []
        for i, row in user_df_chunks.iterrows():
            caption = row['caption']
            
            preferred_image_ud = row['best_image_uid']
            
            image_1 = get_image_from_bytes(row['jpg_0'])
            image_2 = get_image_from_bytes(row['jpg_1'])
            image_1_uid = row['image_0_uid']
            
            image_preferred = image_1 if image_1_uid == preferred_image_ud else image_2
            image_dispreferred = image_2 if image_1_uid == preferred_image_ud else image_1
            image_preferred_uid = image_1_uid if image_1_uid == preferred_image_ud else row['image_1_uid']
            image_dispreferred_uid = row['image_1_uid'] if image_1_uid == preferred_image_ud else image_1_uid
            
            shot_images.append((caption, image_preferred, image_dispreferred, image_preferred_uid, image_dispreferred_uid))
            # shot_images = [(caption, image_preferred, image_dispreferred, image_preferred_uid, image_dispreferred_uid)] * num_shots
        all_shots.append(shot_images)
    return all_shots
    
def process_hidden_states(hidden_states):
    last_hidden_states = hidden_states[-1] # last hidden state is usually the most informative
    tens = torch.cat(last_hidden_states, dim=1).squeeze() # 하나의 시퀀스로 합침
    lst = tens.detach().cpu().numpy().tolist()
    # detach() : gradient 계산에서 제외
    # cpu() : gpu 메모리에서 cpu 메모리로 이동
    # numpy() : numpy 배열로 변환
    # tolist() : 리스트로 변환
    return lst

def main(_):    
    splits = ['train', 'validation', 'test']
    output_dir = FLAGS.output_dir
    os.makedirs(output_dir, exist_ok=True)

    per_user = FLAGS.per_user
    num_shots = FLAGS.num_shots
    pretrained = FLAGS.pretrained
    model_name = FLAGS.model_name
    device = FLAGS.device
    device_map_flag = FLAGS.device_map
    llava_model_args = {"multimodal": True,} # language & vision model
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad" # image padding(사이즈를 맞추기 위해)
    llava_model_args["overwrite_config"] = overwrite_config
    
    # =====================================================
    # 모델 로딩: 3가지 모드
    # (1) --load_4bit        → 4-bit 양자화, 단일 GPU
    # (2) --device_map auto   → multi-GPU pipeline parallel
    # (3) --device_map none   → fp16 단일 GPU (기본)
    # =====================================================
    
    if FLAGS.load_4bit:
        # --- 수정: 4-bit 양자화 상태에서도 사용자가 입력한 device_map을 그대로 따릅니다 ---
        print("=" * 60)
        print(f"[MODE] 4-bit quantization with device_map={device_map_flag}")
        print("=" * 60)
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        if device_map_flag == 'none':
            device_index = int(device.split(":")[-1]) if ":" in device else 0
            d_map = {"": device_index}
        elif device_map_flag == 'auto' and torch.cuda.device_count() == 2:
            print("[HOTFIX] Applying Custom 2-GPU ASYMMETRIC Map to prevent Logit OOM!")
            # lm_head가 동작할 때 단번에 3.5GB 이상의 메모리가 스파이크되므로, 
            # GPU 1번(lm_head 위치)의 레이어 비중을 확 줄여서 메모리를 일부러 텅텅 비워둡니다.
            d_map = {}
            d_map['model.vision_tower'] = 0
            d_map['model.vision_resampler'] = 0
            d_map['model.mm_projector'] = 0
            d_map['model.embed_tokens'] = 0
            d_map['model.image_newline'] = 0
            # 밸런스 튜닝 2차: 18/10 에서는 GPU 0이 44MB 부족, 12/16 에서는 GPU 1이 700MB 부족.
            # 계산 결과 1레이어당 약 200MB 처리를 요구하므로 '17/11'이 수학적인 완벽한 스윗스팟입니다.
            for i in range(17): d_map[f'model.layers.{i}'] = 0
            for i in range(17, 28): d_map[f'model.layers.{i}'] = 1
            d_map['model.norm'] = 1
            d_map['lm_head'] = 1
        elif device_map_flag == 'auto' and torch.cuda.device_count() == 4:
            print("[HOTFIX] Applying Custom 4-GPU 4-BIT Map to completely eliminate OOM!")
            d_map = {}
            d_map['model.vision_tower'] = 0
            d_map['model.vision_resampler'] = 0
            d_map['model.mm_projector'] = 0
            d_map['model.embed_tokens'] = 0
            d_map['model.image_newline'] = 0
            for i in range(12): d_map[f'model.layers.{i}'] = 1
            for i in range(12, 24): d_map[f'model.layers.{i}'] = 2
            for i in range(24, 28): d_map[f'model.layers.{i}'] = 3
            d_map['model.norm'] = 3
            d_map['lm_head'] = 3
        else:
            d_map = device_map_flag
            
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name,
            device_map=d_map,
            attn_implementation= None,
            quantization_config=quantization_config,
            **llava_model_args
        )
        
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            first_module = list(model.hf_device_map.keys())[0]
            first_gpu = model.hf_device_map[first_module]
            input_device = f"cuda:{first_gpu}" if isinstance(first_gpu, int) else str(first_gpu)
        else:
            input_device = device
            
        # [HOTFIX] auto map일 경우 모델 이미지 줄바꿈 토큰 디바이스 맞춤
        if device_map_flag == 'auto':
            try:
                vision_device = next(model.model.vision_tower.parameters()).device
                if hasattr(model.model, 'image_newline'):
                    model.model.image_newline.data = model.model.image_newline.data.to(vision_device)
                    print(f"[HOTFIX] Moved model.model.image_newline to {vision_device} to match vision_tower")
            except Exception as e:
                pass

        
    elif device_map_flag == 'auto':
        # --- 접근법 B: Multi-GPU pipeline parallel ---
        print("=" * 60)
        print("[MODE] Multi-GPU pipeline parallel (device_map='auto')")
        print(f"[INFO] Available GPUs: {torch.cuda.device_count()}")
        print("=" * 60)
        
        num_gpus = torch.cuda.device_count()
        
        num_gpus = torch.cuda.device_count()
        
        # [HOTFIX] LLaVA의 load_pretrained_model은 device_map이 dictionary 형태면 내부 버그가 발생합니다.
        # 하지만 "auto"로 두면 OOM이 일어나므로, python 동적 패칭(Monkey Patching)으로 
        # transformers 라이브러리의 device_map 자동 생성 함수를 가로채어 완벽한 4-GPU 분산 맵을 강제로 주입합니다.
        import transformers
        original_infer = transformers.modeling_utils.infer_auto_device_map
        
        if device_map_flag == 'auto' and num_gpus == 4:
            def custom_infer_auto_device_map(*args, **kwargs):
                d_map = original_infer(*args, **kwargs)
                print("[HOTFIX] Intercepted Accelerate infer_auto_device_map. Enforcing safe 4-GPU layout!")
                
                d_map['model.embed_tokens'] = 0
                d_map['model.image_newline'] = 0  # 0번 GPU에 둬야 버그 안 남
                
                for i in range(10): d_map[f'model.layers.{i}'] = 0
                for i in range(10, 19): d_map[f'model.layers.{i}'] = 1
                for i in range(19, 28): d_map[f'model.layers.{i}'] = 2
                
                d_map['model.norm'] = 3
                d_map['model.vision_tower'] = 3
                d_map['model.vision_resampler'] = 3
                d_map['model.mm_projector'] = 3
                d_map['lm_head'] = 3
                
                # 디스크 및 CPU 오프로드 강제 제거
                clean_map = {k: v for k, v in d_map.items() if v not in ['cpu', 'disk']}
                
                for k in list(clean_map.keys()):
                    if 'model.vision_tower' in k:
                        clean_map[k] = 3
                        
                return clean_map
            
            transformers.modeling_utils.infer_auto_device_map = custom_infer_auto_device_map

        # max_memory 설정은 오히려 디스크 오프로딩을 유발하므로 삭제하고,
        # 오직 위의 완벽한 custom_infer_auto_device_map 에만 의존합니다.
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name,
            device_map="auto",
            attn_implementation=None,
            **llava_model_args
        )
        
        # 패치 원상복구
        if device_map_flag == 'auto' and num_gpus == 4:
            transformers.modeling_utils.infer_auto_device_map = original_infer
        
        # device_map 설정 시 model.to() 호출 금지
        # 입력 텐서의 device는 모델의 첫 번째 파라미터 위치로 결정
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            print("hf_device_map:", model.hf_device_map)
            # embedding 레이어가 있는 GPU를 input device로 사용
            first_module = list(model.hf_device_map.keys())[0]
            first_gpu = model.hf_device_map[first_module]
            input_device = f"cuda:{first_gpu}" if isinstance(first_gpu, int) else str(first_gpu)
        else:
            input_device = next(model.parameters()).device
            input_device = str(input_device)
        
        # [HOTFIX] model.image_newline has to be on the same device as vision_tower output
        try:
            vision_device = next(model.model.vision_tower.parameters()).device
            if hasattr(model.model, 'image_newline'):
                model.model.image_newline.data = model.model.image_newline.data.to(vision_device)
                print(f"[HOTFIX] Moved model.model.image_newline to {vision_device} to match vision_tower")
        except Exception as e:
            print(f"[HOTFIX] Could not move image_newline: {e}")
        
    else:
        # --- 접근법 C: 기본 fp16 단일 GPU ---
        print("=" * 60)
        print(f"[MODE] fp16 single GPU ({device})")
        print("=" * 60)
        
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            pretrained, None, model_name,
            device_map=None,
            attn_implementation=None,
            **llava_model_args
        )
        
        print("before to(device), first param device:", next(model.parameters()).device)
        model = model.to(device)
        print("after to(device), first param device:", next(model.parameters()).device)
        input_device = device
    
    # 공통 확인
    print(f"[INFO] input_device = {input_device}")
    print(f"[INFO] hf_device_map = {getattr(model, 'hf_device_map', None)}")
    try:
        print(f"[INFO] first param device = {next(model.parameters()).device}")
    except Exception as e:
        print(f"[INFO] first param device check failed: {e}")
    model.eval()

    
    ds = datasets.load_dataset('liuhuohuo2/pick-a-pic-v2')
    for split in splits:
        all_unique_users = ds[split].unique('user_id') # 중복 제거된 사용자 ID
        sorted_unique_users = sorted(list(all_unique_users)) # 정렬(오름차순)
        # BUG FIX: 나누어 떨어지지 않는 나머지가 버려지거나, 유저 수가 num_chunks보다 작을 때 0이 되는 것을 방지 (올림 처리)
        shard_size = math.ceil(len(sorted_unique_users) / FLAGS.num_chunks)
        start_idx = FLAGS.which_chunk * shard_size # default 0
        end_idx = start_idx + shard_size # default shard_size
        unique_users = sorted_unique_users[start_idx:end_idx]
        
        split_output_path = os.path.join(output_dir, f"{split}_shard{FLAGS.which_chunk}.json")
        split_data = defaultdict(list)
        total_examples = 0
        for user in tqdm.tqdm(unique_users, desc=f"Processing users in {split}"):
            print('User:', user)
            def filter_fn(examples):
                return_list = []
                for i in range(len(examples['user_id'])):
                    curr_user = examples['user_id'][i] 
                    is_different = examples['are_different'][i] # is the image different?
                    has_label = examples['has_label'][i] # is there a label?
                    return_list.append(curr_user == user and is_different and has_label)
                return return_list
            
            user_ds = ds[split].filter(filter_fn, batched=True, num_proc=1) # filter the dataset(using filter.fn)
            if len(user_ds) < num_shots:
                continue
            
            all_shots = get_all_shots(user_ds, num_shots, per_user)
            for shots in all_shots:
                assert len(shots) == num_shots, "Not enough shots"
                

            for shots in tqdm.tqdm(all_shots, desc=f"Processing shots for user {user}"):
                images = []
                for caption, image_preferred, image_dispreferred, _, _ in shots:
                    images.append(image_preferred)
                    images.append(image_dispreferred)
                image_tensors = process_images(images, image_processor, model.config) # process_images: image -> tensor
                image_tensors = [_image.to(dtype=torch.float16, device=input_device) for _image in image_tensors] # tensor -> float16, input_device
                
                question = "You will be shown a few examples of preferred and dispreferred images."
                for i in range(num_shots):
                    question += f" \n\nHere is the {i+1}st pair of images:"
                    question += f" \n\nHere is the caption: {shots[i][0]}"
                    question += f" \n\n{DEFAULT_IMAGE_TOKEN} This is the preferred image."
                    question += f" \n\n{DEFAULT_IMAGE_TOKEN} This is the dispreferred image."
                if FLAGS.num_shots == 2:
                    question += f""" \n\nYou will be judging the images according to their style, visual quality, and image aesthetics. Follow the instructions below to compare the images:
                        1. For each pair of images, describe the preferred image and the dispreferred image. 
                        2. Explain the differences between the two images in terms of style, visual quality, and image aesthetics.
                        3. After you have described all of the images, summarize the differences between the preferred and dispreferred images into a user profile. 
                        
                        Format your response as follows for the two pairs of images:
                        Pair 1:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 2:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        User Profile: [Description]
                    """
                elif FLAGS.num_shots == 3:
                    question += f""" \n\nYou will be judging the images according to their style, visual quality, and image aesthetics. Follow the instructions below to compare the images:
                        1. For each pair of images, describe the preferred image and the dispreferred image. 
                        2. Explain the differences between the two images in terms of style, visual quality, and image aesthetics.
                        3. After you have described all of the images, summarize the differences between the preferred and dispreferred images into a user profile. 
                        
                        Format your response as follows for the four pairs of images:
                        Pair 1:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 2:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 3:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        User Profile: [Description]
                    """
                elif FLAGS.num_shots == 4:
                    question += f""" \n\nYou will be judging the images according to their style, visual quality, and image aesthetics. Follow the instructions below to compare the images:
                        1. For each pair of images, describe the preferred image and the dispreferred image. 
                        2. Explain the differences between the two images in terms of style, visual quality, and image aesthetics.
                        3. After you have described all of the images, summarize the differences between the preferred and dispreferred images into a user profile. 
                        
                        Format your response as follows for the four pairs of images:
                        Pair 1:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 2:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 3:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 4:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        User Profile: [Description]
                    """
                elif FLAGS.num_shots == 5:
                    question += f""" \n\nYou will be judging the images according to their style, visual quality, and image aesthetics. Follow the instructions below to compare the images:
                        1. For each pair of images, describe the preferred image and the dispreferred image. 
                        2. Explain the differences between the two images in terms of style, visual quality, and image aesthetics.
                        3. After you have described all of the images, summarize the differences between the preferred and dispreferred images into a user profile. 
                        
                        Format your response as follows for the four pairs of images:
                        Pair 1:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 2:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 3:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 4:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        Pair 5:
                        Preferred Image: [Description]
                        Dispreferred Image: [Description]
                        Differences: [Description]
                        
                        User Profile: [Description]
                    """
                else:
                    raise ValueError("Invalid number of shots")
                
                question = dedent(question).strip() # remove leading/trailing whitespace
                
                # Prepare interleaved text-image input
                conv_template = "qwen_1_5"
                
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question) # question
                conv.append_message(conv.roles[1], None) # answer
                prompt_question = conv.get_prompt()
                
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(input_device) # tokenizer_image_token: text + image -> input_ids
                image_sizes = [image.size for image in images] # image.size: (width, height)(혹시 필요로 할때 확인하기 위해 저장)
                
                print('Starting inference...')
                curr_time = time.time()
                # Generate response
                gen_dict = model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False, # 램던성 끄기 -> 더 일관적인 결과
                    temperature=FLAGS.temperature, # 낮을수록 일관적
                    max_new_tokens=FLAGS.max_new_tokens,
                    return_dict_in_generate=True, # 결과값을 단순한 문자열이 아니라, 자세한 정보가 담긴 딕셔너리로 반환
                    output_hidden_states=True, # hidden states 반환 -> 임베딩 추출에 사용
                )
                print('Inference time:', time.time() - curr_time)

                text_output = tokenizer.batch_decode(gen_dict['sequences'], skip_special_tokens=True)[0] #decode answer to text format
                emb = process_hidden_states(gen_dict['hidden_states']) # extract hidden states
                
                split_data['user_id'].append(user) # user id 저장
                split_data['text'].append(text_output) # text 저장(output text)
                split_data['emb'].append(emb) # embedding 저장
                for i in range(len(shots)):
                    caption, _, _, image_preferred_uid, image_dispreferred_uid = shots[i]
                    
                    split_data[f'preferred_image_uid_{i}'].append(image_preferred_uid)
                    split_data[f'dispreferred_image_uid_{i}'].append(image_dispreferred_uid)
                    split_data[f'caption_{i}'].append(caption)

                if total_examples % FLAGS.save_every == 0:
                    df = pd.DataFrame(split_data)
                    df.to_json(split_output_path)
                total_examples += 1
                
                # ------ [HOTFIX] 메모리 반환 ------
                del gen_dict
                del input_ids
                del image_tensors
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
            # Save for every user
            df = pd.DataFrame(split_data)
            df.to_json(split_output_path)
        # Save for every split
        df = pd.DataFrame(split_data) # pandas DataFrame으로 변환
        df.to_json(split_output_path) # json 파일로 저장


if __name__ == '__main__':
    app.run(main)