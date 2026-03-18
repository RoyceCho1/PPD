from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch

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
flags.DEFINE_string('device', 'cuda', 'Device to use for inference')
flags.DEFINE_string('device_map', 'auto', 'Device map to use for inference')
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

    ds = datasets.load_dataset('liuhuohuo2/pick-a-pic-v2')
    for split in splits:
        all_unique_users = ds[split].unique('user_id') # 중복 제거된 사용자 ID
        sorted_unique_users = sorted(list(all_unique_users)) # 정렬(오름차순)
        shard_size = len(sorted_unique_users) // FLAGS.num_chunks 
        start_idx = FLAGS.which_chunk * shard_size # default 0
        end_idx = start_idx + shard_size # default shard_size
        unique_users = sorted_unique_users[start_idx:end_idx]
        
        per_user = FLAGS.per_user
        num_shots = FLAGS.num_shots
        pretrained = FLAGS.pretrained
        model_name = FLAGS.model_name
        device = FLAGS.device
        device_map = FLAGS.device_map
        llava_model_args = {"multimodal": True,} # language & vision model
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad" # image padding(사이즈를 맞추기 위해)
        llava_model_args["overwrite_config"] = overwrite_config
        
        # 4-bit 양자화 설정(VRAM 용량) (load_4bit=True는 transformers 4.45+에서 충돌)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None, quantization_config=quantization_config, **llava_model_args)
        split_output_path = os.path.join(output_dir, f"{split}_shard{FLAGS.which_chunk}.json")
        
        # tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None, **llava_model_args)
        
        model.eval()
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
            
            user_ds = ds[split].filter(filter_fn, batched=True, num_proc=os.cpu_count()) # filter the dataset(using filter.fn)
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
                image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors] # tensor -> float16, device
                
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
                
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device) # tokenizer_image_token: text + image -> input_ids
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
            # Save for every user
            df = pd.DataFrame(split_data)
            df.to_json(split_output_path)
        # Save for every split
        df = pd.DataFrame(split_data) # pandas DataFrame으로 변환
        df.to_json(split_output_path) # json 파일로 저장


if __name__ == '__main__':
    app.run(main)