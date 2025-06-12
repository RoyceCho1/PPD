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
import numpy as np

FLAGS = flags.FLAGS
flags.DEFINE_string('pretrained', 'lmms-lab/llava-onevision-qwen2-7b-ov-chat', 'Pretrained model to use')
flags.DEFINE_string('model_name', "llava_qwen", 'Model name to use')
flags.DEFINE_string('device', 'cuda', 'Device to use for inference')
flags.DEFINE_string('device_map', 'auto', 'Device map to use for inference')
flags.DEFINE_float('temperature', 0.7, 'Temperature to use for inference')
flags.DEFINE_integer('max_new_tokens', 2048, 'Max new tokens to generate')
flags.DEFINE_bool('include_cot', False, 'Include COT in the prompt')
flags.DEFINE_integer('num_chunks', 4, 'Number of chunks to split the data into')
flags.DEFINE_integer('which_chunk', 0, 'Which chunk to process')
flags.DEFINE_bool('randomize_fewshot', False, 'Randomize fewshot examples')

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

def process_row(row, include_cot=False, randomize_fewshot=False):
    pref_first, pref_second = [], []
    question = "You will be shown a few examples of preferred and dispreferred images that a user has labeled."
    for i, (caption, pref, dispref) in enumerate(zip(row['caption'], row['preferred_image'], row['dispreferred_image'])):
        if i < len(row['caption']) - 1:
            if randomize_fewshot:
                question += f"\n\nHere is the {i+1}st pair of images:"
                question += f"\n\nHere is the caption: {caption}"
                question += f"\n\nImage 1: {DEFAULT_IMAGE_TOKEN}"
                question += f"\n\nImage 2: {DEFAULT_IMAGE_TOKEN}"
                if np.random.rand() > 0.5:
                    question += f"Prediction of user preference: 1"
                    pref_first.extend([pref, dispref])
                    pref_second.extend([pref, dispref])
                else:
                    question += f"Prediction of user preference: 2"
                    pref_first.extend([dispref, pref])
                    pref_second.extend([dispref, pref])
            else:
                question += f"\n\nHere is the {i+1}st pair of images:"
                question += f"\n\nHere is the caption: {caption}"
                question += f"\n\n{DEFAULT_IMAGE_TOKEN} This is the preferred image."
                question += f"\n\n{DEFAULT_IMAGE_TOKEN} This is the dispreferred image."
                
                pref_first.extend([pref, dispref])
                pref_second.extend([pref, dispref])
        else:
            pref_first.extend([pref, dispref])
            pref_second.extend([dispref, pref])

    if include_cot:
        suffix = dedent("""
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
            
            Finally, you will be provided with a new pair of images, unlabeled by the user. Your task is to predict which image the user would prefer based on the user profile you have created and the previous examples you have seen.
            
            Pair 5:
            Image 1: {DEFAULT_IMAGE_TOKEN}
            Image 2: {DEFAULT_IMAGE_TOKEN}
            
            Which image would the user prefer?
            
            Format your response as follows:
            Prediction of user preference: [1 or 2]
        """).strip()
    else:
        suffix = dedent("""
            You will be provided with a new pair of images, unlabeled by the user. Your task is to predict which image the user would prefer based on the previous examples you have seen.
            
            Pair 5:
            Image 1: {DEFAULT_IMAGE_TOKEN}
            Image 2: {DEFAULT_IMAGE_TOKEN}
            
            Which image would the user prefer?
            
            Format your response as follows:
            Prediction of user preference: [1 or 2]
        """).strip()
    
    question = dedent(question).strip() + "\n\n" + suffix
    return question, pref_first, pref_second

def main(_):
    ds = datasets.load_dataset('Asap7772/pickapic_user_shots')
    all_splits = {}
    for split in ds.keys():
        df = ds[split].to_pandas()
        all_unique_user_ids = df['user_id'].unique()
        sorted_user_ids = sorted(all_unique_user_ids)
        shard_size = len(sorted_user_ids) // FLAGS.num_chunks
        start_idx = FLAGS.which_chunk * shard_size
        end_idx = (FLAGS.which_chunk + 1) * shard_size
        user_ids = sorted_user_ids[start_idx:end_idx]
        df = df[df['user_id'].isin(user_ids)]
        
        pretrained = FLAGS.pretrained
        model_name = FLAGS.model_name
        device = FLAGS.device
        device_map = FLAGS.device_map
        
        llava_model_args = {"multimodal": True,}
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        llava_model_args["overwrite_config"] = overwrite_config
        tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None, **llava_model_args)
        
        all_scores = []
        all_text_pref_first = []
        all_text_pref_second = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing {}".format(split)):
            try:
                question, pref_first, pref_second = process_row(row, include_cot=FLAGS.include_cot, randomize_fewshot=FLAGS.randomize_fewshot)
                
                pref_first = [get_image_from_bytes(image) for image in pref_first]
                pref_first_tensors = process_images(pref_first, image_processor, model.config)
                pref_first_tensors = [_image.to(dtype=torch.float16, device=device) for _image in pref_first_tensors]
                
                pref_second = [get_image_from_bytes(image) for image in pref_second]
                pref_second_tensors = process_images(pref_second, image_processor, model.config)
                pref_second_tensors = [_image.to(dtype=torch.float16, device=device) for _image in pref_second_tensors]
                
                conv_template = "qwen_1_5"
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
                pref_first_image_sizes = [image.size for image in pref_first_tensors]
                pref_second_image_sizes = [image.size for image in pref_second_tensors]
                
                curr_time = time.time()
                gen_dict_pref_first = model.generate(
                    input_ids,
                    images=pref_first_tensors,
                    image_sizes=pref_first_image_sizes,
                    do_sample=False,
                    temperature=FLAGS.temperature,
                    max_new_tokens=FLAGS.max_new_tokens,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                )
                text_output_pref_first = tokenizer.batch_decode(gen_dict_pref_first['sequences'], skip_special_tokens=True)[0]
                
                try:
                    which_image_pref_first = text_output_pref_first.split("Prediction of user preference: ")[-1].strip()
                    which_image_pref_first = int(which_image_pref_first)
                except:
                    which_image_pref_first = None
                
                gen_dict_pref_second = model.generate(
                    input_ids,
                    images=pref_second_tensors,
                    image_sizes=pref_second_image_sizes,
                    do_sample=False,
                    temperature=FLAGS.temperature,
                    max_new_tokens=FLAGS.max_new_tokens,
                    return_dict_in_generate=True,
                    output_hidden_states=False,
                )
                text_output_pref_second = tokenizer.batch_decode(gen_dict_pref_second['sequences'], skip_special_tokens=True)[0]
                try:
                    which_image_pref_second = text_output_pref_second.split("Prediction of user preference: ")[-1].strip()
                    which_image_pref_second = int(which_image_pref_second)
                except:
                    which_image_pref_second = None
                    
                if which_image_pref_first is not None and which_image_pref_second is not None:
                    if which_image_pref_first == 1 and which_image_pref_second == 2:
                        # correct prediction
                        score = 1
                    elif which_image_pref_first == 2 and which_image_pref_second == 1:
                        # incorrect prediction
                        score = 0
                    else:
                        # inconsistent prediction
                        score = -1
                else:
                    # invalid prediction
                    score = -2
                all_scores.append(score)
                all_text_pref_first.append(text_output_pref_first)
                all_text_pref_second.append(text_output_pref_second)
                        
                print('Inference time:', time.time() - curr_time)
            except Exception as e:
                continue
            
        df['score'] = all_scores
        df['text_pref_first'] = all_text_pref_first
        df['text_pref_second'] = all_text_pref_second
        all_splits[split] = datasets.Dataset.from_pandas(df)
        
        all_splits = datasets.DatasetDict(all_splits)
        all_splits.push_to_hub('Asap7772/pickapic_user_shots_winrate_chunk{}_cot{}_randomize{}'.format(FLAGS.which_chunk, FLAGS.include_cot, FLAGS.randomize_fewshot))

if __name__ == '__main__':
    app.run(main)
