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
import openai
import base64
from pprint import pprint

SYSTEM_PROMPT = "You are an expert in image aesthetics and have been asked to predict which image a user would prefer based on the examples provided."

FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', "gpt-4o-mini", 'Model name to use')
flags.DEFINE_integer('max_new_tokens', 2048, 'Max new tokens to generate')
flags.DEFINE_bool('include_cot', False, 'Include COT in the prompt')
flags.DEFINE_integer('num_chunks', 4, 'Number of chunks to split the data into')
flags.DEFINE_integer('which_chunk', 0, 'Which chunk to process')

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

def pil_img_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def text_to_contents(text):
    return {'type': 'text', 'text': text}

def img_to_contents(img):
    pil_img = get_image_from_bytes(img)
    base64_image = pil_img_to_base64(pil_img)
    return {
        "type": "image_url", 
        "image_url": {
            "url":  f"data:image/jpeg;base64,{base64_image}"
        },
    }

def process_row(row, include_cot=False):
    content_pref_first, content_pref_second = [], []

    question = "You will be shown a pair of images and a user description. Your task is to predict which image the user would prefer based the user description."
    content_pref_first.append(text_to_contents(question))
    content_pref_second.append(text_to_contents(question))

    if include_cot:
        suffix = dedent("""
            1. Describe each image in terms of style, visual quality, and image aesthetics.
            2. Explain the differences between the two images in terms of style, visual quality, and image aesthetics.
            3. Predict which image the user would prefer based on the caption and the user description.
            
            Format your response as follows:
            Image 1: [Description]
            Image 2: [Description]
            Differences: [Description]
            Prediction of user preference: [1 or 2]
        """).strip()
    else:
        suffix = dedent("""
            Which image would the user prefer?
            
            Format your response as follows:
            Prediction of user preference: [1 or 2]
        """).strip()
    content_pref_first.append(text_to_contents(suffix))
    content_pref_second.append(text_to_contents(suffix))
    
    last_pref = row['jpg_model_train']
    last_dispref = row['jpg_model_base']
    content_pref_first.append(text_to_contents(f"Here is Pair:"))
    content_pref_first.append(img_to_contents(last_pref))
    content_pref_first.append(img_to_contents(last_dispref))
    
    content_pref_second.append(text_to_contents(f"Here is Pair:"))
    content_pref_second.append(img_to_contents(last_dispref))
    content_pref_second.append(img_to_contents(last_pref))
    
    pref_first = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT
                }
            ]
        },
        {
            "role": "user",
            "content": content_pref_first
        }
    ]
    
    pref_second = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ]
        },
        {
            "role": "user",
            "content": content_pref_second
        }
    ]   
    return pref_first, pref_second

def validate_messages(messages):
    for message in messages:
        for content in message['content']:
            if content['type'] == 'image_url':
                assert 'image_url' in content and 'text' not in content, f"Invalid image_url content: {content}"
            elif content['type'] == 'text':
                assert 'text' in content and 'image_url' not in content, f"Invalid text content: {content}"
            else:
                raise ValueError(f"Invalid content type: {content}")
    return True

def print_messages(messages):
    for message in messages:
        for content in message['content']:
            if content['type'] == 'image_url':
                print(f"Image URL")
            elif content['type'] == 'text':
                print(f"Text: {content['text']}")
            else:
                raise ValueError(f"Invalid content type: {content}")
    return True

def get_prediction(text_output):
    if '1' in text_output:
        return 1
    elif '2' in text_output:
        return 2
    else:
        return None

def main(_):
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    ds = datasets.load_dataset('TODO: add dataset name')
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
        
        all_scores = []
        all_text_pref_first = []
        all_text_pref_second = []
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing {}".format(split)):
            messages_pref_first, messages_pref_second = process_row(row, include_cot=FLAGS.include_cot)
            
            validate_messages(messages_pref_first)
            validate_messages(messages_pref_second)
            
            curr_time = time.time()
            text_output_pref_first = client.chat.completions.create(
                model=FLAGS.model_name,
                messages=messages_pref_first,
                max_tokens=FLAGS.max_new_tokens,
            ).choices[0].message.content
            try:
                which_image_pref_first = text_output_pref_first.split("Prediction of user preference: ")[-1].strip()
                which_image_pref_first = get_prediction(which_image_pref_second)
            except:
                which_image_pref_first = None
            
            text_output_pref_second = client.chat.completions.create(
                model=FLAGS.model_name,
                messages=messages_pref_second,
                max_tokens=FLAGS.max_new_tokens,
            ).choices[0].message.content
            
            try:
                which_image_pref_second = text_output_pref_second.split("Prediction of user preference: ")[-1].strip()
                which_image_pref_second = get_prediction(which_image_pref_second)
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
            
        df['score'] = all_scores
        df['text_pref_first'] = all_text_pref_first
        df['text_pref_second'] = all_text_pref_second
        all_splits[split] = datasets.Dataset.from_pandas(df)
        
        all_splits = datasets.DatasetDict(all_splits)
        all_splits.push_to_hub('TODO: add dataset name')

if __name__ == '__main__':
    app.run(main)
