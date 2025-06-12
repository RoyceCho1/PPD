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
import concurrent.futures

SYSTEM_PROMPT = "You are an expert in image aesthetics and have been asked to predict which image a user would prefer based on the examples provided."

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', 'TODO: add dataset name', 'Dataset name to load')
flags.DEFINE_string('model_name', "gpt-4o-mini", 'Model name to use')
flags.DEFINE_integer('max_new_tokens', 2048, 'Max new tokens to generate')
flags.DEFINE_bool('include_cot', False, 'Include COT in the prompt')
flags.DEFINE_integer('num_chunks', 4, 'Number of chunks to split the data into')
flags.DEFINE_integer('which_chunk', 0, 'Which chunk to process')
flags.DEFINE_bool('randomize_fewshot', False, 'Randomize fewshot examples')
flags.DEFINE_bool('no_caption', False, 'Do not include caption in the prompt')

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

def process_row(row, include_cot=False, randomize_fewshot=False, no_caption=False):
    content_pref_first, content_pref_second = [], []

    question = "You will be shown a few examples of preferred and dispreferred images that a user has labeled."
    content_pref_first.append(text_to_contents(question))
    content_pref_second.append(text_to_contents(question))
    
    for i, (caption, pref, dispref) in enumerate(zip(row['caption'], row['preferred_image'], row['dispreferred_image'])):
        if i < len(row['caption']) - 1:
            if randomize_fewshot:
                content_pref_first.append(text_to_contents(f"Here is Pair {i+1}:"))
                content_pref_second.append(text_to_contents(f"Here is Pair {i+1}:"))
                if not no_caption:
                    content_pref_first.append(text_to_contents(f"Here is the caption: {caption}"))
                    content_pref_second.append(text_to_contents(f"Here is the caption: {caption}"))
                
                if np.random.rand() > 0.5:
                    content_pref_first.append(img_to_contents(pref))
                    content_pref_first.append(img_to_contents(dispref))
                    content_pref_first.append(text_to_contents('Prediction of user preference: 1'))
                    
                    content_pref_second.append(img_to_contents(pref))
                    content_pref_second.append(img_to_contents(dispref))
                    content_pref_second.append(text_to_contents('Prediction of user preference: 1'))
                else:
                    content_pref_first.append(img_to_contents(dispref))
                    content_pref_first.append(img_to_contents(pref))
                    content_pref_first.append(text_to_contents('Prediction of user preference: 2'))
                    
                    content_pref_second.append(img_to_contents(dispref))
                    content_pref_second.append(img_to_contents(pref))
                    content_pref_second.append(text_to_contents('Prediction of user preference: 2'))
            else:
                raise NotImplementedError("Fixed fewshot is not implemented yet.")

    if include_cot:
        suffix = dedent("""
            1. Describe each image in terms of style, visual quality, and image aesthetics.
            2. Explain the differences between the two images in terms of style, visual quality, and image aesthetics.
            3. After you have described all of the images, summarize the differences between the preferred and dispreferred images into a user profile. 
            
            Format your response as follows for the four pairs of images:
            Pair 1:
            Image 1: [Description]
            Image 2: [Description]
            Differences: [Description]
            
            Pair 2:
            Image 1: [Description]
            Image 2: [Description]
            Differences: [Description]
            
            Pair 3:
            Image 1: [Description]
            Image 2: [Description]
            Differences: [Description]
            
            Pair 4:
            Image 1: [Description]
            Image 2: [Description]
            Differences: [Description]
            
            User Profile: [Description]
            
            Finally, You are provided with a new pair of images, unlabeled by the user. Your task is to predict which image the user would prefer based on the previous examples you have seen.
            
            Which image would the user prefer?
            
            Format your response as follows:
            Prediction of user preference: [1 or 2]
        """).strip()
    else:
        suffix = dedent("""
            You are provided with a new pair of images, unlabeled by the user. Your task is to predict which image the user would prefer based on the previous examples you have seen.
            
            Which image would the user prefer?
            
            Format your response as follows:
            Prediction of user preference: [1 or 2]
        """).strip()
    content_pref_first.append(text_to_contents(suffix))
    content_pref_second.append(text_to_contents(suffix))
    
    last_caption = row['caption'][-1]
    last_pref = row['preferred_image'][-1]
    last_dispref = row['dispreferred_image'][-1]
    
    content_pref_first.append(text_to_contents(f"Here is Pair {i+1}:"))
    content_pref_second.append(text_to_contents(f"Here is Pair {i+1}:"))
    
    if not no_caption:
        content_pref_first.append(text_to_contents(f"Here is the caption: {last_caption}"))
        content_pref_second.append(text_to_contents(f"Here is the caption: {caption}"))
    
    content_pref_first.append(img_to_contents(last_pref))
    content_pref_first.append(img_to_contents(last_dispref))
    
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

def main(_):
    client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'])

    ds = datasets.load_dataset(FLAGS.dataset_name)
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
        
        def process_single_row(row):
            try:
                messages_pref_first, messages_pref_second = process_row(row, include_cot=FLAGS.include_cot, randomize_fewshot=FLAGS.randomize_fewshot, no_caption=FLAGS.no_caption)
                
                validate_messages(messages_pref_first)
                validate_messages(messages_pref_second)

                text_output_pref_first = client.chat.completions.create(
                    model=FLAGS.model_name,
                    messages=messages_pref_first,
                    max_tokens=FLAGS.max_new_tokens,
                ).choices[0].message.content

                text_output_pref_second = client.chat.completions.create(
                    model=FLAGS.model_name,
                    messages=messages_pref_second,
                    max_tokens=FLAGS.max_new_tokens,
                ).choices[0].message.content

                which_image_pref_first = int(text_output_pref_first.split("Prediction of user preference: ")[-1].strip())
                which_image_pref_second = int(text_output_pref_second.split("Prediction of user preference: ")[-1].strip())

                if which_image_pref_first == 1 and which_image_pref_second == 2:
                    score = 1
                elif which_image_pref_first == 2 and which_image_pref_second == 1:
                    score = 0
                else:
                    score = -1

                return score, text_output_pref_first, text_output_pref_second

            except Exception as e:
                return -3, 'Inference failed', 'Inference failed'
        
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm.tqdm(executor.map(process_single_row, [row for _, row in df.iterrows()]), total=len(df), desc="Processing {}".format(split)))

        for score, text_pref_first, text_pref_second in results:
            all_scores.append(score)
            all_text_pref_first.append(text_pref_first)
            all_text_pref_second.append(text_pref_second)
            
        df['score'] = all_scores
        df['text_pref_first'] = all_text_pref_first
        df['text_pref_second'] = all_text_pref_second
        all_splits[split] = datasets.Dataset.from_pandas(df)
        
        all_splits = datasets.DatasetDict(all_splits)
        all_splits.push_to_hub('TODO: add dataset name')

if __name__ == '__main__':
    app.run(main)
