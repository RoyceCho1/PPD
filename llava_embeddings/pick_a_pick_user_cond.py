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

split = 'train'

ds = datasets.load_dataset('yuvalkirstain/pickapic_v1', split=split)
unique_users = ds[split].unique('user_id')

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

def get_shots(user_ds, num_shots):
    shot_images = []
    user_df = user_ds.to_pandas()
    shuffled_user_df = user_df.sample(frac=1)
    
    for i, row in shuffled_user_df.iterrows():
        caption = row['caption']
        
        preferred_image_ud = row['best_image_uid']
        
        image_1 = get_image_from_bytes(row['jpg_0'])
        image_1_uid = row['image_0_uid']
        
        image_2 = get_image_from_bytes(row['jpg_1'])
        image_2_uid = row['image_1_uid']
        
        image_preferred = image_1 if image_1_uid == preferred_image_ud else image_2
        image_dispreferred = image_2 if image_1_uid == preferred_image_ud else image_1
        
        shot_images.append((caption, image_preferred, image_dispreferred))
        
        # Break if we have enough shots
        if len(shot_images) == num_shots:
            break
    return shot_images

def main(_):
    num_shots = 4
    # Load model
    output_dir = '/home/anikait.singh/personalized-t2i/outputs_emb'
    
    pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
    # pretrained = 'lmms-lab/llava-onevision-qwen2-7b-ov-chat'
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    llava_model_args = {
        "multimodal": True,
    }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation=None, **llava_model_args)

    model.eval()
    
    for user in unique_users:
        print('User:', user)
        def filter_fn(examples):
            return_list = []
            for i in range(len(examples['user_id'])):
                curr_user = examples['user_id'][i]
                is_different = examples['are_different'][i]
                has_label = examples['has_label'][i]
                return_list.append(curr_user == user and is_different and has_label)
            return return_list
        
        user_ds = ds[split].filter(filter_fn, batched=True, num_proc=os.cpu_count())
        if len(user_ds) < num_shots:
            continue
        
        shots = get_shots(user_ds, num_shots)
        assert len(shots) == num_shots, "Not enough shots"

        images = []
        for caption, image_preferred, image_dispreferred in shots:
            images.append(image_preferred)
            images.append(image_dispreferred)
        image_tensors = process_images(images, image_processor, model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
        
        question = "You will be shown a few examples of preferred and dispreferred images."
        for i in range(num_shots):
            question += f"\n\nHere is the {i+1}st pair of images:"
            question += f"\n\nHere is the caption: {shots[i][0]}"
            question += f"\n\n{DEFAULT_IMAGE_TOKEN} This is the preferred image."
            question += f"\n\n{DEFAULT_IMAGE_TOKEN} This is the dispreferred image."
        question += f"""
            You will be judging the images according to their style, visual quality, and image aesthetics. Follow the instructions below to compare the images:
            1. For each pair of images, describe the preferred image and the dispreferred image. 
            2. Explain the differences between the two images in terms of style, visual quality, and image aesthetics.
            4. After you have described all of the images, summarize the differences between the preferred and dispreferred images into a user profile. 
            
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
        question = dedent(question).strip()
        
        # Prepare interleaved text-image input
        conv_template = "qwen_1_5"
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
        image_sizes = [image.size for image in images]
        
        # Generate response
        cont = model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0.7,
            max_new_tokens=8092,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(text_outputs[0])
        
        # Save the output
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{user}.txt'), 'w') as f:
            f.write(text_outputs[0])
        for i in range(num_shots):
            shots[i][1].save(os.path.join(output_dir, f'{user}_{i}_preferred.jpg'))
            shots[i][2].save(os.path.join(output_dir, f'{user}_{i}_dispreferred.jpg'))
            with open(os.path.join(output_dir, f'{user}_{i}_caption.txt'), 'w') as f:
                f.write(shots[i][0])

if __name__ == '__main__':
    app.run(main)