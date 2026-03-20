import sys

file_path = "/Users/roycecho/Desktop/Computer Vision/02_demo_code/PPD/llava_embeddings/pick_a_pick_user_emb_7b.py"

with open(file_path, "r") as f:
    lines = f.readlines()

loop_start_idx = -1
for i, line in enumerate(lines):
    if "ds = datasets.load_dataset('liuhuohuo2/pick-a-pic-v2')" in line:
        loop_start_idx = i
        break

model_eval_idx = -1
for i, line in enumerate(lines[loop_start_idx:]):
    if "        model.eval()" in line:
        model_eval_idx = loop_start_idx + i
        break

pre_loop = lines[:loop_start_idx] 
dataset_loading = lines[loop_start_idx:loop_start_idx+9] 
model_loading_indented = lines[loop_start_idx+9:model_eval_idx+1]
post_loop = lines[model_eval_idx+1:] 

model_loading_unindented = []
for line in model_loading_indented:
    if line.startswith("    "):
        model_loading_unindented.append(line[4:])
    else:
        model_loading_unindented.append(line)

output_path_line = ""
new_model_loading = []
for line in model_loading_unindented:
    if "split_output_path =" in line:
        output_path_line = "    " + line 
    else:
        new_model_loading.append(line)

new_lines = pre_loop + new_model_loading + dataset_loading + [output_path_line] + post_loop

with open(file_path, "w") as f:
    f.writelines(new_lines)

print("Refactoring complete.")
