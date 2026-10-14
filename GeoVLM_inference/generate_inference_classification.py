#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#
import os
import json
import time
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
from torch.utils.data._utils.collate import default_collate
import ast

# ------------------------------ Helper Functions ------------------------------
def tensor_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list):
        return [tensor_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: tensor_to_list(v) for k, v in obj.items()}
    return obj

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        return [json.loads(line.strip()) for line in f]

def write_jsonl(data, file_path):
    with open(file_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

# ------------------------------ Dataset ------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]
            image_field = row.get("image", "")
            if image_field == "":
                raise ValueError(f"image field is not found at index {idx}")

            if isinstance(image_field, str):
                try:
                    import ast
                    image_field = ast.literal_eval(image_field)
                except Exception:
                    image_field = [image_field]
            elif isinstance(image_field, (tuple, list)):
                image_field = list(image_field)
            else:
                image_field = [str(image_field)]

            result = {
                "image_path": image_field,
                "question": row.get("question", ""),
                "answer": row.get("ground_truth", ""),
                "question_id": row.get("question_id", ""),
                "dataset": row.get("dataset", ""),
                "type": row.get("type", ""),
                "id": row.get("id", ""),
                "obj_ids": row.get("obj_ids", ""),
                "size_group": row.get("size_group", ""),
            }
            #print(f"Returning item for idx {idx}: {result}")
            return result

        except Exception as e:
            print(f"[Error in __getitem__ at idx {idx}]: {e}")
            raise


            
def safe_collate(batch):
    # Print types of batch items to debug
    #print(f"Batch size: {len(batch)}; Types: {[type(item) for item in batch]}")
    # Just collate normally, to see the exact cause of failure
    return default_collate(batch)


# ------------------------------ Evaluation ------------------------------
def evaluate(model, dataloader, processor, device):
    results = []
    model.eval()

    for batch in tqdm(dataloader, desc="Start Evaluating"):
        # Check if any value is None or contains None
        for k, v in batch.items():
            if v is None:
                print(f"None is found in batch for the key {k}")
            
        try:
            start_time_batch = time.time()
            image_paths = batch["image_path"]
            # Handle batch wrapped in lists
            if isinstance(image_paths, list):
                # Handle list of tuples like [('path1',), ('path2',)]
                image_paths = [p[0] if isinstance(p, tuple) else p for p in image_paths]
            elif isinstance(image_paths, str):
                image_paths = eval(image_paths)
            if isinstance(image_paths, str):
                image_paths = eval(image_paths)  # list of paths

            # Build message format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": p, "resized_height": 448, "resized_width": 448}
                    for p in image_paths
                ] + [{"type": "text", "text": batch["question"]}]
            }]

            text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text_input], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt").to(model.device)

            generated_ids = model.generate(**inputs, max_new_tokens=128)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            prediction = processor.batch_decode(trimmed, skip_special_tokens=True)
            print(f'Prediction is: {prediction}')
            print(f'Groundtruth is: {batch["answer"]}')
            print('**************************************************************************')
            results.append({
                "question_id": batch["question_id"],
                "response": prediction,
                "ground_truth": batch["answer"],
                "No_of_images": len(image_paths),
                "total_time_sec": round(time.time() - start_time_batch, 3),
                "question": batch["question"], 
                "type": batch["type"], 
                "dataset": batch["dataset"],
                "obj_ids": batch["obj_ids"],
                "size_group": batch["size_group"], 
                "id": batch["id"],             
                "image_path": image_paths,
            })

        except Exception as e:
            print(f"[Error] Batch failed for the images: {batch['image_path']} | Error: {str(e)}")

    return results

# ------------------------------ Folder Evaluation ------------------------------
def evaluate_folder(model, processor, folder_path, json_name, output_folder, device):
    qa_file = os.path.join(folder_path, json_name)
    if not os.path.exists(qa_file):
        print(f"[Warning] Missing file is : {qa_file}")
        return
    
    df = pd.DataFrame(read_jsonl(qa_file))  
    dataset = MultimodalDataset(df)  
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=safe_collate)
    result = evaluate(model, dataloader, processor, device)

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, json_name)
    write_jsonl(tensor_to_list(result), save_path)
    print(f"[Output json File is save at] {save_path}")


# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GeoVLM-R1 on EarthDail datasets")
    parser.add_argument("--test_data_directory", default = "./GeoVLM_Bench/", help="The dataset jsonl file paths")
    parser.add_argument("--output_jsonls_directory", default = "./GeoVLM_VLM_R1_Results/", help="The outputs jsonl file paths")
    parser.add_argument("--model_path", default = "./GeoVLM_R1", help="Path to pretrained GeoVLM-R1 model")
    parser.add_argument("--auto", action="store_true", help="Use automatic device mapping")
    args = parser.parse_args()
    
    print(f'results will be saved at:  {args.output_jsonls_directory}')
    print(f'model_path loaded from:  {args.model_path}')

    rank = int(os.getenv('RANK', '0'))
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)

    if torch.cuda.is_available():
        print("CUDA is available.")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA is not available.")
            
    if args.auto:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # ------------------------------ Model Loading ------------------------------
    kwargs = {'device_map': 'auto'} if args.auto else {'device_map': {'': local_rank}}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **kwargs, torch_dtype=torch.float16).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    json_files =['AID.jsonl', 'UCMerced.jsonl', 'WHU_RS19.jsonl', 'Bigearthnet_test.jsonl', 'fmow_rgb_images_val.jsonl']
    
    start_time = time.time()

    for json_file in json_files:
        print(f"\n[Start Processing] {json_file}")
        t0 = time.time()
        evaluate_folder(model, processor, args.test_data_directory, json_file, args.output_jsonls_directory, device)
        print(f"[Evaluation Complete for ] {json_file} in {round(time.time() - t0, 2)} seconds")

    print(f"\n✅ All Evaluation Complete in {round(time.time() - start_time, 2)} seconds.")
