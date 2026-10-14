#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#
import numpy as np
from shapely.geometry import Polygon
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
import json
import re
import os


from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score
import os
import pickle
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from colorama import init
import argparse
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
from colorama import Fore, init

# Init Colorama
init(autoreset=True)

# Set and add NLTK custom data path
NLTK_DATA_DIR = "./nltk_data"
nltk.data.path = [NLTK_DATA_DIR] + nltk.data.path

# Download required data to correct location
nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
nltk.download('omw-1.4', download_dir=NLTK_DATA_DIR)

# Load Punkt tokenizer manually
english_punkt_path = os.path.join(NLTK_DATA_DIR, "tokenizers", "punkt", "english.pickle")
if os.path.exists(english_punkt_path):
    with open(english_punkt_path, "rb") as f:
        punkt_tokenizer = pickle.load(f)

    # Monkeypatch sent_tokenize and word_tokenize
    def safe_sent_tokenize(text):
        return punkt_tokenizer.tokenize(text)

    def safe_word_tokenize(text):
        # Use sent_tokenize first to break sentences
        sentences = punkt_tokenizer.tokenize(text)
        return [word for sent in sentences for word in wordpunct_tokenize(sent)]

    # Test it
    test_text = "Hello world. This is a test!"
    print("Sentences:", safe_sent_tokenize(test_text))
    print("Words:", safe_word_tokenize(test_text))

else:
    print("❌ english.pickle not found in:", english_punkt_path)

    # Fallback so the script does not crash with NameError when the Punkt pickle is unavailable
    def safe_sent_tokenize(text):
        return [text]

    def safe_word_tokenize(text):
        return wordpunct_tokenize(text)

# ----------------------
import os
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # optional but recommended for full support

import nltk
nltk.download('wordnet', download_dir='./nltk_data')
nltk.download('omw-1.4', download_dir='./nltk_data')
import argparse
from colorama import Fore, init
init(autoreset=True)  # Initialize Colorama

def get_rotated_box_vertices(bx_left, by_top, bx_right, by_bottom, theta):
    """
    Convert a rotated bounding box to its vertices.

    Parameters:
    - bx_left, by_top, bx_right, by_bottom: coordinates of the bounding box before rotation
    - theta: rotation angle in degrees

    Returns:
    - List of vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    """
    cx = (bx_left + bx_right) / 2
    cy = (by_top + by_bottom) / 2

    width = bx_right - bx_left
    height = by_bottom - by_top

    rect = np.array([
        [bx_left, by_top],
        [bx_right, by_top],
        [bx_right, by_bottom],
        [bx_left, by_bottom]
    ])

    rotation_matrix = np.array([
        [np.cos(np.radians(theta)), -np.sin(np.radians(theta))],
        [np.sin(np.radians(theta)), np.cos(np.radians(theta))]
    ])

    # Shift to the origin (center), apply rotation, then shift back
    rect_centered = rect - np.array([cx, cy])
    rotated_rect = np.dot(rect_centered, rotation_matrix) + np.array([cx, cy])

    return rotated_rect.tolist()

def calculate_iou(box1, box2):
    """
    Calculate IoU of two rotated bounding boxes.

    Parameters:
    - box1, box2: list of vertices [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    Returns:
    - IoU value
    """
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)

    if not poly1.is_valid or not poly2.is_valid:
        return 0

    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area

    iou = intersection_area / union_area if union_area > 0 else 0
    return iou



def parse_bboxes(bbox_str):
    """Extract bounding boxes from a string containing multiple bbox lists."""
    # Ensure proper list format by adding a comma between consecutive brackets
    formatted_str = bbox_str.replace("][", "], [")
    
    # Extract the list of bounding boxes using regex
    bbox_list = re.findall(r'\[\s*(\d+(?:,\s*\d+)*)\s*\]', formatted_str)
    
    bboxes = []
    for bbox in bbox_list:
        try:
            # Convert the extracted string into a list of integers
            bbox_int = list(map(int, bbox.split(',')))
            bboxes.append(bbox_int)
        except ValueError as e:
            print(f"Error converting {bbox} to integers: {e}")
    
    return bboxes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='xBD_object_detection')
    parser.add_argument('--output_path', type=str, default='./GeoVLM_VLM_R1_Results')

    args = parser.parse_args()

    args.datasets = args.datasets.split(',')

    for ds_name in args.datasets:

        json_path = f'{args.output_path}/{ds_name}.jsonl'
        f = open(json_path)

        data = [json.loads(line) for line in f.readlines()]

        correct = total_cnt = count = 0
        for i, output in enumerate(data):
            
            pred_bboxes = parse_bboxes(output['response'][0])
            gt_bboxes = parse_bboxes(output['groundtruth0'][0])
            
            
            total_cnt += len(pred_bboxes)
            
            for pred_box in pred_bboxes:
                pred_vertices = get_rotated_box_vertices(*pred_box)
                for gt_box in gt_bboxes:
                    gt_vertices = get_rotated_box_vertices(*gt_box)
                    iou = calculate_iou(pred_vertices, gt_vertices)   
                    if iou >= 0.5:
                        correct += 1
                        break  # Count each prediction only once
                    

        print('-'*40)
        
        print(f'{Fore.RED}Precision score at 0.5 for {ds_name} dataset: {correct / total_cnt if total_cnt > 0 else 0.0}')
        
        print('-'*40)  
        
        correct = total_cnt = count = 0
        for i, output in enumerate(data):
            
            pred_bboxes = parse_bboxes(output['response'][0])
            gt_bboxes = parse_bboxes(output['groundtruth0'][0])
            
            
            total_cnt += len(pred_bboxes)
            
            for pred_box in pred_bboxes:
                pred_vertices = get_rotated_box_vertices(*pred_box)
                for gt_box in gt_bboxes:
                    gt_vertices = get_rotated_box_vertices(*gt_box)
                    iou = calculate_iou(pred_vertices, gt_vertices)   
                    if iou >= 0.25:
                        correct += 1
                        break  # Count each prediction only once
                    

        print('-'*40)
        
        print(f'{Fore.RED}Precision score at 0.25 for {ds_name} dataset: {correct / total_cnt if total_cnt > 0 else 0.0}')
        
        print('-'*40)         
       

