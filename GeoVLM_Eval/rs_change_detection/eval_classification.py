#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#
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

# -----------------------------
# Sample F1 evaluation
# -----------------------------
def evaluate_f1(reference, candidate):
    reference = reference.strip().lower().replace(" ", "")
    candidate = candidate.strip().lower().replace(" ", "")
    #ref_tokens = word_tokenize(reference)
    #cand_tokens = word_tokenize(candidate)
    
    ref_tokens = safe_word_tokenize(reference)
    cand_tokens = safe_word_tokenize(candidate)


    common = set(ref_tokens) & set(cand_tokens)
    if not common:
        return 0

    recall = len(common) / len(ref_tokens)
    return recall




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='xBD_testset_1,xBD_testset_2,xBD_testset_3,xBD_reg_cls_testset_1,xBD_reg_cls_testset_2')
    parser.add_argument('--output_path', type=str, default='./GeoVLM_VLM_R1_Results')

    args = parser.parse_args()

    args.datasets = args.datasets.split(',')

    for ds_name in args.datasets:

        json_path = f'{args.output_path}/{ds_name}.jsonl'

        f = open(json_path)

        data = [json.loads(line) for line in f.readlines()]

        recall = 0
        count = 0
        for idx, item in enumerate(data):
            response = item['response'][0]
            reference = item['groundtruth0'][0]
            
            score = evaluate_f1(reference, response)
            count = count + 1
            recall = recall + score
            
        recall = recall/count if count > 0 else 0.0
        print(f'{Fore.RED}Classification accuracy for {ds_name} dataset: {recall}')




