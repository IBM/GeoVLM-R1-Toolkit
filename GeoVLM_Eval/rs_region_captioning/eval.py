#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#
from rouge_score import rouge_scorer
import os
import json
from colorama import Fore, init

from sklearn.metrics import precision_score, recall_score, f1_score
import json
import argparse
from colorama import Fore, init
from nltk.tokenize import word_tokenize
import nltk
# Setup NLTK
nltk.data.path = ["./nltk_data"] + nltk.data.path
init(autoreset=True)

# Ensure Punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print("Punkt tokenizer not found!")
import os
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')  # optional but recommended for full support

import nltk
nltk.download('wordnet', download_dir='./nltk_data')
nltk.download('omw-1.4', download_dir='./nltk_data')


def calculate_rouge_scores(reference_text, candidate_text):
    # Initialize the ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    
    # Calculate ROUGE scores
    scores = scorer.score(reference_text, candidate_text)
    
    # Extract ROUGE-1 and ROUGE-L scores
    rouge1 = scores['rouge1']
    rougel = scores['rougeL']
    
    return rouge1, rougel



def calculate_meteor(reference, hypothesis):
    """
    Calculate the METEOR score between a reference sentence and a hypothesis sentence.

    :param reference: A string or a list of strings representing the reference sentence(s)
    :param hypothesis: A string representing the hypothesis (predicted) sentence
    :return: The METEOR score
    """
    # Tokenize both reference and hypothesis
    ref_tokenized = reference.split()  # Tokenizing by splitting on spaces
    hypothesis_tokenized = hypothesis.split()    
    score = meteor_score([ref_tokenized], hypothesis_tokenized)

    return score



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='Region_captioning_geochat,NWPU_VHR_10_instructions_region_captioning')
    parser.add_argument('--output_path', type=str, default='./GeoVLM_VLM_R1_Results')

    
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')

    for ds_name in args.datasets:

        json_path = f'{args.output_path}/{ds_name}.jsonl'

        f = open(json_path)

        data = [json.loads(line) for line in f.readlines()]       


        rouge1_precision, rouge1_recall, rouge1_fscore = 0, 0, 0
        rougeL_precision, rougeL_recall, rougeL_fscore = 0, 0, 0
        M_score = 0

        count = 0
        for idx, item in enumerate(data):

            response = item['response'][0]
            reference = item['ground_truth'][0]
            
            rouge1, rougel = calculate_rouge_scores(reference, response)

            rouge1_precision = rouge1_precision + rouge1.precision
            rouge1_recall = rouge1_recall + rouge1.recall
            rouge1_fscore = rouge1_fscore + rouge1.fmeasure

            rougeL_precision = rougeL_precision + rougel.precision
            rougeL_recall = rougeL_recall + rougel.recall
            rougeL_fscore = rougeL_fscore + rougel.fmeasure

            M_score = M_score + calculate_meteor(reference, response)
            count = count + 1


        count = max(count, 1)  # avoid ZeroDivisionError
        rouge1_precision = rouge1_precision/count
        rouge1_recall = rouge1_recall/count
        rouge1_fscore = rouge1_fscore/count

        rougeL_precision = rougeL_precision/count
        rougeL_recall = rougeL_recall/count
        rougeL_fscore = rougeL_fscore/count

        M_score = M_score/count


        print('-'*40)

        print(f'{Fore.RED}Scores for {ds_name} dataset')
        print(f'{Fore.RED}ROUGE-1: Precision: {rouge1_precision:.4f}, Recall: {rouge1_recall:.4f}, F1 Score: {rouge1_fscore:.4f}')
        print(f'{Fore.RED}ROUGE-L: Precision: {rougeL_precision:.4f}, Recall: {rougeL_recall:.4f}, F1 Score: {rougeL_fscore:.4f}')
        print(f'{Fore.RED}METEOR Score: {M_score:.4f}')

        print('-'*40)
