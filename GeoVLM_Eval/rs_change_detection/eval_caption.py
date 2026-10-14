#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#


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

def calculate_rouge(hypothesis, references):

    # Tokenize the hypothesis
    hypothesis_tokens = " ".join(wordpunct_tokenize(hypothesis))

    # Initialize the scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Calculate ROUGE scores for each reference and take the average
    scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}
    for ref in references:
        ref_tokens = " ".join(wordpunct_tokenize(ref))
        score = scorer.score(ref_tokens, hypothesis_tokens)
        scores['rouge1'] += score['rouge1'].fmeasure
        scores['rouge2'] += score['rouge2'].fmeasure
        scores['rougeL'] += score['rougeL'].fmeasure

    # Average the scores over the number of references
    for key in scores:
        scores[key] /= len(references)

    return scores['rouge1'], scores['rougeL']



from nltk.translate.meteor_score import meteor_score

def calculate_meteor(hypothesis, references):

    # Tokenize the hypothesis and references
    hypothesis_tokens = wordpunct_tokenize(hypothesis)
    references_tokens = [wordpunct_tokenize(ref) for ref in references]

    return meteor_score(references_tokens, hypothesis_tokens)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='DubaiCC,LevirCC,MUDS,SYSUCC,xBD_image_captioning')
    parser.add_argument('--output_path', type=str, default='./GeoVLM_VLM_R1_Results')
    args = parser.parse_args()

    args.datasets = args.datasets.split(',')

    for ds_name in args.datasets:
        json_path = f'{args.output_path}/{ds_name}.jsonl'
        f = open(json_path)
        #print('processing at path: ', json_path)
        data = [json.loads(line) for line in f.readlines()]

        rouge1_precision, rouge1_recall, rouge1_fscore = 0, 0, 0
        rougeL_precision, rougeL_recall, rougeL_fscore = 0, 0, 0
        M_score = 0

        count = 0
        for idx, item in enumerate(data):

            response = item['response'][0]
            
            # Use all non-empty references (some datasets provide fewer than 5 captions)
            reference = [item[f'groundtruth{k}'][0] for k in range(5)
                         if f'groundtruth{k}' in item and len(item[f'groundtruth{k}']) > 0 and len(item[f'groundtruth{k}'][0].strip()) > 0]
            if len(reference) == 0:
                continue
            
            rouge1, rougeL = calculate_rouge(response, reference)
            rouge1_fscore = rouge1_fscore + rouge1
            rougeL_fscore = rougeL_fscore + rougeL
            
            M_score = M_score + calculate_meteor(response, reference)
            count = count + 1
            
        count = max(count, 1)  # avoid ZeroDivisionError
        rouge1_fscore = rouge1_fscore/count
        rougeL_fscore = rougeL_fscore/count
        M_score = M_score/count
        print('-'*40)
        print(f'{Fore.RED}Scores for {ds_name} dataset')
        print(f'{Fore.RED}ROUGE-1: Fmeasure: {rouge1_fscore:.4f}')
        print(f'{Fore.RED}ROUGE-L: Fmeasure: {rougeL_fscore:.4f}')
        print(f'{Fore.RED}METEOR Score: {M_score:.4f}')
        print('-'*40)
