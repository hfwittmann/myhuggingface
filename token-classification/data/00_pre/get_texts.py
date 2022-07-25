from __future__ import annotations

import json
import os

from tokenizers import Encoding
from transformers import BatchEncoding, BertTokenizerFast

from alignment import align_tokens_and_annotations_bilou

tokenizer = BertTokenizerFast.from_pretrained('distilbert-base-german-cased') # Load a pre-trained tokenizer
        
os.chdir('/home/hfwittmann/Sync/DataScience/labelling')


min_json = "project-1-at-2022-07-25-08-59-7c05d43f-min.json"

with open(min_json, 'r') as file:
    all_documents = json.load(file)
    
    


for ix, one_document in enumerate(all_documents):
    
    text_filename = one_document['text'].replace('/data/', '')
    
    with open(text_filename, 'r') as file:
        text = file.read()
        

    annotations = one_document['label']

    for ix, annotation in enumerate(annotations):
        start = annotation['start']
        end = annotation['end']
        annotations[ix]['label'] = annotations[ix]['labels'][0]
        annotations[ix]['text'] = text[start:end]
        

    tokenized_batch : BatchEncoding = tokenizer(text)
    tokenized_text :Encoding  =tokenized_batch[0]


    tokens = tokenized_text.tokens

    

    labels = align_tokens_and_annotations_bilou(tokenized_text, annotations)
    for token, label in zip(tokens, labels):
        print(token, "-", label)
            


print('finished')
    