# -*- coding: utf-8 -*-
"""inference_modell.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uOVZtV3Mc1j-vySS6bxYbIaxEmPDQr2u
"""

import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import numpy as np

# Load the trained model and tokenizer
model = BertForTokenClassification.from_pretrained('ner_mountain_model')
tokenizer = BertTokenizerFast.from_pretrained('ner_mountain_model')

# Define label list
label_list = ['O', 'B-MTN', 'I-MTN']

# Function to perform NER on input text
def predict_mountain_names(text):
    # Tokenize input text
    tokens = tokenizer.tokenize(text)
    inputs = tokenizer.encode_plus(text, return_tensors="pt")

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    # Convert predictions to labels
    predicted_labels = [label_list[prediction] for prediction in predictions[0].numpy()[1:-1]]  # Skip [CLS] and [SEP]
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])[1:-1]  # Skip [CLS] and [SEP]

    # Extract mountain names
    mountain_names = []
    current_mountain = []
    for token, label in zip(tokens, predicted_labels):
        if label == 'B-MTN':
            if current_mountain:
                mountain_names.append(' '.join(current_mountain))
                current_mountain = []
            current_mountain.append(token)
        elif label == 'I-MTN':
            current_mountain.append(token)
        else:
            if current_mountain:
                mountain_names.append(' '.join(current_mountain))
                current_mountain = []
    if current_mountain:
        mountain_names.append(' '.join(current_mountain))

    # Clean up tokens (remove '##' from subword tokens)
    clean_mountain_names = []
    for name in mountain_names:
        name = name.replace('##', '')
        clean_mountain_names.append(name)

    return clean_mountain_names

# Main function for inference
def main():
    # Input text
    text = input("Enter a sentence: ")

    # Predict mountain names
    mountain_names = predict_mountain_names(text)

    if mountain_names:
        print("Identified mountain names:")
        for name in mountain_names:
            print(name)
    else:
        print("No mountain names identified.")

if __name__ == '__main__':
    main()