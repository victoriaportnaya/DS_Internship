# -*- coding: utf-8 -*-
"""train_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uOVZtV3Mc1j-vySS6bxYbIaxEmPDQr2u
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import Dataset, DatasetDict
import evaluate
import numpy as np
from sklearn.model_selection import train_test_split

# Step 2: Importing Libraries and Downloading NLTK Data
nltk.download('punkt')

# Step 3: Defining the List of Mountains
mountain_names = [
    'Mount Everest', 'K2', 'Kangchenjunga', 'Lhotse', 'Makalu',
    'Cho Oyu', 'Dhaulagiri', 'Manaslu', 'Nanga Parbat', 'Annapurna'
]

# Step 4: Scraping Wikipedia for Sentences Containing Mountain Names
def get_sentences_from_wikipedia(mountain_name):
    mountain_url = mountain_name.replace(' ', '_')
    url = f'https://en.wikipedia.org/wiki/{mountain_url}'

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Could not retrieve page for {mountain_name}")
            return []

        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text from paragraphs
        paragraphs = soup.find_all('p')
        text_content = ''
        for para in paragraphs:
            text_content += para.get_text()

        # Clean text
        text_content = re.sub(r'\n', '', text_content)

        # Split into sentences
        sentences = nltk.sent_tokenize(text_content)

        relevant_sentences = []
        for sentence in sentences:
            if mountain_name in sentence:
                relevant_sentences.append(sentence)

        return relevant_sentences

    except Exception as e:
        print(f"Error processing {mountain_name}: {e}")
        return []

# Step 5: Creating the Dataset
data = []
for mountain in mountain_names:
    print(f"Processing {mountain}")
    sentences = get_sentences_from_wikipedia(mountain)
    for sentence in sentences:
        data.append({'sentence': sentence, 'mountain': mountain})

df = pd.DataFrame(data)
df.to_csv('mountain_dataset.csv', index=False)
print("Dataset creation complete. Saved to 'mountain_dataset.csv'.")

# Step 6: Preparing Data for Model Training
# Load dataset
df = pd.read_csv('mountain_dataset.csv')

# Define label list and mappings
label_list = ['O', 'B-MTN', 'I-MTN']
label_encoding_dict = {'O': 0, 'B-MTN': 1, 'I-MTN': 2}

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

# Tokenizing Sentences and Assigning Labels
def tokenize_and_label(row):
    sentence = row['sentence']
    mountain = row['mountain']

    # Tokenize sentence
    tokens = tokenizer.tokenize(sentence)

    # Initialize labels
    labels = ['O'] * len(tokens)

    # Tokenize mountain name
    mountain_tokens = tokenizer.tokenize(mountain)
    mountain_token_length = len(mountain_tokens)

    # Find the position of the mountain name in the tokenized sentence
    for i in range(len(tokens) - mountain_token_length + 1):
        if tokens[i:i+mountain_token_length] == mountain_tokens:
            labels[i] = 'B-MTN'
            for j in range(1, mountain_token_length):
                labels[i+j] = 'I-MTN'
            break  # Assuming mountain name appears only once per sentence

    return {'tokens': tokens, 'labels': labels}

# Applying the Tokenization and Labeling
processed_data = df.apply(tokenize_and_label, axis=1)

data_df = pd.DataFrame({
    'tokens': processed_data.apply(lambda x: x['tokens']),
    'labels': processed_data.apply(lambda x: x['labels'])
})

# Splitting the Dataset into Training and Testing Sets
train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

# Aligning Tokens with Labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        is_split_into_words=True,
        truncation=True,
        padding='max_length',
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label[word_idx].startswith('I') else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Tokenize datasets
tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# Step 7: Setting Up the BERT Model for NER
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=len(label_list))

# Step 8: Defining Evaluation Metrics
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [
        [label_list[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [label_list[pred] for (pred, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Step 9: Configuring Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    report_to='none',  # Disable wandb logging
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    greater_is_better=True
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Step 10: Training the Model
trainer.train()

# Step 11: Saving the Trained Model and Tokenizer
model.save_pretrained('ner_mountain_model')
tokenizer.save_pretrained('ner_mountain_model')
print("Model and tokenizer saved in 'ner_mountain_model' directory.")