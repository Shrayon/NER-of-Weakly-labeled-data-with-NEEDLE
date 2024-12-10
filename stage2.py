import warnings

warnings.filterwarnings("ignore")

import os
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AdamW
from torch.utils.data import DataLoader
from models import BertCrf
from methods import get_metrics, visualize_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_valid_test_path = 'dataset/'
model_checkpoint = "mlm_model/"
model_name = 'bert'
batch_size = 16
max_sequence_length = 256
task = "ner"
epochs = 2
bert_name = model_checkpoint
use_crf = True
dropout = 0.2
log_every = 20
lr_new_layers = 1e-3
lr_bert = 1e-5
model_path = f'saved_models/{model_name}_{"CRF" if use_crf else ""}.pt'

label_list = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]
num_labels = len(label_list)
label2id = {
    "O": 0,
    "B-Chemical": 1,
    "B-Disease": 2,
    "I-Disease": 3,
    "I-Chemical": 4
}
id2label = {idx: tag for tag, idx in label2id.items()}

datasets = load_dataset("tner/bc5cdr")

with open('data/weakly_labeled_data.txt', "r") as f:
    text = f.read()

weak_data = {'tokens': [], 'tags': []}
segments = text.strip().split("\n\n")[:5000]
for segment in segments:
    lines = segment.split("\n")
    words_and_labels = [(line.split("\t")[0], label2id[line.split("\t")[1]]) for line in lines if
                        line.split("\t")[0].isalpha()]
    words, labels = zip(*words_and_labels)
    weak_data['tokens'].append(words)
    weak_data['tags'].append(labels)

weak_dataset = Dataset.from_dict(weak_data)
datasets['weak'] = weak_dataset

print(datasets)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_and_align_labels(examples, label_all_tokens=False):
    tokenized_inputs = tokenizer(examples["tokens"], max_length=max_sequence_length, truncation=True,
                                 is_split_into_words=True)
    labels = []
    word_ids_arr = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_ids_arr.append(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    tokenized_inputs["word_ids"] = word_ids_arr
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)


def collate_function(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    input_ids = [torch.tensor(item["input_ids"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    attention_mask = [torch.ones(len(item)) for item in input_ids]

    return {
        "input_ids": pad_sequence(input_ids, batch_first=True),
        "labels": pad_sequence(labels, batch_first=True),
        "attention_mask": pad_sequence(attention_mask, batch_first=True),
    }


train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=batch_size, shuffle=True,
                              collate_fn=collate_function)
valid_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=batch_size, shuffle=False,
                              collate_fn=collate_function)
test_dataloader = DataLoader(tokenized_datasets['test'], batch_size=batch_size, shuffle=False,
                             collate_fn=collate_function)
weak_dataloader = DataLoader(tokenized_datasets['weak'], batch_size=batch_size, shuffle=False,
                             collate_fn=collate_function)

model = BertCrf(num_labels=num_labels, bert_name=bert_name, dropout=dropout, use_crf=use_crf)
model = model.to(device)

optimizer = AdamW(
    [
        {"params": model.start_transitions},
        {"params": model.end_transitions},
        {"params": model.hidden2label.parameters()},
        {"params": model.transitions},
        {"params": model.bert.parameters(), "lr": lr_bert},
    ],
    lr=lr_new_layers,
)


def dict_to_device(dict: Dict[str, torch.Tensor], device):
    for key, value in dict.items():
        dict[key] = value.to(device)
    return dict


if os.path.exists(model_path):
    model.load_from(model_path)
else:
    train_loss, valid_loss = 0, 0
    best_valid_loss = float("inf")
    best_model = None
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Training Epoch {epoch}", leave=False):
            optimizer.zero_grad()
            batch = dict_to_device(batch, device)
            loss = model(**batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)

        model.eval()
        valid_predictions, valid_ground_truth = [], []
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, total=len(valid_dataloader), desc=f"Validating Epoch {epoch}",
                              leave=False):
                batch = dict_to_device(batch, device)
                logits = model(**batch)
                valid_loss += logits.item()
                labels = batch["labels"]
                del batch["labels"]
                prediction = model.decode(**batch)
                valid_predictions = [item for sublist in prediction for item in sublist]
                valid_ground_truth = torch.masked_select(labels.to(device), batch["attention_mask"].bool()).tolist()
        valid_loss /= len(valid_dataloader)
        f1 = f1_score(valid_ground_truth, valid_predictions, average='weighted')
        accuracy = accuracy_score(valid_ground_truth, valid_predictions)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = model
        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, f1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    if best_model:
        model = best_model
        model.save_to(f'saved_models/{model_name}_{"CRF" if use_crf else ""}.pt')

model.eval()
predictions, ground_truth = [], []

for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Testing"):
    labels = batch["labels"].to(device)
    del batch["labels"]
    batch = dict_to_device(batch, device)
    prediction = model.decode(**batch)
    flatten_prediction = [item for sublist in prediction for item in sublist]
    flatten_labels = torch.masked_select(labels, batch["attention_mask"].bool()).tolist()
    predictions.append(flatten_prediction)
    ground_truth.append(flatten_labels)

true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, ground_truth)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, ground_truth)
]

result_df = get_metrics(true_predictions, true_labels, label_list)
result_df.to_csv(f'output/initial_report.csv', index=False)
print(result_df)

visualize_confusion_matrix(true_predictions, true_labels, label_list, f'output/initial_confusion_matrix.png')

tokenized_weak_dataset = tokenized_datasets['weak']
weak_dataloader = DataLoader(tokenized_weak_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_function)

model.eval()
all_probs = []
all_correct = []
all_predictions = []
with torch.no_grad():
    for batch in tqdm(weak_dataloader, desc="Calculating Confidence Scores"):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch['labels']
        predictions, probs = model.decode_with_confidence(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        for i in range(len(predictions)):
            seq_labels = labels[i][batch['attention_mask'][i].bool()].cpu().numpy()
            seq_preds = predictions[i]
            seq_probs = probs[i]
            valid_preds = [seq_preds[token_idx] for token_idx, mask in
                           enumerate(batch['attention_mask'][i].cpu().numpy()) if mask]
            all_predictions.append(valid_preds)
            for label_id, pred_id, prob in zip(seq_labels, seq_preds, seq_probs):
                if label_id != -100:
                    all_probs.append(prob)
                    all_correct.append(int(label_id == pred_id))

assert len(all_predictions) == len(
    tokenized_weak_dataset['input_ids']), "Mismatch between predictions and dataset size."
tokenized_weak_dataset = tokenized_weak_dataset.add_column('predicted_labels', all_predictions)

tokenized_weak_dataset = tokenized_weak_dataset.remove_columns('labels')
tokenized_weak_dataset = tokenized_weak_dataset.rename_column('predicted_labels', 'labels')


def histogram_binning(probs, correct, num_bins=10):
    bins = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_correct = np.zeros(num_bins)
    bin_total = np.zeros(num_bins)
    for idx, bin_idx in enumerate(bin_indices):
        bin_correct[bin_idx] += correct[idx]
        bin_total[bin_idx] += 1
    bin_accuracy = bin_correct / np.maximum(bin_total, 1)
    calibrated_probs = np.zeros_like(probs)
    for i in range(num_bins):
        indices = bin_indices == i
        calibrated_probs[indices] = bin_accuracy[i]
    return calibrated_probs, bins, bin_accuracy


calibrated_probs, bins, bin_accuracy = histogram_binning(np.array(all_probs), np.array(all_correct), num_bins=10)

aligned_confidence = []
current_index = 0
for i, batch in enumerate(tqdm(weak_dataloader, desc="Aligning Confidence Scores")):
    batch_size = batch['input_ids'].shape[0]
    attention_masks = batch['attention_mask'].cpu().numpy()
    for j in range(batch_size):
        mask = attention_masks[j]
        num_tokens = int(mask.sum())
        confidence_seq = calibrated_probs[int(current_index):int(current_index) + num_tokens].tolist()
        current_index += num_tokens
        confidence_seq = [round(f, 1) for f in confidence_seq]
        aligned_confidence.append(confidence_seq)

assert len(aligned_confidence) == len(tokenized_weak_dataset['input_ids'])

tokenized_weak_dataset = tokenized_datasets['weak'].add_column('confidence', aligned_confidence)
tokenized_train_dataset = tokenized_datasets['train'].map(lambda x: {'confidence': [1.0] * len(x['labels'])})

from datasets import Sequence, Value

updated_features = tokenized_weak_dataset.features.copy()
updated_features['tags'] = Sequence(feature=Value(dtype='int32'))

tokenized_weak_dataset = tokenized_weak_dataset.cast(updated_features)

from datasets import concatenate_datasets

combined_dataset = concatenate_datasets([tokenized_train_dataset, tokenized_weak_dataset])

tokenized_datasets['combined'] = combined_dataset
print(tokenized_datasets)

combined_dataloader = DataLoader(tokenized_datasets['combined'], batch_size=batch_size, shuffle=True,
                                 collate_fn=collate_function)

train_loss, valid_loss = 0, 0
best_valid_loss = float("inf")
best_model = None
epochs = 1
for epoch in range(1, epochs + 1):
    model.train()
    for batch in tqdm(combined_dataloader, total=len(combined_dataloader), desc=f"Training Epoch {epoch}", leave=False):
        optimizer.zero_grad()
        batch = dict_to_device(batch, device)
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(combined_dataloader)
    model.eval()
    valid_predictions, valid_ground_truth = [], []
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, total=len(valid_dataloader), desc=f"Validating Epoch {epoch}",
                          leave=False):
            batch = dict_to_device(batch, device)
            logits = model(**batch)
            valid_loss += logits.item()
            labels = batch["labels"]
            del batch["labels"]
            prediction = model.decode(**batch)
            valid_predictions = [item for sublist in prediction for item in sublist]
            valid_ground_truth = torch.masked_select(labels.to(device), batch["attention_mask"].bool()).tolist()
    valid_loss /= len(valid_dataloader)
    f1 = f1_score(valid_ground_truth, valid_predictions, average='weighted')
    accuracy = accuracy_score(valid_ground_truth, valid_predictions)
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        best_model = model
    print(
        f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, f1: {f1:.4f}, Accuracy: {accuracy:.4f}")
if best_model:
    model = best_model
    model.save_to(f'saved_models/weak_{model_name}_{"CRF" if use_crf else ""}.pt')

model.eval()
predictions, ground_truth = [], []

for batch in tqdm(test_dataloader, total=len(test_dataloader), desc="Testing"):
    labels = batch["labels"].to(device)
    del batch["labels"]
    batch = dict_to_device(batch, device)
    prediction = model.decode(**batch)
    flatten_prediction = [item for sublist in prediction for item in sublist]
    flatten_labels = torch.masked_select(labels, batch["attention_mask"].bool()).tolist()
    predictions.append(flatten_prediction)
    ground_truth.append(flatten_labels)

true_predictions = [
    [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, ground_truth)
]
true_labels = [
    [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, ground_truth)
]

result_df = get_metrics(true_predictions, true_labels, label_list)
result_df.to_csv(f'output/pre_trained_report.csv', index=False)
print(result_df)

visualize_confusion_matrix(true_predictions, true_labels, label_list, f'output/pre_trained_confusion_matrix.png')
