import warnings

warnings.filterwarnings("ignore")

import torch
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from datasets import load_dataset
from transformers import AutoTokenizer, AdamW
from torch.utils.data import DataLoader
from models import BertCrf
from methods import get_metrics, visualize_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

train_valid_test_path = 'dataset/'
model_checkpoint = "saved_models/weak_bert_CRF.pt"
model_name = 'bert'
batch_size = 16
max_sequence_length = 256
task = "ner"
epochs = 1
bert_name = model_checkpoint
use_crf = True
dropout = 0.2
log_every = 20
lr_new_layers = 1e-3
lr_bert = 1e-5
model_path = f'saved_models/weak_{model_name}_{"CRF" if use_crf else ""}.pt'

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

print(datasets)

tokenizer = AutoTokenizer.from_pretrained("mlm_model")


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

model = BertCrf(num_labels=num_labels, bert_name="mlm_model", dropout=dropout, use_crf=use_crf)
model.load_from(model_path)
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
    model.save_to(f'saved_models/final_{model_name}_{"CRF" if use_crf else ""}')

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
result_df.to_csv(f'output/final_report.csv', index=False)
print(result_df)

visualize_confusion_matrix(true_predictions, true_labels, label_list, f'output/final_confusion_matrix.png')
