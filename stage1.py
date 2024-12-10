from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset
import torch

unlabeled_file = "data/cleaned_unlabeled_sentence_data.txt"
model_checkpoint = "google-bert/bert-base-uncased"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

with open(unlabeled_file, "r") as f:
    lines = f.readlines()
unlabeled_data = {"text": [line.strip() for line in lines if line.strip()]}
dataset = Dataset.from_dict(unlabeled_data)

print(dataset)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(tokenized_dataset)

def group_texts(examples, block_size=128):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)

print(lm_dataset)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, mlm=True)

model = AutoModelForMaskedLM.from_pretrained(model_checkpoint, trust_remote_code=True).to(device)

training_args = TrainingArguments(
    output_dir="./mlm_model",
    overwrite_output_dir=True,
    eval_strategy="no",
    per_device_train_batch_size=32,
    num_train_epochs=1,
    learning_rate=5e-5,
    warmup_steps=500,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)
trainer.train()

trainer.save_model("./mlm_model")
tokenizer.save_pretrained("./mlm_model")
print("Training completed and model saved!")