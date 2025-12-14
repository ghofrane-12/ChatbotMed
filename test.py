from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np

# 1) Charger dataset CSV
dataset = load_dataset(
    "csv",
    data_files={"data": "medicament_chatbot/data/medications_6000_multilang.csv"}
)

# 2) Preprocess (batched!)
lang_map = {"fr": 0, "en": 1, "ar": 2}

def preprocess(batch):
    # construire le texte
    generic = [str(x) if x is not None else "" for x in batch["generic_name"]]
    form    = [str(x) if x is not None else "" for x in batch["form"]]
    dosage  = [str(x) if x is not None else "" for x in batch["dosage"]]
    sale    = [str(x) if x is not None else "" for x in batch["sale_type"]]

    batch["text"] = [
        f"{g} {f} {d} {s}".strip()
        for g, f, d, s in zip(generic, form, dosage, sale)
    ]

    # labels
    batch["labels"] = [lang_map.get(str(l).strip().lower(), 1) for l in batch["lang"]]
    return batch

dataset = dataset["data"].map(preprocess, batched=True)

# 3) Split train/test
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
test_dataset  = train_test["test"]

# 4) Modèle multilingue
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 5) Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

# 6) Nettoyer colonnes (Trainer attend input_ids/attention_mask/labels)
cols_to_remove = [c for c in train_dataset.column_names if c not in ["input_ids", "attention_mask", "labels"]]
train_dataset = train_dataset.remove_columns(cols_to_remove)
test_dataset  = test_dataset.remove_columns(cols_to_remove)

train_dataset.set_format("torch")
test_dataset.set_format("torch")

# 7) Modèle classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# (Optionnel) métriques
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# 8) TrainingArguments
training_args = TrainingArguments(
    output_dir="medicament_chatbot/models/fine_tuned_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
)

# 9) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 10) Save final
trainer.save_model("medicament_chatbot/models/fine_tuned_model")
tokenizer.save_pretrained("medicament_chatbot/models/fine_tuned_model")

print("✅ Modèle fine-tuned sauvegardé dans medicament_chatbot/models/fine_tuned_model")
