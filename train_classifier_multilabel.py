import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, get_scheduler
from torch.optim import AdamW
from model import BertForMultiLabelClassification
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
from preprocess import clean_text
from tqdm import tqdm
import os
import logging
from logging import config
import yaml

model_checkpoint = "bert-base-multilingual-cased"
num_epochs = 5
batch_size = 16
max_length = 128
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
save_dir = "multilabel_model"
validation_split = 0.1
logger_name = "train_classifier_multilabel"


with open('logging_config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger(logger_name)

df = pd.read_csv("dataset.csv")

df["text"] = df["text"].apply(clean_text)
df["labels"] = df["labels"].apply(lambda x: x.split(","))

mlb = MultiLabelBinarizer()
logger.info("Инициализация MultiLabelBinarizer")
y = mlb.fit_transform(df["labels"])

os.makedirs(save_dir, exist_ok=True)
joblib.dump(mlb, os.path.join(save_dir, "mlb.pkl"))

tokenizer = BertTokenizer.from_pretrained(model_checkpoint)


class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


dataset = TextDataset(df["text"].tolist(), y)

val_size = int(len(dataset) * validation_split)
train_size = len(dataset) - val_size
logger.info("Разделение на тренировочную и валидационную выборки")
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

logger.info("Инициализация модели")
model = BertForMultiLabelClassification.from_pretrained(model_checkpoint, num_labels=len(mlb.classes_))
model.to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer,
    num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs
)

best_val_loss = float('inf')

logger.info("Тренировка модели")
for epoch in range(num_epochs):
    logger.info(f"Epoch: {epoch + 1}/{num_epochs}")
    model.train()
    train_loss = 0.0

    loop = tqdm(train_loader, desc="Training", leave=False)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )

        loss = outputs["loss"]
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    logger.info(f"Train loss: {avg_train_loss:.4f}")

    # Валидация
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            val_loss += outputs["loss"].item()

    avg_val_loss = val_loss / len(val_loader)
    logger.info(f"Validation loss: {avg_val_loss:.4f}")

    # Сохраняем только лучшую модель
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        logger.info(f"✅ Новая модель! Сохраняем в папку {save_dir}")
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

logger.info("✅ Тренировка завершена!")