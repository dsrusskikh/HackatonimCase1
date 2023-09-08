import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
import stop_words

from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, RobertaForSequenceClassification, AdamW, Trainer, TrainingArguments
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report


#ввод данных
df = pd.read_excel('CRA_train_1200.xlsx')
df['pr_txt'] = df['pr_txt'].astype(str).str.zfill(6)
df = df.head(100) # для быстрого теста работы кода

# перекодируем текстовые значения в числовые
label_encoder = LabelEncoder()
df['Категория'] = label_encoder.fit_transform(df['Категория'])

# разделим данные на обучающие и тестовые
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=0)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0)

# загрузим токенизатор и применим его
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

def tokenize_text(text):
    return tokenizer(
        text,
        padding='max_length',  # Pad sequences to the same length
        truncation=True,       # Truncate sequences if they exceed the maximum length
        max_length=128,        # You can adjust the maximum sequence length
        return_tensors='pt',   # Return PyTorch tensors
    )

train_dataset = train_data['pr_txt'].apply(tokenize_text)
val_dataset = val_data['pr_txt'].apply(tokenize_text)
test_dataset = test_data['pr_txt'].apply(tokenize_text)

# задаем модель для классификатора
from transformers import DistilBertForSequenceClassification

class CreditRatingClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CreditRatingClassifier, self).__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-multilingual-cased", num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.logits

# загружаем модель
num_classes = len(df['Категория'].unique())
model = CreditRatingClassifier(num_classes)

# зададим датасет
class TextClassificationDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# сделаем подгрузку данных для обучения
train_dataset = TextClassificationDataset(train_dataset, train_data['Категория'])
val_dataset = TextClassificationDataset(val_dataset, val_data['Категория'])

# зададим директорию
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir='./output',
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

# запуск
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# настройка
trainer.train()

# проверим результативность модели на тестовой части
test_results = trainer.predict(test_dataset)
test_preds = test_results.predictions.argmax(axis=1)
test_labels = test_data['Категория']

# конвертируем номера категорий обратно в текстовые значения
test_preds_text = label_encoder.inverse_transform(test_preds)
test_labels_text = label_encoder.inverse_transform(test_labels)

# просматриваем результат отработки модели
classification_rep = classification_report(test_labels_text, test_preds_text)

print("Classification Report:")
print(classification_rep)
