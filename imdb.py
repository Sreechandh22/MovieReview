import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import zipfile
from tqdm import tqdm  # For the progress bar
import matplotlib.pyplot as plt
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load IMDB data
def load_imdb_data(data_file):
    df = pd.read_csv(data_file)
    texts = df['review'].tolist()
    labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]
    return texts, labels

# Extract data
zip_path = "C:\\Users\\sreec\\OneDrive\\Desktop\\Research\\archive.zip"
extract_path = "C:\\Users\\sreec\\OneDrive\\Desktop\\Research"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

data_file = "C:\\Users\\sreec\\OneDrive\\Desktop\\Research\\IMDB Dataset.csv"
texts, labels = load_imdb_data(data_file)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Define your BERT model class here
class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# Define your training function here
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for batch in tqdm(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

# Define your evaluation function here
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

# Hyperparameters and setup
bert_model_name = 'bert-base-uncased'
num_classes = 2
max_length = 128
batch_size = 16
num_epochs = 4
learning_rate = 2e-5

# Define the TextClassificationDataset class
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

# Data preparation
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Model, optimizer, and scheduler
model = BERTClassifier(bert_model_name, num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    total_loss = 0  # Initialize total_loss to accumulate loss over accumulation_steps
    
    # Training
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss = loss / accumulation_steps  # Divide loss by accumulation_steps
        loss.backward()
        
        total_loss += loss.item()  # Accumulate loss
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Calculate average loss for the epoch
    average_loss = total_loss / len(train_dataloader)
    print(f"Average Loss: {average_loss:.4f}")
    
    # Evaluation
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)

# Save the model
torch.save(model.state_dict(), "bert_classifier.pth")

def visualize_predictions(model, texts, labels, tokenizer, max_length, num_samples=5):
    plt.figure(figsize=(12, 8))

    for i in range(num_samples):
        idx = random.randint(0, len(texts) - 1)
        text = texts[idx]
        label = labels[idx]

        encoding = tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            padding='max_length',
            truncation=True
        )

        input_ids = encoding['input_ids'].flatten().to(device)
        attention_mask = encoding['attention_mask'].flatten().to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            predicted_label = torch.argmax(outputs, dim=1).item()

        sentiment = "positive" if predicted_label == 1 else "negative"
        true_sentiment = "positive" if label == 1 else "negative"

        plt.subplot(2, num_samples, i + 1)
        plt.text(
            0.5, 0.5,
            f"Text: {text[:50]}...\nPredicted: {sentiment}\nTrue: {true_sentiment}",
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10,
            wrap=True
        )
        plt.axis('off')

    plt.show()
