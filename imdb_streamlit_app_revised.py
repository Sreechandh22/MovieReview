import streamlit as st
import torch
from torch import nn
from transformers import BertTokenizer, BertModel

# Define your BERT model class here (copied from your original model code)
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

@st.cache(allow_output_mutation=True)
def load_model():
    # Update this to the path where your bert_classifier.pth is located
    model_path = 'C:/Users/sreec/OneDrive/Desktop/Research/bert_classifier.pth'
    model = BERTClassifier('bert-base-uncased', 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@st.cache_data
def load_tokenizer():
    # Load the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    return tokenizer

# Function to get sentiment
def get_sentiment(text, model, tokenizer):
    # Tokenize and prepare the text
    encoding = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Get model output
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Get the predicted label (0 or 1)
    _, preds = torch.max(outputs, dim=1)
    predicted_label = preds.item()
    
    return "Positive" if predicted_label == 1 else "Negative"

# Load model and tokenizer
model = load_model()
tokenizer = load_tokenizer()

# Streamlit app
st.title("Sentiment Analysis with BERT")

# Textbox for user input
user_input = st.text_area("Enter a movie review:", "")

# Button to get sentiment
if st.button("Get Sentiment"):
    if user_input:
        # Get and display the sentiment
        sentiment = get_sentiment(user_input, model, tokenizer)
        st.write(f"Predicted Sentiment: {sentiment}")
    else:
        st.write("Please enter a review first.")
