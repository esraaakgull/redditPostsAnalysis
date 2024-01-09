import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
num_labels = 3  # Adjust based on your sentiment classes
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(model_name)

# Prepare your training data
# Make sure to organize your data as input texts and corresponding sentiment labels
texts = ["I love this movie!", "This is terrible."]
labels = [1, 0]  # Example labels: 1 for positive, 0 for negative

# Tokenize input texts
encoded_texts = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

# Convert labels to tensors
label_tensors = torch.tensor(labels)

# Create a dataset
from torch.utils.data import TensorDataset

dataset = TensorDataset(encoded_texts['input_ids'], encoded_texts['attention_mask'], label_tensors)

# Create a data loader
from torch.utils.data import DataLoader

batch_size = 2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 3  # Adjust as needed
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, label = batch
        outputs = model(input_ids, attention_mask=attention_mask)[0]
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

# Inference on new texts
new_texts = ["This is great!", "I'm not happy with this.", "I dont like you'"]
encoded_new_texts = tokenizer(new_texts, padding=True, truncation=True, return_tensors='pt')
with torch.no_grad():
    predictions = model(encoded_new_texts['input_ids'], attention_mask=encoded_new_texts['attention_mask'])[0]
predicted_labels = predictions.argmax(dim=1)

# Convert predicted labels to sentiment labels
sentiment_labels = ['Negative', 'Neutral', 'Positive']
predicted_sentiments = [sentiment_labels[label] for label in predicted_labels]

# Print results
for text, sentiment in zip(new_texts, predicted_sentiments):
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}\n")
