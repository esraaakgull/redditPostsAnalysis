import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2 labels for binary sentiment analysis

# Prepare data
texts = ["I love this product!", "This is terrible."]
labels = [1, 0]  # Assuming 1 is positive sentiment and 0 is negative sentiment

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
inputs["labels"] = torch.tensor(labels)

# Fine-tuning (optional)
# Train the model using a suitable optimizer and loss function

# Predict sentiment
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1)

# Map predicted labels back to sentiments
predicted_sentiments = ["Positive" if label == 1 else "Negative" for label in predicted_labels]

# Print results
for text, sentiment in zip(texts, predicted_sentiments):
    print(f"Text: {text}")
    print(f"Predicted Sentiment: {sentiment}")
    print()
