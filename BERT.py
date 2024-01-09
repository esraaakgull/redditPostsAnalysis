import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup

# Step 1: Installing required libraries
# pip install transformers
# pip install torch

# Step 2: Preparing our dataset (mock data)
train_text_list = [
    "I love this product!",
    "This movie is amazing.",
    "The food was delicious.",
    "I'm happy with the service.",
    "The weather is perfect.",
    "Chocolate was great",
    "That color is nice",
    "Perfect place that i have ever seen",
    "This book is terrible.",
    "The customer support was terrible.",
    "I hate this place.",
    "The experience was awful.",
    "This is the worst product ever.",
    "She is a bad girl",
    "I do not like you",
    "It is disgusting",
]

train_labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative

val_text_list = [
    "The hotel was nice.",
    "The staff was friendly.",
    "The coffee tasted bad.",
    "The app is user-friendly.",
    "The design is outdated.",
    "Taste of cake is terrible"
]

val_labels = [1, 1, 0, 1, 0, 0]

test_text_list = [
    "The concert was fantastic!",
    "The package arrived on time.",
    "The interface is confusing.",
    "The movie was a waste of time."
]

test_labels = [1, 1, 0, 0]

num_classes = 2  # Number of sentiment classes (positive and negative)

# Step 3: Loading Pre-trained BERT Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Step 4: Data Preprocessing
encoded_data_train = tokenizer.batch_encode_plus(
    train_text_list,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt'
)

encoded_data_val = tokenizer.batch_encode_plus(
    val_text_list,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt'
)

encoded_data_test = tokenizer.batch_encode_plus(
    test_text_list,
    add_special_tokens=True,
    return_attention_mask=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt'
)

input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(train_labels)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(val_labels)

input_ids_test = encoded_data_test['input_ids']
attention_masks_test = encoded_data_test['attention_mask']
labels_test = torch.tensor(test_labels)

train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

val_dataset = TensorDataset(input_ids_val, attention_masks_val, labels_val)
val_loader = DataLoader(val_dataset, batch_size=16)

test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)
test_loader = DataLoader(test_dataset, batch_size=16)

# Step 6: Defining Sentiment Analysis Model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Step 7: Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)

num_epochs = 3

# Enabling fine-tuning
model.train()

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) // 10,
    num_training_steps=len(train_loader) * num_epochs
)

for epoch in range(num_epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(inputs, attention_mask=masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Step 8: Evaluation
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for batch in val_loader:
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"Validation Accuracy: {accuracy}")

# Step 9: Testing
model.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for batch in test_loader:
        inputs = batch[0].to(device)
        masks = batch[1].to(device)
        labels = batch[2].to(device)
        outputs = model(inputs, attention_mask=masks)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        total_correct += (predicted_labels == labels).sum().item()
        total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"Test Accuracy: {accuracy}")

# Step 10: Inference
text_to_classify = 'I took two years of Java and wanted to kill myself'
encoded_text = tokenizer.encode_plus(
    text_to_classify,
    add_special_tokens=True,
    return_tensors='pt'
)
input_ids = encoded_text['input_ids'].to(device)
attention_mask = encoded_text['attention_mask'].to(device)

model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    print(f"Predicted Sentiment Class: {predicted_class}")
