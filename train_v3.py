import os
import json
import torch
import time
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import pandas as pd

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5
MODEL_NAME = "roberta-base"
WARMUP_STEPS = 0.1  # 10% of total steps will be warmup

# Class mappings for the three tasks
SENTIMENT_CLASSES = {"Positive": 0, "Neutral": 1, "Negative": 2}
RESPONSE_NEEDED_CLASSES = {"No": 0, "Yes": 1}
CRISIS_CLASSES = {"No Crisis": 0, "Crisis": 1}

# Loss weights to handle class imbalance
# Higher weight for crisis class to prioritize recall
SENTIMENT_WEIGHTS = torch.tensor([1.0, 1.0, 1.0])
RESPONSE_NEEDED_WEIGHTS = torch.tensor([1.0, 1.0])
CRISIS_WEIGHTS = torch.tensor([1.0, 3.0])  # Much higher weight for the Crisis class

# Task importance weights
TASK_WEIGHTS = {
    "sentiment": 1.0,
    "response_needed": 1.0,
    "crisis": 2.0  # Higher importance for crisis detection
}

class GriefDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_len):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.conversations) * 2  # Each conversation has 2 turns
        
    def __getitem__(self, idx):
        # Determine which conversation and which turn
        conv_idx = idx // 2
        turn_idx = idx % 2
        
        conversation = self.conversations[conv_idx]
        turns = conversation["turns"]
        
        # For the first turn, we only have the first reply
        # For the second turn, we include the history
        if turn_idx == 0:
            # First turn - no history
            supportive_msg = turns[0]["support_message"]
            recipient_reply = turns[0]["recipient_reply"]
            # No history before first turn, so no [CURRENT] marker needed
            text = f"{supportive_msg} [SEP] {recipient_reply}"
        else:
            # Second turn - include history with [CURRENT] marker
            supportive_msg1 = turns[0]["support_message"]
            recipient_reply1 = turns[0]["recipient_reply"]
            supportive_msg2 = turns[1]["support_message"]
            recipient_reply2 = turns[1]["recipient_reply"]
            # Add [CURRENT] to indicate where history ends and current message begins
            text = f"{supportive_msg1} [SEP] {recipient_reply1} [CURRENT] {supportive_msg2} [SEP] {recipient_reply2}"
        
        # Get the labels
        sentiment = SENTIMENT_CLASSES[turns[turn_idx]["labels"]["sentiment"]]
        response_needed = RESPONSE_NEEDED_CLASSES[turns[turn_idx]["labels"]["response_needed"]]
        crisis = CRISIS_CLASSES[turns[turn_idx]["labels"]["crisis_detection"]]
        
        # Tokenize
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'response_needed': torch.tensor(response_needed, dtype=torch.long),
            'crisis': torch.tensor(crisis, dtype=torch.long),
            'turn_idx': torch.tensor(turn_idx, dtype=torch.long)  # Store turn index for analysis
        }

class MultitaskRobertaClassifier(nn.Module):
    def __init__(self, tokenizer, n_sentiment_classes=3, n_response_classes=2, n_crisis_classes=2):
        super(MultitaskRobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.tokenizer = tokenizer
        
        # Get the ID for the [CURRENT] token
        self.current_token_id = tokenizer.convert_tokens_to_ids("[CURRENT]")
        
        # Hidden size from RoBERTa
        hidden_size = self.roberta.config.hidden_size
        
        # Attention mechanism for weighting history vs current message
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # 2 weights: one for history, one for current
        )
        
        # Task-specific heads will now take the hidden size (not doubled) due to attention mechanism
        self.sentiment_classifier = nn.Linear(hidden_size, n_sentiment_classes)
        self.response_needed_classifier = nn.Linear(hidden_size, n_response_classes)
        self.crisis_classifier = nn.Linear(hidden_size, n_crisis_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        batch_size = input_ids.shape[0]
        
        # CLS token for history context
        history_rep = sequence_output[:, 0, :]
        
        # Find position of [CURRENT] token for each sample in batch
        current_reply_reps = []
        for i in range(batch_size):
            # Find where [CURRENT] token is
            current_pos = (input_ids[i] == self.current_token_id).nonzero(as_tuple=True)[0]
            
            if len(current_pos) > 0:  # Turn 2 with history
                # Get the position of [CURRENT]
                current_pos = current_pos[0]
                # Get all tokens after [CURRENT]
                reply_vectors = sequence_output[i, current_pos+1:, :]
                # Mean pooling of these tokens
                if reply_vectors.size(0) > 0:  # Make sure there are tokens after [CURRENT]
                    reply_rep = torch.mean(reply_vectors, dim=0)
                else:  # Fallback if there are no tokens after [CURRENT]
                    reply_rep = sequence_output[i, -1, :]  # Last token
            else:  # Turn 1 (no history)
                # For Turn 1, use mean of all tokens except CLS
                reply_vectors = sequence_output[i, 1:, :]
                reply_rep = torch.mean(reply_vectors, dim=0)
            
            current_reply_reps.append(reply_rep)
        
        # Stack individual reply representations
        current_reply_rep = torch.stack(current_reply_reps)
        
        # Apply attention mechanism to weight history vs current message
        # First concatenate for feature extraction
        concat_features = torch.cat([history_rep, current_reply_rep], dim=1)
        
        # Generate attention scores and weights
        attention_scores = self.attention_layer(concat_features)  # [batch_size, 2]
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, 2]
        
        # Apply attention weights
        history_weighted = history_rep * attention_weights[:, 0].unsqueeze(1)
        current_weighted = current_reply_rep * attention_weights[:, 1].unsqueeze(1)
        
        # Combine with attention weights
        combined_rep = history_weighted + current_weighted
        
        # Apply dropout
        pooled_output = self.dropout(combined_rep)
        
        # Task-specific predictions
        sentiment_logits = self.sentiment_classifier(pooled_output)
        response_needed_logits = self.response_needed_classifier(pooled_output)
        crisis_logits = self.crisis_classifier(pooled_output)
        
        return sentiment_logits, response_needed_logits, crisis_logits, attention_weights

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def compute_metrics(preds, labels):
    # For multi-class (sentiment)
    sentiment_accuracy = accuracy_score(labels["sentiment"], preds["sentiment"])
    sentiment_f1 = f1_score(labels["sentiment"], preds["sentiment"], average='weighted')
    
    # For binary (response_needed)
    response_accuracy = accuracy_score(labels["response_needed"], preds["response_needed"])
    response_f1 = f1_score(labels["response_needed"], preds["response_needed"])
    
    # For binary (crisis) - focus on recall
    crisis_accuracy = accuracy_score(labels["crisis"], preds["crisis"])
    crisis_precision = precision_score(labels["crisis"], preds["crisis"])
    crisis_recall = recall_score(labels["crisis"], preds["crisis"])
    crisis_f1 = f1_score(labels["crisis"], preds["crisis"])
    
    metrics = {
        "sentiment_accuracy": sentiment_accuracy,
        "sentiment_f1": sentiment_f1,
        "response_accuracy": response_accuracy,
        "response_f1": response_f1,
        "crisis_accuracy": crisis_accuracy,
        "crisis_precision": crisis_precision,
        "crisis_recall": crisis_recall,
        "crisis_f1": crisis_f1
    }
    
    return metrics

def train_model(model, train_dataloader, val_dataloader, epochs=EPOCHS):
    # Move model to the device
    model.to(device)
    
    # Define optimizers and loss functions
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Create the learning rate scheduler with cosine warmup
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * WARMUP_STEPS)  # Convert percentage to steps
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Define loss functions with class weights
    sentiment_weights = SENTIMENT_WEIGHTS.to(device)
    response_needed_weights = RESPONSE_NEEDED_WEIGHTS.to(device)
    crisis_weights = CRISIS_WEIGHTS.to(device)
    
    criterion_sentiment = nn.CrossEntropyLoss(weight=sentiment_weights)
    criterion_response = nn.CrossEntropyLoss(weight=response_needed_weights)
    criterion_crisis = nn.CrossEntropyLoss(weight=crisis_weights)
    
    # Initialize best metrics
    best_val_crisis_recall = 0.0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        avg_attention_weights = torch.zeros(2).to(device)
        num_batches = 0
        
        for batch in train_dataloader:
            # Get inputs
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['sentiment'].to(device)
            response_needed_labels = batch['response_needed'].to(device)
            crisis_labels = batch['crisis'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Calculate losses
            loss_sentiment = criterion_sentiment(sentiment_logits, sentiment_labels)
            loss_response = criterion_response(response_needed_logits, response_needed_labels)
            loss_crisis = criterion_crisis(crisis_logits, crisis_labels)
            
            # Weighted sum of losses based on task importance
            loss = (TASK_WEIGHTS["sentiment"] * loss_sentiment + 
                    TASK_WEIGHTS["response_needed"] * loss_response + 
                    TASK_WEIGHTS["crisis"] * loss_crisis)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track average attention weights to monitor focus on history vs current
            avg_attention_weights += attention_weights.mean(dim=0).detach()
            num_batches += 1
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_attention_weights /= num_batches
        
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average attention weights: History={avg_attention_weights[0]:.4f}, Current={avg_attention_weights[1]:.4f}")
        
        # Validation phase
        model.eval()
        val_predictions = {
            "sentiment": [],
            "response_needed": [],
            "crisis": []
        }
        val_labels = {
            "sentiment": [],
            "response_needed": [],
            "crisis": []
        }
        val_attention_weights = torch.zeros(2).to(device)
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                # Get inputs
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Track attention weights
                val_attention_weights += attention_weights.mean(dim=0)
                val_batches += 1
                
                # Get predictions - use a lower threshold for crisis detection
                sentiment_preds = torch.argmax(sentiment_logits, dim=1)
                response_needed_preds = torch.argmax(response_needed_logits, dim=1)
                
                # For crisis, use a lower threshold to favor recall
                crisis_probs = torch.softmax(crisis_logits, dim=1)
                crisis_preds = (crisis_probs[:, 1] > 0.3).long()  # Lower threshold (0.3 instead of 0.5)
                
                # Move to CPU and convert to lists
                val_predictions["sentiment"].extend(sentiment_preds.cpu().tolist())
                val_predictions["response_needed"].extend(response_needed_preds.cpu().tolist())
                val_predictions["crisis"].extend(crisis_preds.cpu().tolist())
                
                val_labels["sentiment"].extend(batch["sentiment"].cpu().tolist())
                val_labels["response_needed"].extend(batch["response_needed"].cpu().tolist())
                val_labels["crisis"].extend(batch["crisis"].cpu().tolist())
        
        # Calculate metrics
        metrics = compute_metrics(val_predictions, val_labels)
        
        # Calculate average validation attention weights
        val_attention_weights /= val_batches
        print(f"Validation attention weights: History={val_attention_weights[0]:.4f}, Current={val_attention_weights[1]:.4f}")
        
        print("Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # Save best model based on crisis recall
        if metrics["crisis_recall"] > best_val_crisis_recall:
            best_val_crisis_recall = metrics["crisis_recall"]
            best_model_state = model.state_dict().copy()
            print(f"  New best crisis recall: {best_val_crisis_recall:.4f}")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with crisis recall: {best_val_crisis_recall:.4f}")
    
    return model

def evaluate_model(model, test_dataloader):
    model.eval()
    
    test_predictions = {
        "sentiment": [],
        "response_needed": [],
        "crisis": []
    }
    test_labels = {
        "sentiment": [],
        "response_needed": [],
        "crisis": []
    }
    test_turn_indices = []
    test_attention_weights = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            # Get inputs
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Store attention weights for analysis
            test_attention_weights.append(attention_weights.cpu())
            
            # Get predictions
            sentiment_preds = torch.argmax(sentiment_logits, dim=1)
            response_needed_preds = torch.argmax(response_needed_logits, dim=1)
            
            # For crisis, use a lower threshold to favor recall
            crisis_probs = torch.softmax(crisis_logits, dim=1)
            crisis_preds = (crisis_probs[:, 1] > 0.3).long()  # Lower threshold for high recall
            
            # Move to CPU and convert to lists
            test_predictions["sentiment"].extend(sentiment_preds.cpu().tolist())
            test_predictions["response_needed"].extend(response_needed_preds.cpu().tolist())
            test_predictions["crisis"].extend(crisis_preds.cpu().tolist())
            
            test_labels["sentiment"].extend(batch["sentiment"].cpu().tolist())
            test_labels["response_needed"].extend(batch["response_needed"].cpu().tolist())
            test_labels["crisis"].extend(batch["crisis"].cpu().tolist())
            
            test_turn_indices.extend(batch["turn_idx"].cpu().tolist())
    
    # Calculate overall metrics
    metrics = compute_metrics(test_predictions, test_labels)
    
    # Combine all attention weights
    all_attention_weights = torch.cat(test_attention_weights)
    avg_attention_weights = all_attention_weights.mean(dim=0)
    
    print("\nTest Metrics (Overall):")
    print(f"Average attention weights: History={avg_attention_weights[0]:.4f}, Current={avg_attention_weights[1]:.4f}")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")
    
    # Separate metrics by turn
    turn1_indices = [i for i, turn_idx in enumerate(test_turn_indices) if turn_idx == 0]
    turn2_indices = [i for i, turn_idx in enumerate(test_turn_indices) if turn_idx == 1]
    
    for turn_name, indices in [("Turn 1 (No History)", turn1_indices), ("Turn 2 (With History)", turn2_indices)]:
        turn_predictions = {
            "sentiment": [test_predictions["sentiment"][i] for i in indices],
            "response_needed": [test_predictions["response_needed"][i] for i in indices],
            "crisis": [test_predictions["crisis"][i] for i in indices]
        }
        
        turn_labels = {
            "sentiment": [test_labels["sentiment"][i] for i in indices],
            "response_needed": [test_labels["response_needed"][i] for i in indices],
            "crisis": [test_labels["crisis"][i] for i in indices]
        }
        
        # Calculate attention weights for this turn
        turn_attention_weights = torch.stack([all_attention_weights[i] for i in indices])
        turn_avg_attention = turn_attention_weights.mean(dim=0)
        
        turn_metrics = compute_metrics(turn_predictions, turn_labels)
        
        print(f"\nTest Metrics ({turn_name}):")
        print(f"  Attention weights: History={turn_avg_attention[0]:.4f}, Current={turn_avg_attention[1]:.4f}")
        for metric_name, metric_value in turn_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    return metrics

def save_model(model, tokenizer, output_path):
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), f"{output_path}/model.pt")
    tokenizer.save_pretrained(output_path)
    print(f"Model saved to {output_path}")

def predict_single_message(model, tokenizer, support_message, recipient_reply, history=None):
    """
    Make predictions for a single message
    
    Args:
        model: Trained MultitaskRobertaClassifier
        tokenizer: RobertaTokenizer
        support_message: Current supportive message
        recipient_reply: Current recipient reply
        history: Optional tuple of (prev_support_msg, prev_recipient_reply)
        
    Returns:
        Dictionary with predictions for all three tasks
    """
    model.eval()
    
    # Prepare input text based on whether history is provided
    if history:
        prev_support_msg, prev_recipient_reply = history
        text = f"{prev_support_msg} [SEP] {prev_recipient_reply} [CURRENT] {support_message} [SEP] {recipient_reply}"
    else:
        text = f"{support_message} [SEP] {recipient_reply}"
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Convert logits to probabilities and predictions
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)
        response_needed_probs = torch.softmax(response_needed_logits, dim=1)
        crisis_probs = torch.softmax(crisis_logits, dim=1)
        
        # Get class predictions
        sentiment_pred = torch.argmax(sentiment_probs, dim=1).item()
        response_needed_pred = torch.argmax(response_needed_probs, dim=1).item()
        
        # For crisis, use lower threshold
        crisis_pred = 1 if crisis_probs[0, 1].item() > 0.3 else 0
        
        # Get confidence scores
        sentiment_confidence = sentiment_probs[0, sentiment_pred].item()
        response_needed_confidence = response_needed_probs[0, response_needed_pred].item()
        crisis_confidence = crisis_probs[0, 1].item()  # Always report crisis probability
    
    # Map numeric predictions back to class names
    sentiment_map = {v: k for k, v in SENTIMENT_CLASSES.items()}
    response_needed_map = {v: k for k, v in RESPONSE_NEEDED_CLASSES.items()}
    crisis_map = {v: k for k, v in CRISIS_CLASSES.items()}
    
    return {
        "sentiment": {
            "prediction": sentiment_map[sentiment_pred],
            "confidence": sentiment_confidence
        },
        "response_needed": {
            "prediction": response_needed_map[response_needed_pred],
            "confidence": response_needed_confidence
        },
        "crisis": {
            "prediction": crisis_map[crisis_pred],
            "confidence": crisis_confidence,
            "requires_attention": crisis_confidence > 0.2  # Flag even lower probability crises for review
        },
        "attention_weights": {
            "history": attention_weights[0, 0].item(),
            "current": attention_weights[0, 1].item()
        }
    }

def main():
    # Load and prepare data
    print("Loading data...")
    train_data = load_data("grief_support_dataset/train.json")
    val_data = load_data("grief_support_dataset/validation.json")
    test_data = load_data("grief_support_dataset/test.json")
    
    print(f"Train data: {len(train_data)} conversations")
    print(f"Validation data: {len(val_data)} conversations")
    print(f"Test data: {len(test_data)} conversations")
    
    # Initialize tokenizer and add [CURRENT] token
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[CURRENT]']})
    
    # Create datasets
    train_dataset = GriefDataset(train_data, tokenizer, MAX_LEN)
    val_dataset = GriefDataset(val_data, tokenizer, MAX_LEN)
    test_dataset = GriefDataset(test_data, tokenizer, MAX_LEN)
    
    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = MultitaskRobertaClassifier(tokenizer)
    model.roberta.resize_token_embeddings(len(tokenizer))  # Resize for new token
    
    # Train model
    print("\nTraining model...")
    start_time = time.time()
    model = train_model(model, train_dataloader, val_dataloader, epochs=EPOCHS)
    elapsed_time = time.time() - start_time
    print(f"Training and evaluation took {elapsed_time/60:.2f} minutes")
    
    # Evaluate model
    print("\nEvaluating model...")
    evaluate_model(model, test_dataloader)

    
    # Save model
    save_model(model, tokenizer, "grief_model_v3")
    
    # Example usage
    print("\nExample prediction:")
    prediction = predict_single_message(
        model, 
        tokenizer,
        "We're here to support you during this difficult time.",
        "I don't know how to keep going. Everything reminds me of him."
    )
    
    print(json.dumps(prediction, indent=2))

if __name__ == "__main__":
    main()