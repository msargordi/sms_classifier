import os
import gc
import json
import torch
import time
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_cosine_schedule_with_warmup
from peft import PeftModel
from llm2vec import LLM2Vec
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
import shutil
import time
import psutil

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train a multitask classifier with LLM2Vec')
parser.add_argument('--max_data_points', type=int, default=None, 
                    help='Maximum number of conversations to use for training (default: use all)')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size for training (default: 4)')
parser.add_argument('--epochs', type=int, default=5,  # Changed to match RoBERTa's 5 epochs
                    help='Number of epochs to train (default: 5)')
parser.add_argument('--gradient_accumulation', type=int, default=16,  # Adjusted to get effective batch size closer to 64
                    help='Number of steps for gradient accumulation (default: 16)')
args = parser.parse_args()

# Set constants for training
MODELS_DIR = "models"
BASE_MODEL_NAME = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUPERVISED_MODEL_NAME = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
LOCAL_BASE_DIR = os.path.join(MODELS_DIR, "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
LOCAL_SUPERVISED_DIR = os.path.join(MODELS_DIR, "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")

# Training hyperparameters
MAX_LEN = 256
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation
EPOCHS = args.epochs
LEARNING_RATE = 2e-5
WARMUP_STEPS = 0.1  # 10% of total steps will be warmup
MAX_MEMORY_GB = 32  # Maximum memory to use (leave some for system)
MAX_DATA_POINTS = args.max_data_points  # Maximum number of conversations to use

# Device configuration optimized for M3
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Class mappings for the tasks
SENTIMENT_CLASSES = {"Positive": 0, "Neutral": 1, "Negative": 2}
RESPONSE_NEEDED_CLASSES = {"No": 0, "Yes": 1}
CRISIS_CLASSES = {"No Crisis": 0, "Crisis": 1}

# Loss weights to handle class imbalance
SENTIMENT_WEIGHTS = torch.tensor([1.0, 1.0, 1.0])
RESPONSE_NEEDED_WEIGHTS = torch.tensor([1.0, 1.0])
CRISIS_WEIGHTS = torch.tensor([1.0, 3.0])  # Much higher weight for the Crisis class

# Task importance weights
TASK_WEIGHTS = {
    "sentiment": 1.0,
    "response_needed": 1.0,
    "crisis": 2.0  # Higher importance for crisis detection
}

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / (1024 * 1024 * 1024)
    return memory_gb

def check_memory_usage(threshold_gb=32):
    """Check if memory usage is approaching threshold"""
    memory_gb = get_memory_usage()
    if memory_gb > threshold_gb:
        print(f"⚠️ WARNING: Memory usage ({memory_gb:.2f} GB) exceeding threshold ({threshold_gb} GB)")
        print("Running garbage collection...")
        gc.collect()
        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        memory_gb_after = get_memory_usage()
        print(f"Memory after cleanup: {memory_gb_after:.2f} GB")
    return memory_gb

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
            text = f"{supportive_msg} [SEP] {recipient_reply}"
            # Store as dict for better handling in the model
            text_parts = {
                "history": None,  # No history for first turn
                "current": text
            }
        else:
            # Second turn - include history and current separately
            supportive_msg1 = turns[0]["support_message"]
            recipient_reply1 = turns[0]["recipient_reply"]
            history_text = f"{supportive_msg1} [SEP] {recipient_reply1}"
            
            supportive_msg2 = turns[1]["support_message"]
            recipient_reply2 = turns[1]["recipient_reply"]
            current_text = f"{supportive_msg2} [SEP] {recipient_reply2}"
            
            text_parts = {
                "history": history_text,
                "current": current_text
            }
            # Combined text (only for compatibility with old code)
            text = f"{history_text} [SEP] {current_text}"
        
        # Get the labels
        sentiment = SENTIMENT_CLASSES[turns[turn_idx]["labels"]["sentiment"]]
        response_needed = RESPONSE_NEEDED_CLASSES[turns[turn_idx]["labels"]["response_needed"]]
        crisis = CRISIS_CLASSES[turns[turn_idx]["labels"]["crisis_detection"]]
        
        # Create item
        item = {
            'text': text,
            'text_parts': text_parts,
            'sentiment': torch.tensor(sentiment, dtype=torch.long),
            'response_needed': torch.tensor(response_needed, dtype=torch.long),
            'crisis': torch.tensor(crisis, dtype=torch.long),
            'turn_idx': torch.tensor(turn_idx, dtype=torch.long)
        }
        
        return item

class MultitaskClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=4096, n_sentiment_classes=3, n_response_classes=2, n_crisis_classes=2):
        super(MultitaskClassifier, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.3)
        
        # Add attention mechanism for history vs current content (like RoBERTa version)
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 2)  # 2 weights: one for history, one for current
        )
        
        # Task-specific heads
        self.sentiment_classifier = nn.Linear(hidden_size, n_sentiment_classes)
        self.response_needed_classifier = nn.Linear(hidden_size, n_response_classes)
        self.crisis_classifier = nn.Linear(hidden_size, n_crisis_classes)
        
    def forward(self, text_parts_batch, batch_size=4):
        """Process texts using text_parts format to avoid tokenizer issues"""
        batch_size = len(text_parts_batch)
        history_embeddings = []
        current_embeddings = []
        
        # Process in smaller sub-batches for memory efficiency
        with torch.no_grad():  # Keep encoder frozen
            # Process history parts
            history_texts = []
            for i in range(batch_size):
                if text_parts_batch[i]["history"] is not None:
                    history_texts.append(text_parts_batch[i]["history"])
                else:
                    # For turns without history, use an empty string placeholder
                    history_texts.append("")
            
            # Process history in sub-batches
            for i in range(0, len(history_texts), batch_size):
                sub_batch = history_texts[i:i+batch_size]
                embeddings = self.encoder.encode(sub_batch)
                history_embeddings.extend([torch.tensor(emb) for emb in embeddings])
            
            # Process current parts
            current_texts = [text_parts_batch[i]["current"] for i in range(batch_size)]
            
            # Process current in sub-batches
            for i in range(0, len(current_texts), batch_size):
                sub_batch = current_texts[i:i+batch_size]
                embeddings = self.encoder.encode(sub_batch)
                current_embeddings.extend([torch.tensor(emb) for emb in embeddings])
        
        # Stack embeddings and move to device
        history_rep = torch.stack(history_embeddings).to(device)
        current_rep = torch.stack(current_embeddings).to(device)
        
        # For first turns (with no real history), use the current embedding
        for i in range(batch_size):
            if text_parts_batch[i]["history"] is None or text_parts_batch[i]["history"] == "":
                history_rep[i] = current_rep[i]
        
        # Apply attention mechanism to weight history vs current message
        # First concatenate for feature extraction
        concat_features = torch.cat([history_rep, current_rep], dim=1)
        
        # Generate attention scores and weights
        attention_scores = self.attention_layer(concat_features)  # [batch_size, 2]
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, 2]
        
        # Apply attention weights
        history_weighted = history_rep * attention_weights[:, 0].unsqueeze(1)
        current_weighted = current_rep * attention_weights[:, 1].unsqueeze(1)
        
        # Combine with attention weights
        combined_rep = history_weighted + current_weighted
        
        # Apply dropout to the combined representation
        pooled_output = self.dropout(combined_rep)
        
        # Task-specific predictions
        sentiment_logits = self.sentiment_classifier(pooled_output)
        response_needed_logits = self.response_needed_classifier(pooled_output)
        crisis_logits = self.crisis_classifier(pooled_output)
        
        return sentiment_logits, response_needed_logits, crisis_logits, attention_weights

def collate_batch(batch):
    """Memory-efficient collate function to collect texts along with tensors"""
    # Only collect necessary fields
    texts = [item['text'] for item in batch]
    text_parts = [item['text_parts'] for item in batch]
    sentiment = torch.stack([item['sentiment'] for item in batch])
    response_needed = torch.stack([item['response_needed'] for item in batch])
    crisis = torch.stack([item['crisis'] for item in batch])
    turn_idx = torch.stack([item['turn_idx'] for item in batch])
    
    return {
        'texts': texts,
        'text_parts': text_parts,
        'sentiment': sentiment,
        'response_needed': response_needed,
        'crisis': crisis,
        'turn_idx': turn_idx
    }

def load_data(file_path, max_conversations=None):
    """Load data from a JSON file with an optional limit on the number of conversations."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Limit the number of conversations if specified
    if max_conversations is not None and max_conversations > 0:
        data = data[:max_conversations]
    
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

def verify_directory_contents(directory):
    """Verify that a directory contains expected model files"""
    # For the supervised model directory, we only need the adapter files
    if "supervised" in directory:
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
    else:
        required_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(directory, f))]
    
    if missing_files:
        print(f"Warning: Missing required files in {directory}: {missing_files}")
        return False
    return True

def load_llm2vec_model():
    """Load the LLM2Vec model from local storage"""
    print("\n=== Loading LLM2Vec model from local storage ===")
    memory_before = get_memory_usage()
    print(f"Memory before loading: {memory_before:.2f} GB")
    
    # Verify model directories exist
    if not os.path.exists(LOCAL_BASE_DIR):
        raise FileNotFoundError(f"Base model directory not found at {LOCAL_BASE_DIR}. Cannot proceed.")
    
    if not os.path.exists(LOCAL_SUPERVISED_DIR):
        raise FileNotFoundError(f"Supervised model directory not found at {LOCAL_SUPERVISED_DIR}. Cannot proceed.")
    
    # Load tokenizer and config
    print("Loading tokenizer and configuration...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_BASE_DIR)
    config = AutoConfig.from_pretrained(LOCAL_BASE_DIR, trust_remote_code=True)
    
    # Do NOT add special tokens - we'll handle the text splitting differently
    
    # Load base model
    print("Loading base model weights...")
    start_time = time.time()
    
    # Set torch memory format to preserve memory
    if hasattr(torch._C, '_jit_set_profiling_executor'):
        torch._C._jit_set_profiling_executor(False)
    if hasattr(torch._C, '_jit_set_profiling_mode'):
        torch._C._jit_set_profiling_mode(False)
    
    # Load model with memory-efficient settings
    model = AutoModel.from_pretrained(
        LOCAL_BASE_DIR,
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # First load to CPU to avoid memory issues with MPS
        low_cpu_mem_usage=True,  # Use memory-efficient loading
    )
    print(f"Base model loaded in {time.time() - start_time:.2f} seconds")
    memory_after_base = get_memory_usage()
    print(f"Memory after base model: {memory_after_base:.2f} GB (added {memory_after_base - memory_before:.2f} GB)")
    
    # Load MNTP LoRA weights and merge
    print("Loading and merging MNTP weights (this may take several minutes)...")
    start_time = time.time()
    model = PeftModel.from_pretrained(model, LOCAL_BASE_DIR)
    model = model.merge_and_unload()
    print(f"MNTP weights merged in {time.time() - start_time:.2f} seconds")
    memory_after_mntp = get_memory_usage()
    print(f"Memory after MNTP: {memory_after_mntp:.2f} GB (added {memory_after_mntp - memory_after_base:.2f} GB)")
    
    # Force garbage collection
    gc.collect()
    torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load supervised LoRA weights
    print("Loading supervised weights...")
    start_time = time.time()
    model = PeftModel.from_pretrained(model, LOCAL_SUPERVISED_DIR)
    print(f"Supervised weights loaded in {time.time() - start_time:.2f} seconds")
    memory_after_supervised = get_memory_usage()
    print(f"Memory after supervised: {memory_after_supervised:.2f} GB (added {memory_after_supervised - memory_after_mntp:.2f} GB)")
    
    # Create LLM2Vec wrapper
    print("Creating LLM2Vec wrapper...")
    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
    
    # Move to device if using MPS
    if device.type == "mps":
        print(f"Moving model to {device}...")
        l2v.model = l2v.model.to(device)
    
    memory_after_device = get_memory_usage()
    print(f"Final memory usage: {memory_after_device:.2f} GB")
    
    print("=== Model loading complete ===\n")
    return l2v, tokenizer

def train_model(model, train_dataloader, val_dataloader, epochs=EPOCHS):
    """Train the model with memory optimizations and gradient accumulation"""
    print("\n=== Beginning model training ===")
    # Move model to the device
    model = model.to(device)
    
    # Explicitly make sure the encoder is frozen (parameters won't receive gradients)
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Check which parts of the model are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
    print(f"Total parameters: {total_params:,}")
    
    # Define optimizers and loss functions
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    
    # Create the learning rate scheduler with cosine warmup
    total_steps = len(train_dataloader) * epochs // GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * WARMUP_STEPS)
    
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
        
        # Reset optimizer at the start of each epoch
        optimizer.zero_grad()
        
        # Track gradients for accumulation
        accumulated_batches = 0
        total_batches = len(train_dataloader)
        
        # Create a single progress bar for the entire epoch
        with tqdm(total=total_batches, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch_idx, batch in enumerate(train_dataloader):
                # Only check memory occasionally to reduce output
                if batch_idx % 20 == 0:
                    memory_usage = check_memory_usage(threshold_gb=MAX_MEMORY_GB)
                
                # Get inputs
                text_parts = batch['text_parts']
                sentiment_labels = batch['sentiment'].to(device)
                response_needed_labels = batch['response_needed'].to(device)
                crisis_labels = batch['crisis'].to(device)
                
                # Process in smaller sub-batches if needed
                sub_batch_size = min(2, len(text_parts))  # Process at most 2 samples at once
                sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
                    text_parts_batch=text_parts, 
                    batch_size=sub_batch_size
                )
                
                # Track attention weights
                avg_attention_weights += attention_weights.mean(dim=0).detach()
                num_batches += 1
                
                # Calculate losses
                loss_sentiment = criterion_sentiment(sentiment_logits, sentiment_labels)
                loss_response = criterion_response(response_needed_logits, response_needed_labels)
                loss_crisis = criterion_crisis(crisis_logits, crisis_labels)
                
                # Weighted sum of losses based on task importance
                loss = (TASK_WEIGHTS["sentiment"] * loss_sentiment + 
                        TASK_WEIGHTS["response_needed"] * loss_response + 
                        TASK_WEIGHTS["crisis"] * loss_crisis)
                
                # Normalize loss for gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS
                
                # Backward pass
                loss.backward()
                
                # Update weights only after accumulating gradients
                accumulated_batches += 1
                if accumulated_batches == GRADIENT_ACCUMULATION_STEPS:
                    # Apply gradient clipping to prevent memory spikes
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accumulated_batches = 0
                    
                    # Clear cache occasionally
                    if batch_idx % 50 == 0:
                        if device.type == "mps":
                            if hasattr(torch.mps, 'empty_cache'):
                                torch.mps.empty_cache()
                        elif device.type == "cuda":
                            torch.cuda.empty_cache()
                
                train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                
                # Update progress bar with current loss and completion percentage
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}"
                })
        
        # Handle any remaining gradients
        if accumulated_batches > 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / len(train_dataloader)
        avg_attention_weights /= num_batches
        
        print(f"Average training loss: {avg_train_loss:.4f}")
        print(f"Average attention weights: History={avg_attention_weights[0]:.4f}, Current={avg_attention_weights[1]:.4f}")
        
        # Force garbage collection before validation
        gc.collect()
        torch.mps.empty_cache() if hasattr(torch.mps, 'empty_cache') else None
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Validation phase
        memory_usage = get_memory_usage()
        print(f"Memory before validation: {memory_usage:.2f} GB")
        
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
        
        # Use a single progress bar for validation
        print("Running validation...")
        with tqdm(total=len(val_dataloader), desc="Validation") as pbar:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_dataloader):
                    # Get inputs
                    text_parts = batch['text_parts']
                    
                    # Forward pass in smaller sub-batches
                    sub_batch_size = min(2, len(text_parts))
                    sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
                        text_parts_batch=text_parts,
                        batch_size=sub_batch_size
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
                    
                    pbar.update(1)
                    
                    # Explicitly release GPU memory
                    del sentiment_logits, response_needed_logits, crisis_logits
                    del sentiment_preds, response_needed_preds, crisis_probs, crisis_preds
                    if device.type == "mps" and batch_idx % 5 == 0:
                        if hasattr(torch.mps, 'empty_cache'):
                            torch.mps.empty_cache()
                    elif device.type == "cuda" and batch_idx % 5 == 0:
                        torch.cuda.empty_cache()
        
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
            # Save full model state including attention mechanism
            best_model_state = {
                'attention_layer': model.attention_layer.state_dict(),
                'sentiment_classifier': model.sentiment_classifier.state_dict(),
                'response_needed_classifier': model.response_needed_classifier.state_dict(),
                'crisis_classifier': model.crisis_classifier.state_dict(),
            }
            print(f"  New best crisis recall: {best_val_crisis_recall:.4f}")
        
        # Report memory usage
        memory_usage = get_memory_usage()
        print(f"Memory after validation: {memory_usage:.2f} GB")
    
    # Load best model
    if best_model_state:
        model.attention_layer.load_state_dict(best_model_state['attention_layer'])
        model.sentiment_classifier.load_state_dict(best_model_state['sentiment_classifier'])
        model.response_needed_classifier.load_state_dict(best_model_state['response_needed_classifier'])
        model.crisis_classifier.load_state_dict(best_model_state['crisis_classifier'])
        print(f"Loaded best model with crisis recall: {best_val_crisis_recall:.4f}")
    
    print("=== Model training complete ===\n")
    return model

def evaluate_model(model, test_dataloader):
    """Evaluate the model on test data with memory optimizations"""
    print("\n=== Evaluating model on test data ===")
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
    
    # Use a single progress bar for test evaluation
    print("Running evaluation...")
    with tqdm(total=len(test_dataloader), desc="Testing") as pbar:
        with torch.no_grad():
            for batch in test_dataloader:
                # Get inputs
                text_parts = batch['text_parts']
                
                # Forward pass in sub-batches
                sub_batch_size = 2
                sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
                    text_parts_batch=text_parts,
                    batch_size=sub_batch_size
                )
                
                # Store attention weights
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
                
                pbar.update(1)
                
                # Explicitly release memory occasionally
                del sentiment_logits, response_needed_logits, crisis_logits
                del sentiment_preds, response_needed_preds, crisis_probs, crisis_preds
    
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
    
    print("=== Model evaluation complete ===\n")
    return metrics

def save_model(model, output_path):
    """Save model components including attention mechanism"""
    print(f"\n=== Saving model to {output_path} ===")
    os.makedirs(output_path, exist_ok=True)
    
    # Save classifier heads and attention layer
    classifier_state = {
        'attention_layer': model.attention_layer.state_dict(),
        'sentiment_classifier': model.sentiment_classifier.state_dict(),
        'response_needed_classifier': model.response_needed_classifier.state_dict(),
        'crisis_classifier': model.crisis_classifier.state_dict(),
    }
    
    torch.save(classifier_state, os.path.join(output_path, "classifier_heads.pt"))
    
    # Save model architecture information
    model_info = {
        "hidden_size": model.sentiment_classifier.in_features,
        "n_sentiment_classes": model.sentiment_classifier.out_features,
        "n_response_classes": model.response_needed_classifier.out_features,
        "n_crisis_classes": model.crisis_classifier.out_features,
    }
    
    with open(os.path.join(output_path, "model_info.json"), "w") as f:
        json.dump(model_info, f)
    
    print(f"Model components saved to {output_path}")
    print("=== Model saving complete ===\n")

def predict_single_message(model, support_message, recipient_reply, history=None):
    """
    Make predictions for a single message
    
    Args:
        model: Trained MultitaskClassifier
        support_message: Current supportive message
        recipient_reply: Current recipient reply
        history: Optional tuple of (prev_support_msg, prev_recipient_reply)
        
    Returns:
        Dictionary with predictions for all three tasks
    """
    model.eval()
    
    # Prepare input text based on whether history is provided
    current_text = f"{support_message} [SEP] {recipient_reply}"
    if history:
        prev_support_msg, prev_recipient_reply = history
        history_text = f"{prev_support_msg} [SEP] {prev_recipient_reply}"
        text_parts = {
            "history": history_text,
            "current": current_text
        }
    else:
        text_parts = {
            "history": None,
            "current": current_text
        }
    
    # Process with model
    with torch.no_grad():
        sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
            text_parts_batch=[text_parts],
            batch_size=1
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
    print("\n===== GRIEF SUPPORT MULTITASK CLASSIFICATION WITH LLM2VEC =====")
    print(f"Training on device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Gradient accumulation steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"Epochs: {EPOCHS}")
    
    if MAX_DATA_POINTS:
        print(f"Training on limited dataset: {MAX_DATA_POINTS} conversations")
    else:
        print("Training on full dataset")
    
    # Monitor memory usage
    start_memory = get_memory_usage()
    print(f"Initial memory usage: {start_memory:.2f} GB")
    
    try:
        # Import psutil for memory monitoring
        import psutil
    except ImportError:
        print("psutil not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "psutil"])
        import psutil
    
    # Load the LLM2Vec model from local storage
    llm2vec_model, tokenizer = load_llm2vec_model()
    
    # Define the embedding dimension from the LLM2Vec model
    embedding_dim = llm2vec_model.model.config.hidden_size
    print(f"Embedding dimension from LLM2Vec model: {embedding_dim}")
    
    # Load and prepare data
    print("\n=== Loading and preparing data ===")
    data_dir = "grief_support_dataset"  # Update this to your data directory
    try:
        train_data = load_data(os.path.join(data_dir, "train.json"), MAX_DATA_POINTS)
        val_data = load_data(os.path.join(data_dir, "validation.json"))
        test_data = load_data(os.path.join(data_dir, "test.json"))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Data files not found in {data_dir}. Please make sure the dataset is in the correct location.")
        return
    
    print(f"Train data: {len(train_data)} conversations")
    print(f"Validation data: {len(val_data)} conversations")
    print(f"Test data: {len(test_data)} conversations")
    
    # Create datasets and dataloaders
    print("Creating datasets and dataloaders...")
    train_dataset = GriefDataset(train_data, tokenizer, MAX_LEN)
    val_dataset = GriefDataset(val_data, tokenizer, MAX_LEN)
    test_dataset = GriefDataset(test_data, tokenizer, MAX_LEN)
    
    # Use the memory-efficient collate function
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_batch,
        pin_memory=False  # Don't pin memory for MPS
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_batch,
        pin_memory=False
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_batch,
        pin_memory=False
    )
    
    print("=== Data preparation complete ===\n")
    
    # Initialize the classifier model
    print("\n=== Initializing classifier model ===")
    classifier_model = MultitaskClassifier(
        encoder=llm2vec_model,
        hidden_size=embedding_dim
    )
    print("Classifier model initialized with frozen encoder")
    print("=== Model initialization complete ===\n")
    
    # Train the model
    start_time = time.time()
    classifier_model = train_model(
        classifier_model, 
        train_dataloader, 
        val_dataloader, 
        epochs=EPOCHS
    )
    elapsed_time = time.time() - start_time
    print(f"Training and evaluation took {elapsed_time/3600:.2f} hours")

    # Evaluate the model
    metrics = evaluate_model(classifier_model, test_dataloader)
    
    # Save the model
    output_dir = "grief_model_llm2vec_improved"
    if MAX_DATA_POINTS:
        output_dir = f"{output_dir}_{MAX_DATA_POINTS}_samples"
    save_model(classifier_model, output_dir)
    
    # Final memory report
    final_memory = get_memory_usage()
    print(f"\nFinal memory usage: {final_memory:.2f} GB (change: {final_memory - start_memory:.2f} GB)")
    
    print("\n===== COMPLETE =====")
    print("The model has been successfully trained, evaluated, and saved.")
    print(f"Model saved to: {output_dir}")
    print("You can now use the saved model for inference without needing to re-download the base model.")

if __name__ == "__main__":
    # Check if MPS is available for Mac M-series chips
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available - using Apple Silicon acceleration")
    else:
        print("MPS not available - falling back to CPU")
    
    main()