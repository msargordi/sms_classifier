import os
import gc
import json
import torch
import numpy as np
from torch import nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from llm2vec import LLM2Vec
import time

# Force CPU usage for more stable evaluation, but allow for MPS/CUDA if available
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Class mappings
SENTIMENT_CLASSES = {"Positive": 0, "Neutral": 1, "Negative": 2}
RESPONSE_NEEDED_CLASSES = {"No": 0, "Yes": 1}
CRISIS_CLASSES = {"No Crisis": 0, "Crisis": 1}

# Constants
MAX_LEN = 256
MODELS_DIR = "models"
BASE_MODEL_NAME = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
SUPERVISED_MODEL_NAME = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
LOCAL_BASE_DIR = os.path.join(MODELS_DIR, "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
LOCAL_SUPERVISED_DIR = os.path.join(MODELS_DIR, "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
MODEL_PATH = "grief_model_llm2vec_v2"  # Updated model path for LLM2Vec classifiers

def get_memory_usage():
    """Get current memory usage in GB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024 * 1024 * 1024)
        return memory_gb
    except ImportError:
        return 0  # Return 0 if psutil not available

class MultitaskClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=4096, n_sentiment_classes=3, n_response_classes=2, n_crisis_classes=2):
        super(MultitaskClassifier, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.3)
        
        # Add attention mechanism for history vs current content
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
        """Process texts using text_parts format with attention mechanism"""
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
                # Filter out empty strings to avoid errors
                non_empty = [text for text in sub_batch if text]
                if non_empty:
                    embeddings = self.encoder.encode(non_empty)
                    for j, text in enumerate(sub_batch):
                        if text:  # If text is not empty
                            history_embeddings.append(torch.tensor(embeddings[len(history_embeddings) - i]))
                        else:  # If text is empty, add a zero tensor
                            history_embeddings.append(torch.zeros(self.encoder.model.config.hidden_size))
                else:
                    # All empty strings, add zero tensors
                    for _ in sub_batch:
                        history_embeddings.append(torch.zeros(self.encoder.model.config.hidden_size))
            
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

def load_thresholds(model_path):
    """Load the threshold values from the saved JSON file"""
    try:
        with open(f"{model_path}/thresholds.json", 'r') as f:
            thresholds = json.load(f)
        print(f"Loaded thresholds from {model_path}/thresholds.json")
        return thresholds
    except FileNotFoundError:
        print(f"Warning: Could not find thresholds.json in {model_path}")
        # Return default thresholds
        return {
            'best_f1': {'threshold': 0.5},
            'high_recall': {'threshold': 0.3}
        }

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
        device_map="cpu",  # First load to CPU to avoid memory issues
        low_cpu_mem_usage=True,  # Use memory-efficient loading
    )
    print(f"Base model loaded in {time.time() - start_time:.2f} seconds")
    memory_after_base = get_memory_usage()
    print(f"Memory after base model: {memory_after_base:.2f} GB (added {memory_after_base - memory_before:.2f} GB)")
    
    # Load MNTP LoRA weights and merge
    print("Loading and merging MNTP weights...")
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
    
    # Move to device if using GPU/MPS
    if device.type != "cpu":
        print(f"Moving model to {device}...")
        l2v.model = l2v.model.to(device)
    
    memory_after_device = get_memory_usage()
    print(f"Final memory usage: {memory_after_device:.2f} GB")
    
    print("=== Model loading complete ===\n")
    return l2v, tokenizer

def load_model_info(model_path):
    """Load model information including hidden size from saved JSON"""
    try:
        with open(f"{model_path}/model_info.json", 'r') as f:
            model_info = json.load(f)
        print(f"Loaded model info from {model_path}/model_info.json")
        return model_info
    except FileNotFoundError:
        print(f"Warning: Could not find model_info.json in {model_path}")
        # Return default values based on LLM2Vec model
        return {
            "hidden_size": 4096,  # Default for LLM2Vec embeddings
            "n_sentiment_classes": 3,
            "n_response_classes": 2,
            "n_crisis_classes": 2
        }

def test_example(model, tokenizer, support_message, recipient_reply, thresholds, history=None):
    """
    Test a single example with attention weights
    
    Args:
        model: The MultitaskClassifier model
        tokenizer: Tokenizer (not directly used for encoding with LLM2Vec but kept for API consistency)
        support_message: Current supportive message
        recipient_reply: Current recipient reply
        thresholds: Dictionary containing threshold values
        history: Optional tuple of (prev_support_msg, prev_recipient_reply)
    """
    # Format text parts
    current_text = f"{support_message} [SEP] {recipient_reply}"
    
    # Create text_parts dict based on whether history is provided
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
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
            text_parts_batch=[text_parts], 
            batch_size=1
        )
        
        # Convert logits to probabilities
        sentiment_probs = torch.softmax(sentiment_logits, dim=1)
        response_needed_probs = torch.softmax(response_needed_logits, dim=1)
        crisis_probs = torch.softmax(crisis_logits, dim=1)
        
        # Get class predictions
        sentiment_pred = torch.argmax(sentiment_probs, dim=1).item()
        response_needed_pred = torch.argmax(response_needed_probs, dim=1).item()
        
        # Get class probabilities
        sentiment_confidence = sentiment_probs[0, sentiment_pred].item()
        response_needed_confidence = response_needed_probs[0, response_needed_pred].item()
        
        # Get crisis probability (prob of class 1 - "Crisis")
        crisis_prob = crisis_probs[0, 1].item()
        
        # Use the best_f1 threshold for crisis detection
        crisis_threshold = thresholds['best_f1']['threshold']
        high_recall_threshold = thresholds['high_recall']['threshold']
        
        # Make prediction based on best_f1 threshold
        crisis_pred = 1 if crisis_prob > crisis_threshold else 0
        
        # Get confidence for the predicted crisis class
        crisis_confidence = crisis_prob if crisis_pred == 1 else 1 - crisis_prob
        
        # Get attention weights
        history_attention = attention_weights[0, 0].item()
        current_attention = attention_weights[0, 1].item()
    
    # Map numeric predictions back to class names
    sentiment_map = {v: k for k, v in SENTIMENT_CLASSES.items()}
    response_needed_map = {v: k for k, v in RESPONSE_NEEDED_CLASSES.items()}
    crisis_map = {v: k for k, v in CRISIS_CLASSES.items()}
    
    # Display inputs
    print("\n" + "-" * 50)
    print("INPUT:")
    if history:
        print(f"Previous support message: {history[0]}")
        print(f"Previous recipient reply: {history[1]}")
    print(f"Current support message: {support_message}")
    print(f"Current recipient reply: {recipient_reply}")
    
    # Display attention weights
    print("\nATTENTION WEIGHTS:")
    if history:
        print(f"History attention: {history_attention:.4f}")
        print(f"Current attention: {current_attention:.4f}")
    else:
        print("No history present - using only current message")
    
    # Display predictions
    print("\nPREDICTIONS:")
    print(f"Sentiment: {sentiment_map[sentiment_pred]} (confidence: {sentiment_confidence:.4f})")
    print(f"Response needed: {response_needed_map[response_needed_pred]} (confidence: {response_needed_confidence:.4f})")
    print(f"Crisis detection: {crisis_map[crisis_pred]} (confidence: {crisis_confidence:.4f})")
    print(f"Crisis probability: {crisis_prob:.4f} (thresholds: best_f1={crisis_threshold:.4f}, high_recall={high_recall_threshold:.4f})")
    
    # Only flag as requiring attention if it's an actual crisis or close to threshold
    if crisis_pred == 1 or (crisis_prob > high_recall_threshold and crisis_prob <= crisis_threshold):
        print("*** REQUIRES ATTENTION ***")
    
    # Show additional info for borderline cases
    if crisis_prob > high_recall_threshold and crisis_prob <= crisis_threshold and crisis_pred == 0:
        print(f"Note: Would be flagged as crisis with high_recall threshold ({high_recall_threshold:.4f})")

def main():
    print("\n===== GRIEF SUPPORT CLASSIFIER EVALUATION (LLM2Vec) =====")
    try:
        # Import psutil for memory monitoring if available
        import psutil
    except ImportError:
        print("psutil not installed. Memory monitoring will be disabled.")
    
    # Load LLM2Vec model and tokenizer
    llm2vec_model, tokenizer = load_llm2vec_model()
    
    # Load model information
    model_info = load_model_info(MODEL_PATH)
    hidden_size = model_info["hidden_size"]
    print(f"Using hidden size: {hidden_size}")
    
    # Initialize model with encoder
    classifier_model = MultitaskClassifier(
        encoder=llm2vec_model,
        hidden_size=hidden_size,
        n_sentiment_classes=model_info.get("n_sentiment_classes", 3),
        n_response_classes=model_info.get("n_response_classes", 2),
        n_crisis_classes=model_info.get("n_crisis_classes", 2)
    )
    
    # Load the classifier heads and attention layer
    print(f"Loading classifier heads from {MODEL_PATH}/classifier_heads.pt")
    try:
        classifier_state = torch.load(f"{MODEL_PATH}/classifier_heads.pt", map_location=device)
        
        # Load the state dict for all components
        if 'attention_layer' in classifier_state:
            classifier_model.attention_layer.load_state_dict(classifier_state['attention_layer'])
            print("Loaded attention layer weights")
        else:
            print("Warning: No attention layer weights found in saved model")
            
        classifier_model.sentiment_classifier.load_state_dict(classifier_state['sentiment_classifier'])
        classifier_model.response_needed_classifier.load_state_dict(classifier_state['response_needed_classifier'])
        classifier_model.crisis_classifier.load_state_dict(classifier_state['crisis_classifier'])
        
        classifier_model.to(device)
        classifier_model.eval()
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Error: Could not find classifier_heads.pt in {MODEL_PATH}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load thresholds
    thresholds = load_thresholds(MODEL_PATH)
    print(f"Using crisis thresholds - best_f1: {thresholds['best_f1']['threshold']:.4f}, high_recall: {thresholds['high_recall']['threshold']:.4f}")
    
    print("\nTesting examples...")
    
    # Example 1: Potential crisis (without history)
    test_example(
        classifier_model, 
        tokenizer,
        "We're here to support you during this difficult time.",
        "I don't know how to keep going. Everything reminds me of him.",
        thresholds
    )
    
    # Example 2: Positive sentiment (without history)
    test_example(
        classifier_model, 
        tokenizer,
        "Remember that grief has no timeline. Be patient with yourself.",
        "Thank you. I've been trying to focus on happy memories and it's starting to help a little.",
        thresholds
    )
    
    # Example 3: Neutral sentiment with history
    test_example(
        classifier_model, 
        tokenizer,
        "Many people find that grief changes over time. How are you feeling today?",
        "It's been 6 months since my mom died. I'm not sure what to feel anymore.",
        thresholds,
        history=(
            "I'm here to listen if you need to talk.",
            "I lost my mom recently and I don't know how to cope."
        )
    )
    
    # Example 4: Clear crisis with history
    test_example(
        classifier_model, 
        tokenizer,
        "It's okay to feel overwhelmed. We're here whenever you need someone to listen.",
        "I want to kill myself.",
        thresholds,
        history=(
            "How are you dealing with things today?",
            "I've been trying to distract myself, but nothing helps."
        )
    )
    
    # Example 5: More complex crisis detection with history
    test_example(
        classifier_model, 
        tokenizer,
        "That sounds really difficult. Can you tell me more about what's been happening?",
        "I've lost my job, my apartment, and my partner left me. I keep thinking about how much better everyone would be without me.",
        thresholds,
        history=(
            "I'm here to support you. How are you feeling today?",
            "I don't see any reason to continue anymore."
        )
    )
    
    # Example 6: Comparing message with vs without history
    print("\n----- COMPARISON: SAME MESSAGE WITH AND WITHOUT HISTORY -----")
    # Without history first
    test_example(
        classifier_model, 
        tokenizer,
        "It's normal to feel a range of emotions after a loss. How have you been coping?",
        "Some days are better than others, but nights are still really hard.",
        thresholds
    )
    
    # Same message with history
    test_example(
        classifier_model, 
        tokenizer,
        "It's normal to feel a range of emotions after a loss. How have you been coping?",
        "Some days are better than others, but nights are still really hard.",
        thresholds,
        history=(
            "I understand this is a difficult time. Have you been able to sleep?",
            "Not really. I keep having nightmares about the accident."
        )
    )
    
    print("\n===== EVALUATION COMPLETE =====")

if __name__ == "__main__":
    main()