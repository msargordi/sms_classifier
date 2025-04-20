import torch
import json
from transformers import RobertaTokenizer, RobertaModel
from torch import nn
import torch.nn.functional as F

# Force CPU usage to avoid MPS/Metal alignment issues
device = torch.device("cpu")
print(f"Using device: {device}")

# Class mappings
SENTIMENT_CLASSES = {"Positive": 0, "Neutral": 1, "Negative": 2}
RESPONSE_NEEDED_CLASSES = {"No": 0, "Yes": 1}
CRISIS_CLASSES = {"No Crisis": 0, "Crisis": 1}

# Constants
MAX_LEN = 256
MODEL_PATH = "grief_model_v3"  # Updated model path to v3

class MultitaskRobertaClassifier(nn.Module):
    def __init__(self, tokenizer, n_sentiment_classes=3, n_response_classes=2, n_crisis_classes=2):
        super(MultitaskRobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
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
        
        # Task-specific heads take the hidden size (not doubled) due to attention mechanism
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

def load_thresholds(model_path):
    """Load the threshold values from the saved JSON file"""
    try:
        with open(f"{model_path}/thresholds.json", 'r') as f:
            thresholds = json.load(f)
        print(f"Loaded thresholds from {model_path}/thresholds.json")
        
        # The thresholds in the file appear to be for detecting the "No Crisis" class
        # We need to invert them for predicting the "Crisis" class
        best_f1 = 1.0 - thresholds['best_f1']['threshold']
        high_recall = 1.0 - thresholds['high_recall']['threshold']
        
        # Create new thresholds dictionary
        adjusted_thresholds = {
            'best_f1': {'threshold': 0.5},  # Default
            'high_recall': {'threshold': 0.3}  # Default
        }
        
        # Only use inverted thresholds if they make sense (between 0 and 1)
        if 0 <= best_f1 <= 1:
            adjusted_thresholds['best_f1']['threshold'] = best_f1
        if 0 <= high_recall <= 1:
            adjusted_thresholds['high_recall']['threshold'] = high_recall
            
        print(f"Adjusted thresholds - best_f1: {adjusted_thresholds['best_f1']['threshold']:.4f}, " + 
              f"high_recall: {adjusted_thresholds['high_recall']['threshold']:.4f}")
        
        return adjusted_thresholds
        
    except FileNotFoundError:
        print(f"Warning: Could not find thresholds.json in {model_path}")
        # Return default thresholds
        return {
            'best_f1': {'threshold': 0.5},
            'high_recall': {'threshold': 0.3}
        }

def test_example(model, tokenizer, support_message, recipient_reply, thresholds, history=None):
    """
    Test a single example
    
    Args:
        model: The MultitaskRobertaClassifier model
        tokenizer: RoBERTa tokenizer
        support_message: Current supportive message
        recipient_reply: Current recipient reply
        thresholds: Dictionary containing threshold values
        history: Optional tuple of (prev_support_msg, prev_recipient_reply)
    """
    # Format text based on whether history is provided
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
    
    # Get data
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        sentiment_logits, response_needed_logits, crisis_logits, attention_weights = model(
            input_ids=input_ids,
            attention_mask=attention_mask
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
    
    # Display predictions
    print("\nPREDICTIONS:")
    print(f"Sentiment: {sentiment_map[sentiment_pred]} (confidence: {sentiment_confidence:.4f})")
    print(f"Response needed: {response_needed_map[response_needed_pred]} (confidence: {response_needed_confidence:.4f})")
    print(f"Crisis detection: {crisis_map[crisis_pred]} (confidence: {crisis_confidence:.4f})")
    print(f"Crisis probability: {crisis_prob:.4f} (thresholds: best_f1={crisis_threshold:.4f}, high_recall={high_recall_threshold:.4f})")
    
    # Display attention weights
    print(f"Attention weights: History={history_attention:.4f}, Current={current_attention:.4f}")
    
    # Only flag as requiring attention if it's an actual crisis or close to threshold
    if crisis_pred == 1 or (crisis_prob > high_recall_threshold and crisis_prob <= crisis_threshold):
        print("*** REQUIRES ATTENTION ***")
    
    # Show additional info for borderline cases
    if crisis_prob > high_recall_threshold and crisis_prob <= crisis_threshold and crisis_pred == 0:
        print(f"Note: Would be flagged as crisis with high_recall threshold ({high_recall_threshold:.4f})")

def main():
    print("Loading tokenizer and model...")
    
    # Load tokenizer first
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    print(f"Tokenizer loaded successfully with vocab size: {len(tokenizer)}")
    
    # Make sure tokenizer has [CURRENT] token
    if "[CURRENT]" not in tokenizer.get_vocab():
        print("Adding [CURRENT] token to tokenizer")
        tokenizer.add_special_tokens({'additional_special_tokens': ['[CURRENT]']})
    
    # Initialize model with tokenizer
    model = MultitaskRobertaClassifier(tokenizer)
    
    # Resize embeddings to match tokenizer
    model.roberta.resize_token_embeddings(len(tokenizer))
    
    # Load the saved state dict
    print(f"Loading model from {MODEL_PATH}/model.pt")
    model.load_state_dict(torch.load(f"{MODEL_PATH}/model.pt", map_location="cpu"))
    model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Load thresholds
    thresholds = load_thresholds(MODEL_PATH)
    print(f"Using crisis thresholds - best_f1: {thresholds['best_f1']['threshold']:.4f}, high_recall: {thresholds['high_recall']['threshold']:.4f}")
    
    print("Testing examples...")
    
    # Example 1: Potential crisis (without history)
    test_example(
        model, 
        tokenizer,
        "We're here to support you during this difficult time.",
        "I don't know how to keep going. Everything reminds me of him.",
        thresholds
    )
    
    # Example 2: Positive sentiment (without history)
    test_example(
        model, 
        tokenizer,
        "Remember that grief has no timeline. Be patient with yourself.",
        "Thank you. I've been trying to focus on happy memories and it's starting to help a little.",
        thresholds
    )
    
    # Example 3: Neutral sentiment with history
    test_example(
        model, 
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
        model, 
        tokenizer,
        "It's okay to feel overwhelmed. We're here whenever you need someone to listen.",
        "I want to kill myself.",
        thresholds,
        history=(
            "How are you dealing with things today?",
            "I've been trying to distract myself, but nothing helps."
        )
    )

if __name__ == "__main__":
    main()