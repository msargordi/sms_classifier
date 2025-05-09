# IMPORTANT: Save this file as "grief_eval.py" (NOT "gradio.py")
# The filename must be different from "gradio.py" to avoid import conflicts

import gradio as gr
import torch
import time
import os

# Import from evaluate.py (RoBERTa model)
from evaluate import (
    MultitaskRobertaClassifier, 
    RobertaTokenizer, 
    SENTIMENT_CLASSES, 
    RESPONSE_NEEDED_CLASSES, 
    CRISIS_CLASSES,
    load_thresholds,
    device
)

# Import from evaluate_rag.py (RAG model)
from evaluate_rag import (
    prompt,
    normalize_field,
    extract_classifications,
    calculate_confidence_scores,
    response_schemas,
)

# Import from evaluate_LLM2Vec.py (LLM2Vec model)
from evaluate_LLM2Vec import (
    MultitaskClassifier,
    load_llm2vec_model,
    load_model_info,
    device
)

# Import LangChain components for the RAG model
try:
    # Try newer import structure first
    from langchain_community.llms import Ollama
    from langchain_core.output_parsers import StructuredOutputParser
except ImportError:
    # Fall back to older import structure
    try:
        from langchain.llms import Ollama
        from langchain.output_parsers import StructuredOutputParser
    except ImportError:
        print("Please install required packages:")
        print("pip install langchain langchain-community gradio")
        raise

# Constants
ROBERTA_MODEL_PATH = "grief_model"
LLM2VEC_MODEL_PATH = "grief_model_llm2vec"
MAX_LEN = 256

# Define example scenarios
scenarios = [
    {
        "name": "General grief support",
        "history": None,
        "support_message": "We're here to support you during this difficult time.",
        "recipient_reply": ""
    },
    {
        "name": "Recent loss",
        "history": None,
        "support_message": "Remember that grief has no timeline. Be patient with yourself.",
        "recipient_reply": ""
    },
    {
        "name": "Loss of parent - with history",
        "history": (
            "I'm here to listen if you need to talk.",
            "I lost my mom recently and I don't know how to cope."
        ),
        "support_message": "Many people find that grief changes over time. How are you feeling today?",
        "recipient_reply": ""
    },
    {
        "name": "Feeling overwhelmed - with history",
        "history": (
            "How are you dealing with things today?",
            "I've been trying to distract myself, but nothing helps."
        ),
        "support_message": "It's okay to feel overwhelmed. We're here whenever you need someone to listen.",
        "recipient_reply": ""
    }
]

def load_roberta_model():
    """Load and initialize the RoBERTa model"""
    print("Loading RoBERTa model...")
    
    try:
        # Load tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_PATH)
        
        # Make sure tokenizer has [CURRENT] token
        if "[CURRENT]" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({'additional_special_tokens': ['[CURRENT]']})
        
        # Initialize model with tokenizer
        model = MultitaskRobertaClassifier(tokenizer)
        
        # Resize embeddings to match tokenizer
        model.roberta.resize_token_embeddings(len(tokenizer))
        
        # Load the saved state dict
        model.load_state_dict(torch.load(f"{ROBERTA_MODEL_PATH}/model.pt", map_location="cpu"))
        model.to(device)
        model.eval()
        
        # Load thresholds
        thresholds = load_thresholds(ROBERTA_MODEL_PATH)
        
        print("RoBERTa model loaded successfully")
        return model, tokenizer, thresholds
    except Exception as e:
        print(f"Error loading RoBERTa model: {str(e)}")
        return None, None, None

def load_llm2vec_classifier():
    """Load and initialize the LLM2Vec model"""
    print("Loading LLM2Vec model...")
    
    try:
        # Load LLM2Vec base model
        llm2vec_model, tokenizer = load_llm2vec_model()
        
        # Load model information to get hidden size
        model_info = load_model_info(LLM2VEC_MODEL_PATH)
        hidden_size = model_info["hidden_size"]
        print(f"Using hidden size: {hidden_size}")
        
        # Initialize classifier with encoder
        classifier_model = MultitaskClassifier(
            encoder=llm2vec_model,
            hidden_size=hidden_size,
            n_sentiment_classes=model_info.get("n_sentiment_classes", 3),
            n_response_classes=model_info.get("n_response_classes", 2),
            n_crisis_classes=model_info.get("n_crisis_classes", 2)
        )
        
        # Load the classifier heads and attention layer
        print(f"Loading classifier heads from {LLM2VEC_MODEL_PATH}/classifier_heads.pt")
        classifier_state = torch.load(f"{LLM2VEC_MODEL_PATH}/classifier_heads.pt", map_location=device)
        
        # Load state dict for all components
        if 'attention_layer' in classifier_state:
            classifier_model.attention_layer.load_state_dict(classifier_state['attention_layer'])
        
        classifier_model.sentiment_classifier.load_state_dict(classifier_state['sentiment_classifier'])
        classifier_model.response_needed_classifier.load_state_dict(classifier_state['response_needed_classifier'])
        classifier_model.crisis_classifier.load_state_dict(classifier_state['crisis_classifier'])
        
        classifier_model.to(device)
        classifier_model.eval()
        
        # Load thresholds
        thresholds = load_thresholds(LLM2VEC_MODEL_PATH)
        
        print("LLM2Vec model loaded successfully")
        return classifier_model, tokenizer, thresholds
    except Exception as e:
        print(f"Error loading LLM2Vec model: {str(e)}")
        return None, None, None

def load_rag_model():
    """Load and initialize the RAG model (Ollama)"""
    print("Loading Llama 3.1 model for RAG...")
    
    try:
        llm = Ollama(model="llama3.1:latest", temperature=0)
        
        # Set default thresholds
        thresholds = {
            'best_f1': {'threshold': 0.5},
            'high_recall': {'threshold': 0.3}
        }
        
        # Create parser for structured output
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
        print("RAG model loaded successfully")
        return llm, parser, thresholds
    except Exception as e:
        print(f"Error loading Ollama model: {str(e)}")
        print("Please make sure Ollama is installed and running")
        return None, None, None

def evaluate_with_roberta(model, tokenizer, support_message, recipient_reply, thresholds, history=None):
    """Evaluate message with RoBERTa model"""
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
        
        # Get confidence scores
        sentiment_confidence = sentiment_probs[0, sentiment_pred].item()
        response_needed_confidence = response_needed_probs[0, response_needed_pred].item()
        
        # Get crisis probability (prob of class 1 - "Crisis")
        crisis_prob = crisis_probs[0, 1].item()
        
        # Use a much more sensitive threshold or override for suicide mentions
        crisis_threshold = thresholds['high_recall']['threshold']
        # Use a hardcoded lower threshold for extra sensitivity
        crisis_threshold = min(crisis_threshold, 0.2)  # Even more sensitive
        
        # Check for explicit suicide mentions
        suicide_phrases = ["kill myself", "end my life", "suicide", "don't want to live", 
                          "want to die", "better off dead", "won't be here", "end it all"]
        explicit_mention = any(phrase in recipient_reply.lower() for phrase in suicide_phrases)
        
        # Force crisis detection for explicit mentions
        if explicit_mention:
            crisis_pred = 1
            crisis_prob = 0.95  # High confidence for explicit mentions
        else:
            # Make prediction based on very sensitive threshold
            crisis_pred = 1 if crisis_prob > crisis_threshold else 0
        
        # Get confidence for crisis prediction
        crisis_confidence = crisis_prob if crisis_pred == 1 else 1 - crisis_prob
        
        # Get attention weights for history vs current message
        if history and attention_weights is not None:
            history_weight = attention_weights[0, 0].item()
            current_weight = attention_weights[0, 1].item()
        else:
            history_weight = None
            current_weight = None
    
    # Map numeric predictions back to class names
    sentiment_map = {v: k for k, v in SENTIMENT_CLASSES.items()}
    response_needed_map = {v: k for k, v in RESPONSE_NEEDED_CLASSES.items()}
    crisis_map = {v: k for k, v in CRISIS_CLASSES.items()}
    
    # Prepare results
    sentiment = sentiment_map[sentiment_pred]
    response_needed = response_needed_map[response_needed_pred]
    crisis = crisis_map[crisis_pred]
    
    return sentiment, response_needed, crisis, sentiment_confidence, response_needed_confidence, crisis_confidence, crisis_threshold, history_weight, current_weight

def evaluate_with_llm2vec(model, tokenizer, support_message, recipient_reply, thresholds, history=None):
    """Evaluate message with LLM2Vec model"""
    # Format text parts for LLM2Vec model
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
        
        # Get confidence scores
        sentiment_confidence = sentiment_probs[0, sentiment_pred].item()
        response_needed_confidence = response_needed_probs[0, response_needed_pred].item()
        
        # Get crisis probability (prob of class 1 - "Crisis")
        crisis_prob = crisis_probs[0, 1].item()
        
        # Use high_recall threshold for crisis detection but make it even more sensitive
        crisis_threshold = thresholds['high_recall']['threshold']
        # Use a hardcoded lower threshold for extra sensitivity
        crisis_threshold = min(crisis_threshold, 0.2)  # Even more sensitive
        
        # Check for explicit suicide mentions
        suicide_phrases = ["kill myself", "end my life", "suicide", "don't want to live", 
                          "want to die", "better off dead", "won't be here", "end it all"]
        explicit_mention = any(phrase in recipient_reply.lower() for phrase in suicide_phrases)
        
        # Force crisis detection for explicit mentions
        if explicit_mention:
            crisis_pred = 1
            crisis_prob = 0.95  # High confidence for explicit mentions
        else:
            # Make prediction based on very sensitive threshold
            crisis_pred = 1 if crisis_prob > crisis_threshold else 0
        
        # Get confidence for the predicted crisis class
        crisis_confidence = crisis_prob if crisis_pred == 1 else 1 - crisis_prob
        
        # Get attention weights
        history_attention = attention_weights[0, 0].item() if history else None
        current_attention = attention_weights[0, 1].item() if history else None
    
    # Map numeric predictions back to class names
    sentiment_map = {v: k for k, v in SENTIMENT_CLASSES.items()}
    response_needed_map = {v: k for k, v in RESPONSE_NEEDED_CLASSES.items()}
    crisis_map = {v: k for k, v in CRISIS_CLASSES.items()}
    
    # Prepare results
    sentiment = sentiment_map[sentiment_pred]
    response_needed = response_needed_map[response_needed_pred]
    crisis = crisis_map[crisis_pred]
    
    # Set borderline flag for cases that are close to the threshold
    borderline = (not explicit_mention and 
                 crisis_prob > 0.1 and  # Detect even very low probabilities as borderline
                 crisis_prob <= crisis_threshold)
    
    return sentiment, response_needed, crisis, sentiment_confidence, response_needed_confidence, crisis_confidence, crisis_threshold, history_attention, current_attention, borderline

def evaluate_with_rag(llm, parser, support_message, recipient_reply, thresholds, history=None):
    """Evaluate message with RAG model (Ollama)"""
    # Direct pattern-based crisis detection for high-confidence cases
    direct_crisis_phrases = [
        "kill myself", "end my life", "suicide", "don't want to live",
        "want to die", "better off dead", "won't be here", "end it all"
    ]
    
    if any(phrase in recipient_reply.lower() for phrase in direct_crisis_phrases):
        return "Negative", "Yes", "Crisis", 0.95, 0.98, 0.98, thresholds['high_recall']['threshold'], None, None, "Direct crisis language detected"
    
    # Add history context if available
    if history:
        prev_support_msg, prev_recipient_reply = history
        history_context = f"""Previous messages:
Support: "{prev_support_msg}"
Reply: "{prev_recipient_reply}"

Current messages:"""
        
        formatted_prompt = prompt.format(
            support_message=support_message,
            recipient_reply=recipient_reply
        )
        formatted_prompt = formatted_prompt.replace("Now, analyze this exchange:", 
                                                  f"{history_context}\nNow, analyze this exchange:")
    else:
        formatted_prompt = prompt.format(
            support_message=support_message,
            recipient_reply=recipient_reply
        )
    
    # Get model response with retries
    max_retries = 3
    raw_response = None
    for attempt in range(max_retries):
        try:
            raw_response = llm.invoke(formatted_prompt)
            break
        except Exception as e:
            print(f"Error calling LLM (attempt {attempt+1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                print("All LLM call attempts failed")
    
    if raw_response:
        try:
            parsed_response = parser.parse(raw_response)
            
            # Validate and normalize responses
            sentiment = normalize_field(parsed_response.get("sentiment", ""), 
                                      ["Positive", "Neutral", "Negative"], "Neutral")
            response_needed = normalize_field(parsed_response.get("response_needed", ""), 
                                            ["Yes", "No"], "Yes")
            crisis_detection = normalize_field(parsed_response.get("crisis_detection", ""), 
                                             ["Crisis", "No Crisis"], "No Crisis")
            reasoning = parsed_response.get("reasoning", "")
            
            # Check only for strong crisis indicators in reasoning
            critical_crisis_indicators = ["suicid", "self-harm", "danger to self", "end life"]
            
            if (crisis_detection == "No Crisis" and 
                any(indicator in reasoning.lower() for indicator in critical_crisis_indicators)):
                crisis_detection = "Crisis"
            
            # Calculate confidence scores
            confidence = calculate_confidence_scores(raw_response, {
                "sentiment": sentiment,
                "response_needed": response_needed,
                "crisis_detection": crisis_detection,
                "reasoning": reasoning
            }, recipient_reply)
            
            # Get confidence scores for each classification
            sentiment_confidence = confidence["sentiment"]
            response_needed_confidence = confidence["response_needed"]
            crisis_confidence = confidence["crisis"]
            
            # Get crisis probability based on confidence
            crisis_prob = confidence["crisis"] if crisis_detection == "Crisis" else 1 - confidence["crisis"]
            
            return sentiment, response_needed, crisis_detection, sentiment_confidence, response_needed_confidence, crisis_confidence, thresholds['high_recall']['threshold'], None, None, reasoning
        
        except Exception as e:
            print(f"Error parsing structured response: {str(e)}")
            # Fall back to manual extraction
            result = extract_classifications(raw_response)
            return (
                result["sentiment"], 
                result["response_needed"], 
                result["crisis_detection"],
                0.7,  # Default sentiment confidence
                0.7,  # Default response confidence  
                0.7 if result["crisis_detection"] == "Crisis" else 0.7,  # Default crisis confidence
                thresholds['high_recall']['threshold'],
                None,
                None,
                result["reasoning"]
            )
    else:
        # If no response was received
        return (
            "Neutral", 
            "Yes", 
            "No Crisis",
            0.5,  # Default sentiment confidence
            0.5,  # Default response confidence
            0.5,  # Default crisis confidence 
            thresholds['high_recall']['threshold'],
            None,
            None,
            "Error: Failed to get response from LLM"
        )

def select_scenario(choice):
    """Handle scenario selection"""
    for scenario in scenarios:
        if scenario["name"] == choice:
            history_text = ""
            if scenario["history"]:
                prev_support, prev_reply = scenario["history"]
                history_text = f"Previous support: {prev_support}\nPrevious reply: {prev_reply}"
            support_message = scenario["support_message"]
            return history_text, support_message, ""
    
    # Default if not found
    return "", "We're here to support you during this difficult time.", ""

def evaluate_message(model_choice, history_text, support_message, recipient_reply):
    """Main evaluation function"""
    if not recipient_reply.strip():
        return "Please enter a recipient reply to evaluate."
    
    # Process history if available
    history_tuple = None
    if history_text:
        # Extract history from text format
        lines = history_text.strip().split("\n")
        if len(lines) >= 2:
            prev_support = lines[0].replace("Previous support: ", "")
            prev_reply = lines[1].replace("Previous reply: ", "")
            history_tuple = (prev_support, prev_reply)
    
    # Evaluate with selected model
    if model_choice == "RoBERTa":
        # Check if model is available
        if roberta_model is None:
            return "RoBERTa model not loaded. Please check if the model files exist."
        
        sentiment, response_needed, crisis, sentiment_conf, response_conf, crisis_conf, threshold, history_weight, current_weight = evaluate_with_roberta(
            roberta_model,
            roberta_tokenizer,
            support_message,
            recipient_reply,
            roberta_thresholds,
            history_tuple
        )
        
        # Format attention weights info if available
        attention_info = ""
        if history_weight is not None and current_weight is not None:
            # Verify they sum to 1 (within float precision)
            sum_check = ""
            if abs(history_weight + current_weight - 1.0) > 0.0001:
                # Normalize to ensure they sum to 1
                total = history_weight + current_weight
                history_weight = history_weight / total
                current_weight = current_weight / total
                sum_check = " (normalized)"
                
            attention_info = f"""
Attention Weights{sum_check}:
  History: {history_weight:.4f}
  Current: {current_weight:.4f}"""
  
        
        result = f"""Results from RoBERTa model:
        
Sentiment: {sentiment} (confidence: {sentiment_conf:.4f})
Response Needed: {response_needed} (confidence: {response_conf:.4f})
Crisis Detection: {crisis} (confidence: {crisis_conf:.4f})

{" ⚠️ REQUIRES ATTENTION ⚠️" if crisis == "Crisis" else ""}
"""
    
    elif model_choice == "LLM2Vec":
        # Check if model is available
        if llm2vec_model is None:
            return "LLM2Vec model not loaded. Please check if the model files exist."
        
        sentiment, response_needed, crisis, sentiment_conf, response_conf, crisis_conf, threshold, history_attention, current_attention, borderline = evaluate_with_llm2vec(
            llm2vec_model,
            llm2vec_tokenizer,
            support_message,
            recipient_reply,
            llm2vec_thresholds,
            history_tuple
        )
        
        # Format attention weights info if available
        attention_info = ""
        if history_attention is not None and current_attention is not None:
            # Verify they sum to 1 (within float precision)
            sum_check = ""
            if abs(history_attention + current_attention - 1.0) > 0.0001:
                # Normalize to ensure they sum to 1
                total = history_attention + current_attention
                history_attention = history_attention / total
                current_attention = current_attention / total
                sum_check = " (normalized)"
                
            attention_info = f"""
Attention Weights{sum_check}:
  History: {history_attention:.4f}
  Current: {current_attention:.4f}"""
        
        # Special note for borderline cases - simplified
        attention_flag = " ⚠️ REQUIRES ATTENTION ⚠️" if crisis == "Crisis" else ""
        
        result = f"""Results from LLM2Vec model:
        
Sentiment: {sentiment} (confidence: {sentiment_conf:.4f})
Response Needed: {response_needed} (confidence: {response_conf:.4f})
Crisis Detection: {crisis} (confidence: {crisis_conf:.4f})

{attention_flag}
"""
    
    elif model_choice == "RAG (LLM)":
        # Check if model is available
        if rag_model is None:
            return "RAG model not loaded. Please make sure Ollama is installed and running."
        
        sentiment, response_needed, crisis, sentiment_conf, response_conf, crisis_conf, threshold, _, _, reasoning = evaluate_with_rag(
            rag_model,
            rag_parser,
            support_message,
            recipient_reply,
            rag_thresholds,
            history_tuple
        )
        
        result = f"""Results from RAG model:
        
Sentiment: {sentiment} (confidence: {sentiment_conf:.4f})
Response Needed: {response_needed} (confidence: {response_conf:.4f})
Crisis Detection: {crisis} (confidence: {crisis_conf:.4f})

Reasoning: {reasoning}

{" ⚠️ REQUIRES ATTENTION ⚠️" if crisis == "Crisis" else ""}
"""
    
    else:
        return "Please select a valid model."
    
    return result

# Load models globally to avoid reloading
roberta_model, roberta_tokenizer, roberta_thresholds = None, None, None
llm2vec_model, llm2vec_tokenizer, llm2vec_thresholds = None, None, None
rag_model, rag_parser, rag_thresholds = None, None, None

# Try to load models
try:
    roberta_model, roberta_tokenizer, roberta_thresholds = load_roberta_model()
except Exception as e:
    print(f"Error loading RoBERTa model: {str(e)}")
    print("RoBERTa model will not be available.")

try:
    llm2vec_model, llm2vec_tokenizer, llm2vec_thresholds = load_llm2vec_classifier()
except Exception as e:
    print(f"Error loading LLM2Vec model: {str(e)}")
    print("LLM2Vec model will not be available.")

try:
    rag_model, rag_parser, rag_thresholds = load_rag_model()
except Exception as e:
    print(f"Error loading RAG model: {str(e)}")
    print("RAG model will not be available.")

# Determine which model to select by default based on availability
default_model = None
if roberta_model is not None:
    default_model = "RoBERTa"
elif llm2vec_model is not None:
    default_model = "LLM2Vec"
elif rag_model is not None:
    default_model = "RAG (LLM)"
else:
    default_model = "RoBERTa"  # Default even if not available

# Create interface with simple layout
with gr.Blocks(title="Grief Support Message Evaluation") as demo:
    gr.Markdown("# Grief Support Message Evaluation")
    
    with gr.Row():
        with gr.Column():
            model_choice = gr.Radio(
                ["RoBERTa", "LLM2Vec", "RAG (LLM)"],
                label="Select Model",
                value=default_model
            )
            
            scenario_choice = gr.Dropdown(
                [scenario["name"] for scenario in scenarios],
                label="Select Scenario",
                value=scenarios[0]["name"]
            )
            
            scenario_btn = gr.Button("Load Scenario")
            
    with gr.Row():
        with gr.Column():
            history_text = gr.Textbox(
                label="Conversation History",
                placeholder="No previous conversation",
                interactive=False,
                lines=3
            )
            
            support_message = gr.Textbox(
                label="Support Message",
                placeholder="Loading...",
                interactive=False,
                lines=2,
                value=scenarios[0]["support_message"]
            )
            
            recipient_reply = gr.Textbox(
                label="Recipient Reply",
                placeholder="Type a response...",
                lines=3
            )
            
            evaluate_btn = gr.Button("Evaluate Response", variant="primary")
            
            result_text = gr.Textbox(
                label="Evaluation Results",
                interactive=False,
                lines=15
            )
    
    # Set up event handlers
    scenario_btn.click(
        select_scenario,
        inputs=[scenario_choice],
        outputs=[history_text, support_message, recipient_reply]
    )
    
    evaluate_btn.click(
        evaluate_message,
        inputs=[model_choice, history_text, support_message, recipient_reply],
        outputs=[result_text]
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(share=True)