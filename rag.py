import os
import json
import time
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

# LangChain imports - updated for newer LangChain structure
try:
    # Try newer import structure first
    from langchain_community.llms import Ollama
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import ResponseSchema, StructuredOutputParser
except ImportError:
    # Fall back to older import structure
    try:
        from langchain.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.output_parsers import ResponseSchema, StructuredOutputParser
    except ImportError:
        print("Please install required packages:")
        print("pip install langchain langchain-community")
        raise

# Constants - same as in original code
SENTIMENT_CLASSES = {"Positive": 0, "Neutral": 1, "Negative": 2}
RESPONSE_NEEDED_CLASSES = {"No": 0, "Yes": 1}
CRISIS_CLASSES = {"No Crisis": 0, "Crisis": 1}

# Define the output schemas for structured parsing
response_schemas = [
    ResponseSchema(name="sentiment", 
                  description="The sentiment classification: Positive, Neutral, or Negative"),
    ResponseSchema(name="response_needed", 
                  description="Whether a response is needed: Yes or No"),
    ResponseSchema(name="crisis_detection", 
                  description="Whether there's a crisis situation: Crisis or No Crisis"),
    ResponseSchema(name="reasoning", 
                  description="A brief explanation of your classification")
]

# Create parser
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# Set up few-shot examples with clearly defined crisis boundaries
few_shot_examples = """
Example 1:
Support Message: "We're here for you during this difficult time. Remember to be gentle with yourself."
Recipient Reply: "Thank you. I'm taking it one day at a time and trying to focus on good memories."
Classification:
sentiment: Positive
response_needed: No
crisis_detection: No Crisis
reasoning: The reply shows healthy coping mechanisms with no signs of distress.

Example 2:
Support Message: "Grief can be overwhelming. Is there anything specific you're struggling with today?"
Recipient Reply: "I can't sleep and I keep thinking about how unfair it all is. It's been months but it still hurts."
Classification:
sentiment: Negative
response_needed: Yes
crisis_detection: No Crisis
reasoning: The reply shows distress and sleep issues but these are normal grief reactions, not indicating danger.

Example 3:
Support Message: "It's okay to feel lost sometimes. What has been helping you get through the day?"
Recipient Reply: "Nothing helps. I just want to be with my husband again. I don't see the point anymore."
Classification:
sentiment: Negative
response_needed: Yes
crisis_detection: Crisis
reasoning: "I just want to be with my husband" combined with "don't see the point anymore" suggests suicidal ideation.

Example 4:
Support Message: "How are you managing with your grief today?"
Recipient Reply: "I took all my pills. I'm finally going to be at peace."
Classification:
sentiment: Negative
response_needed: Yes
crisis_detection: Crisis
reasoning: This clearly indicates a suicide attempt requiring urgent intervention.

Example 5:
Support Message: "Take each day as it comes, and remember it's okay to seek support."
Recipient Reply: "I feel so empty without him. I miss him every day."
Classification:
sentiment: Negative
response_needed: Yes
crisis_detection: No Crisis
reasoning: This expresses normal grief emotions without indicating danger or self-harm.

Example 6:
Support Message: "It's normal to have ups and downs. How are you feeling today?"
Recipient Reply: "I'm thinking about killing myself. I can't do this anymore."
Classification:
sentiment: Negative
response_needed: Yes
crisis_detection: Crisis
reasoning: This contains an explicit statement of suicidal thoughts requiring immediate attention.
"""

# Create prompt template with improved crisis detection guidance
template = """You are an expert grief counselor specializing in classifying support messages. Analyze text messages from people experiencing grief and classify them according to:

1. Sentiment: Determine if the message is Positive, Neutral, or Negative.
2. Response Needed: Determine if a response is needed (Yes or No).
3. Crisis Detection: Determine if there's a crisis situation (Crisis or No Crisis).

FOR CRISIS DETECTION:
A message should ONLY be classified as "Crisis" if it contains CLEAR AND EXPLICIT indicators of:
- Suicidal thoughts or intentions
- Self-harm references or intentions
- Specific plans to harm oneself
- Direct statements about wanting to die or end one's life

DO NOT classify as crisis if you only see:
- General expressions of emptiness or sadness
- Missing the deceased person
- Statements of grief or loss without danger indications
- Feeling lost or alone
- Difficulty coping with grief
- Expressions like "I feel empty" or "I miss them so much"

IMPORTANT: Normal grief expressions are NOT crisis situations. Only classify as crisis when there is a clear risk of harm to self.

Here are examples:
{few_shot_examples}

Now, analyze this exchange:

Support Message: "{support_message}"
Recipient Reply: "{recipient_reply}"

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["support_message", "recipient_reply"],
    partial_variables={
        "few_shot_examples": few_shot_examples,
        "format_instructions": format_instructions
    }
)

# Initialize Ollama with Llama 3.1 (temperature=0 for deterministic responses)
llm = Ollama(model="llama3.1:latest", temperature=0)

def load_data(file_path):
    """Load the dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def normalize_field(value, valid_values, default):
    """Normalize field values to ensure consistency"""
    value = value.strip()
    if value in valid_values:
        return value
    
    # Special handling for each field type
    if valid_values == ["Positive", "Neutral", "Negative"]:
        if "positive" in value.lower():
            return "Positive"
        elif "negative" in value.lower():
            return "Negative"
        else:
            return "Neutral"
    
    elif valid_values == ["Yes", "No"]:
        if value.lower() in ["yes", "true", "1", "needed"]:
            return "Yes"
        else:
            return "No"
    
    elif valid_values == ["Crisis", "No Crisis"]:
        # More balanced approach to crisis detection
        if "crisis" in value.lower() and "no crisis" not in value.lower():
            return "Crisis"
        else:
            return "No Crisis"
    
    return default

def extract_classifications(text):
    """Extract classifications from unstructured text when parsing fails"""
    lines = text.lower().split('\n')
    
    sentiment = "Neutral"
    response_needed = "Yes"
    crisis_detection = "No Crisis"
    reasoning = ""
    
    for line in lines:
        if "sentiment" in line and ":" in line:
            value = line.split(":", 1)[1].strip()
            if "positive" in value:
                sentiment = "Positive"
            elif "negative" in value:
                sentiment = "Negative"
            elif "neutral" in value:
                sentiment = "Neutral"
        
        elif "response" in line and "need" in line and ":" in line:
            value = line.split(":", 1)[1].strip()
            response_needed = "Yes" if "yes" in value or "true" in value else "No"
        
        elif "crisis" in line and ":" in line:
            value = line.split(":", 1)[1].strip()
            crisis_detection = "Crisis" if "crisis" in value and "no crisis" not in value else "No Crisis"
        
        elif "reasoning" in line and ":" in line:
            reasoning = line.split(":", 1)[1].strip()
    
    # Only use explicitly dangerous crisis keywords
    explicit_crisis_keywords = ["suicide", "kill myself", "end my life", "harm myself", "took all my pills"]
    
    # Check for crisis keywords in text
    if any(keyword in text.lower() for keyword in explicit_crisis_keywords):
        crisis_detection = "Crisis"
    
    return {
        "sentiment": sentiment,
        "response_needed": response_needed,
        "crisis_detection": crisis_detection,
        "reasoning": reasoning or "Extracted from unstructured response"
    }

def classify_message(support_message, recipient_reply, history=None):
    """Classify a single grief support message exchange"""
    # Direct pattern-based crisis detection for high-confidence cases only
    direct_crisis_phrases = [
        "kill myself", "end my life", "suicide", "don't want to live",
        "want to die", "better off dead", "won't be here", "end it all",
        "took all my pills", "going to end", "last message", "goodbye forever"
    ]
    
    if any(phrase in recipient_reply.lower() for phrase in direct_crisis_phrases):
        return {
            "sentiment": "Negative",
            "response_needed": "Yes",
            "crisis_detection": "Crisis",
            "reasoning": "Direct crisis language detected"
        }
    
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
    for attempt in range(max_retries):
        try:
            response = llm.invoke(formatted_prompt)
            
            try:
                parsed_response = parser.parse(response)
                
                # Validate and normalize responses
                sentiment = normalize_field(parsed_response.get("sentiment", ""), 
                                         ["Positive", "Neutral", "Negative"], "Neutral")
                response_needed = normalize_field(parsed_response.get("response_needed", ""), 
                                               ["Yes", "No"], "Yes")
                crisis_detection = normalize_field(parsed_response.get("crisis_detection", ""), 
                                                ["Crisis", "No Crisis"], "No Crisis")
                reasoning = parsed_response.get("reasoning", "")
                
                # Check only for strong crisis indicators in reasoning without printing debug messages
                critical_crisis_indicators = ["suicid", "self-harm", "danger to self", "end life"]
                
                if (crisis_detection == "No Crisis" and 
                    any(indicator in reasoning.lower() for indicator in critical_crisis_indicators)):
                    crisis_detection = "Crisis"
                
                return {
                    "sentiment": sentiment,
                    "response_needed": response_needed,
                    "crisis_detection": crisis_detection,
                    "reasoning": reasoning
                }
            
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt, use manual extraction
                    return extract_classifications(response)
        
        except Exception as e:
            time.sleep(2)  # Wait before retrying
    
    # If all attempts failed
    return {
        "sentiment": "Neutral",
        "response_needed": "Yes",
        "crisis_detection": "No Crisis",
        "reasoning": "Error: Failed to get valid response"
    }

def compute_metrics(preds, labels):
    """Calculate evaluation metrics"""
    metrics = {}
    
    # Sentiment metrics
    metrics["sentiment_accuracy"] = accuracy_score(labels["sentiment"], preds["sentiment"])
    metrics["sentiment_f1"] = f1_score(labels["sentiment"], preds["sentiment"], average='weighted')
    
    # Response needed metrics
    metrics["response_accuracy"] = accuracy_score(labels["response_needed"], preds["response_needed"])
    metrics["response_f1"] = f1_score(labels["response_needed"], preds["response_needed"])
    metrics["response_precision"] = precision_score(labels["response_needed"], preds["response_needed"])
    metrics["response_recall"] = recall_score(labels["response_needed"], preds["response_needed"])
    
    # Crisis detection metrics - focus on recall for safety
    metrics["crisis_accuracy"] = accuracy_score(labels["crisis"], preds["crisis"])
    metrics["crisis_precision"] = precision_score(labels["crisis"], preds["crisis"])
    metrics["crisis_recall"] = recall_score(labels["crisis"], preds["crisis"])
    metrics["crisis_f1"] = f1_score(labels["crisis"], preds["crisis"])
    
    return metrics

def main():
    # Set up output directory
    output_dir = "llama3_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading test data...")
    try:
        test_data = load_data("grief_support_dataset/test.json")
        print(f"Test data loaded: {len(test_data)} conversations")
    except FileNotFoundError:
        # Try loading from paste.txt if the test file isn't found
        try:
            test_data = load_data("paste.txt")
            print(f"Test data loaded from paste.txt: {len(test_data)} conversations")
        except FileNotFoundError:
            print("Test data files not found. Please check the paths.")
            return
    
    # Process a subset of the test data for evaluation
    # Adjust sample size based on your needs and available time
    sample_size = min(150, len(test_data))
    test_sample = test_data[:sample_size]
    
    # Storage for predictions
    results = []
    
    print(f"Classifying {sample_size} conversations with Llama 3.1...")
    
    # Process each conversation
    for conv_idx, conversation in enumerate(tqdm(test_sample)):
        conv_id = conversation.get("conversation_id", str(conv_idx))
        turns = conversation["turns"]
        
        # Process each turn in the conversation
        for turn_idx, turn in enumerate(turns):
            support_message = turn["support_message"]
            recipient_reply = turn["recipient_reply"]
            
            # Get history for second turn
            history = None
            if turn_idx == 1:  # Second turn
                history = (turns[0]["support_message"], turns[0]["recipient_reply"])
            
            # Classify the message
            result = classify_message(support_message, recipient_reply, history)
            
            # Store the results
            results.append({
                "conversation_id": conv_id,
                "turn_idx": turn_idx,
                "support_message": support_message,
                "recipient_reply": recipient_reply,
                "sentiment": result["sentiment"],
                "response_needed": result["response_needed"],
                "crisis_detection": result["crisis_detection"],
                "reasoning": result["reasoning"],
                "true_sentiment": turn["labels"]["sentiment"],
                "true_response_needed": turn["labels"]["response_needed"],
                "true_crisis_detection": turn["labels"]["crisis_detection"]
            })
            
            # Small delay to avoid overwhelming Ollama
            time.sleep(0.5)
    
    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/classification_results.csv", index=False)
    print(f"Results saved to {output_dir}/classification_results.csv")
    
    # Convert to numeric values for metrics calculation
    numeric_predictions = {
        "sentiment": [SENTIMENT_CLASSES[pred] for pred in results_df["sentiment"]],
        "response_needed": [RESPONSE_NEEDED_CLASSES[pred] for pred in results_df["response_needed"]],
        "crisis": [CRISIS_CLASSES[pred] for pred in results_df["crisis_detection"]]
    }
    
    numeric_labels = {
        "sentiment": [SENTIMENT_CLASSES[label] for label in results_df["true_sentiment"]],
        "response_needed": [RESPONSE_NEEDED_CLASSES[label] for label in results_df["true_response_needed"]],
        "crisis": [CRISIS_CLASSES[label] for label in results_df["true_crisis_detection"]]
    }
    
    # Calculate overall metrics
    metrics = compute_metrics(numeric_predictions, numeric_labels)
    
    print("\nOVERALL METRICS:")
    print("=" * 50)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:20}: {metric_value:.4f}")
    
    # Save metrics to file
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Calculate metrics by turn (separate Turn 1 and Turn 2)
    for turn_idx in [0, 1]:
        turn_df = results_df[results_df["turn_idx"] == turn_idx]
        
        turn_predictions = {
            "sentiment": [SENTIMENT_CLASSES[pred] for pred in turn_df["sentiment"]],
            "response_needed": [RESPONSE_NEEDED_CLASSES[pred] for pred in turn_df["response_needed"]],
            "crisis": [CRISIS_CLASSES[pred] for pred in turn_df["crisis_detection"]]
        }
        
        turn_labels = {
            "sentiment": [SENTIMENT_CLASSES[label] for label in turn_df["true_sentiment"]],
            "response_needed": [RESPONSE_NEEDED_CLASSES[label] for label in turn_df["true_response_needed"]],
            "crisis": [CRISIS_CLASSES[label] for label in turn_df["true_crisis_detection"]]
        }
        
        turn_metrics = compute_metrics(turn_predictions, turn_labels)
        
        print(f"\nTURN {turn_idx + 1} METRICS:")
        print("=" * 50)
        for metric_name, metric_value in turn_metrics.items():
            print(f"{metric_name:20}: {metric_value:.4f}")
        
        # Save turn metrics
        with open(f"{output_dir}/turn{turn_idx+1}_metrics.json", 'w') as f:
            json.dump(turn_metrics, f, indent=2)
    
    # Calculate confusion matrices
    print("\nCONFUSION MATRICES:")
    print("=" * 50)
    
    for task, task_name in [("sentiment", "Sentiment"), 
                            ("response_needed", "Response Needed"), 
                            ("crisis_detection", "Crisis Detection")]:
        
        true_labels = results_df[f"true_{task}"]
        pred_labels = results_df[task]
        
        # Create and display confusion matrix
        labels = sorted(list(set(true_labels) | set(pred_labels)))
        cm = pd.DataFrame(0, index=labels, columns=labels)
        
        for i, row in results_df.iterrows():
            true_label = row[f"true_{task}"]
            pred_label = row[task]
            cm.loc[true_label, pred_label] += 1
        
        print(f"\n{task_name} Confusion Matrix:")
        print(cm)
        
        # Save confusion matrix
        cm.to_csv(f"{output_dir}/{task}_confusion.csv")

if __name__ == "__main__":
    main()