import os
import json
import time
from typing import Dict, List, Optional

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

def calculate_confidence_scores(text, result, recipient_reply):
    """
    Calculate confidence scores for each classification based on the text content,
    reasoning, and presence of certain keywords.
    
    Args:
        text: The full LLM response
        result: The parsed classification result
        recipient_reply: The original message being analyzed
        
    Returns:
        Dict with confidence scores for each classification
    """
    reasoning = result.get("reasoning", "").lower()
    text_lower = text.lower() if text else ""
    recipient_lower = recipient_reply.lower()
    
    # Initialize confidence scores
    confidence = {
        "sentiment": 0.7,  # Base confidence
        "response_needed": 0.7,  # Base confidence
        "crisis": 0.7  # Base confidence
    }
    
    # Sentiment confidence scoring
    sentiment_high_conf_indicators = ["clearly", "definitely", "strong", "obvious", "very", "extremely"]
    sentiment_medium_conf_indicators = ["seems", "appears", "likely", "mostly", "generally"]
    sentiment_low_conf_indicators = ["might be", "could be", "possibly", "maybe", "somewhat", "slightly"]
    
    # Increase confidence for explicit indicators
    if any(indicator in reasoning for indicator in sentiment_high_conf_indicators):
        confidence["sentiment"] = min(0.95, confidence["sentiment"] + 0.25)
    elif any(indicator in reasoning for indicator in sentiment_medium_conf_indicators):
        confidence["sentiment"] = min(0.85, confidence["sentiment"] + 0.15)
    elif any(indicator in reasoning for indicator in sentiment_low_conf_indicators):
        confidence["sentiment"] = max(0.5, confidence["sentiment"] - 0.2)
    
    # Check sentiment against message content
    if result["sentiment"] == "Positive":
        positive_words = ["thank", "grateful", "appreciate", "better", "good", "happy", "improving"]
        if any(word in recipient_lower for word in positive_words):
            confidence["sentiment"] = min(0.95, confidence["sentiment"] + 0.15)
        else:
            confidence["sentiment"] = max(0.5, confidence["sentiment"] - 0.1)
            
    elif result["sentiment"] == "Negative":
        negative_words = ["sad", "depressed", "hopeless", "pain", "hurt", "suffering", "miss", "lonely", "angry"]
        if any(word in recipient_lower for word in negative_words):
            confidence["sentiment"] = min(0.95, confidence["sentiment"] + 0.15)
        else:
            confidence["sentiment"] = max(0.5, confidence["sentiment"] - 0.1)
    
    # Response needed confidence scoring
    if result["response_needed"] == "Yes":
        response_indicators = ["need help", "please", "?", "can you", "i'm struggling", "don't know"]
        urgent_indicators = ["urgent", "emergency", "immediately", "asap", "now", "desperate"]
        
        if any(indicator in recipient_lower for indicator in urgent_indicators):
            confidence["response_needed"] = 0.98
        elif any(indicator in recipient_lower for indicator in response_indicators):
            confidence["response_needed"] = min(0.95, confidence["response_needed"] + 0.2)
    else:  # "No" response needed
        gratitude_closure = ["thank you", "thanks", "appreciate", "that's helpful", "got it"]
        statements = recipient_lower.endswith((".", "!"))
        
        if any(phrase in recipient_lower for phrase in gratitude_closure) and statements:
            confidence["response_needed"] = min(0.95, confidence["response_needed"] + 0.2)
    
    # Crisis confidence scoring is more nuanced 
    if result["crisis_detection"] == "Crisis":
        # Direct crisis indicators (highest confidence)
        direct_crisis_phrases = [
            "kill myself", "end my life", "suicide", "don't want to live",
            "want to die", "better off dead", "won't be here", "end it all",
            "took all my pills", "going to end", "last message", "goodbye forever"
        ]
        
        # Indirect crisis indicators (medium confidence)
        indirect_crisis_phrases = [
            "no point", "can't go on", "what's the point", "no reason to live",
            "nothing to live for", "rather be dead", "can't take this anymore"
        ]
        
        # Check for direct mentions (highest confidence)
        if any(phrase in recipient_lower for phrase in direct_crisis_phrases):
            confidence["crisis"] = 0.98
        # Check for indirect mentions (medium confidence)
        elif any(phrase in recipient_lower for phrase in indirect_crisis_phrases):
            confidence["crisis"] = 0.85
        # Check reasoning for crisis indicators
        elif any(term in reasoning for term in ["suicid", "self-harm", "danger", "risk", "harm"]):
            confidence["crisis"] = 0.8
    else:  # "No Crisis"
        # Check for positive coping language that indicates no crisis
        coping_phrases = [
            "getting better", "feeling better", "making progress", "managing", 
            "therapy is helping", "support group", "one day at a time"
        ]
        
        if any(phrase in recipient_lower for phrase in coping_phrases):
            confidence["crisis"] = min(0.95, confidence["crisis"] + 0.15)
    
    return confidence

def test_example(llm, support_message, recipient_reply, thresholds=None, history=None):
    """
    Test a single example
    
    Args:
        llm: The LLM model (Ollama)
        support_message: Current supportive message
        recipient_reply: Current recipient reply
        thresholds: Dictionary containing threshold values (for compatibility with evaluate_v3)
        history: Optional tuple of (prev_support_msg, prev_recipient_reply)
    """
    # Direct pattern-based crisis detection for high-confidence cases only
    direct_crisis_phrases = [
        "kill myself", "end my life", "suicide", "don't want to live",
        "want to die", "better off dead", "won't be here", "end it all",
        "took all my pills", "going to end", "last message", "goodbye forever"
    ]
    
    # Initialize response variable to track LLM output for confidence calculation
    response = None
    
    if any(phrase in recipient_reply.lower() for phrase in direct_crisis_phrases):
        result = {
            "sentiment": "Negative",
            "response_needed": "Yes",
            "crisis_detection": "Crisis",
            "reasoning": "Direct crisis language detected"
        }
        # Set high confidence for direct crisis detection
        confidence = {
            "sentiment": 0.95,
            "response_needed": 0.98,
            "crisis": 0.98
        }
    else:
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
        response = None
        for attempt in range(max_retries):
            try:
                response = llm.invoke(formatted_prompt)
                break
            except Exception as e:
                print(f"Error calling LLM (attempt {attempt+1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                else:
                    print("All LLM call attempts failed")
        
        if response:
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
                
                result = {
                    "sentiment": sentiment,
                    "response_needed": response_needed,
                    "crisis_detection": crisis_detection,
                    "reasoning": reasoning
                }
            except Exception as e:
                print(f"Error parsing structured response: {str(e)}")
                # Fall back to manual extraction
                result = extract_classifications(response)
        else:
            # If no response was received
            result = {
                "sentiment": "Neutral",
                "response_needed": "Yes",
                "crisis_detection": "No Crisis",
                "reasoning": "Error: Failed to get response from LLM"
            }
        
        # Calculate confidence scores based on the response and results
        confidence = calculate_confidence_scores(response, result, recipient_reply)
    
    # Get probabilities based on confidence scores
    sentiment_prob_map = {
        "Positive": confidence["sentiment"] if result["sentiment"] == "Positive" else (1 - confidence["sentiment"]) / 2,
        "Neutral": confidence["sentiment"] if result["sentiment"] == "Neutral" else (1 - confidence["sentiment"]) / 2,
        "Negative": confidence["sentiment"] if result["sentiment"] == "Negative" else (1 - confidence["sentiment"]) / 2
    }
    
    response_prob_map = {
        "Yes": confidence["response_needed"] if result["response_needed"] == "Yes" else 1 - confidence["response_needed"],
        "No": confidence["response_needed"] if result["response_needed"] == "No" else 1 - confidence["response_needed"]
    }
    
    # Crisis probability based on confidence
    crisis_prob = confidence["crisis"] if result["crisis_detection"] == "Crisis" else 1 - confidence["crisis"]
    
    # Set default thresholds if not provided
    if thresholds is None:
        thresholds = {
            'best_f1': {'threshold': 0.5},
            'high_recall': {'threshold': 0.3}
        }
    
    # Display inputs
    print("\n" + "-" * 50)
    print("INPUT:")
    if history:
        print(f"Previous support message: {history[0]}")
        print(f"Previous recipient reply: {history[1]}")
    print(f"Current support message: {support_message}")
    print(f"Current recipient reply: {recipient_reply}")
    
    # Display predictions with confidence scores
    print("\nPREDICTIONS:")
    print(f"Sentiment: {result['sentiment']} (confidence: {confidence['sentiment']:.4f})")
    print(f"  Probabilities: Positive={sentiment_prob_map['Positive']:.4f}, " +
          f"Neutral={sentiment_prob_map['Neutral']:.4f}, " +
          f"Negative={sentiment_prob_map['Negative']:.4f}")
    
    print(f"Response needed: {result['response_needed']} (confidence: {confidence['response_needed']:.4f})")
    print(f"  Probabilities: Yes={response_prob_map['Yes']:.4f}, No={response_prob_map['No']:.4f}")
    
    print(f"Crisis detection: {result['crisis_detection']} (confidence: {confidence['crisis']:.4f})")
    print(f"Crisis probability: {crisis_prob:.4f} (thresholds: best_f1={thresholds['best_f1']['threshold']:.4f}, high_recall={thresholds['high_recall']['threshold']:.4f})")
    print(f"Reasoning: {result['reasoning']}")
    
    # Only flag as requiring attention if it's a crisis or close to threshold
    if result['crisis_detection'] == "Crisis":
        print("*** REQUIRES ATTENTION ***")
    
    # For compatibility with evaluate_v3, show threshold info
    high_recall_threshold = thresholds['high_recall']['threshold']
    if crisis_prob > high_recall_threshold and crisis_prob <= thresholds['best_f1']['threshold'] and result['crisis_detection'] == "No Crisis":
        print(f"Note: Would be flagged as crisis with high_recall threshold ({high_recall_threshold:.4f})")

def main():
    print("Loading Llama 3.1 model...")
    
    # Initialize Ollama with Llama 3.1 (temperature=0 for deterministic responses)
    try:
        llm = Ollama(model="llama3.1:latest", temperature=0)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading Ollama model: {str(e)}")
        print("Please make sure Ollama is installed and running:")
        print("https://ollama.com/download")
        print("And that you have the llama3.1 model: 'ollama pull llama3.1'")
        return
    
    # Default thresholds (for compatibility with evaluate_v3)
    thresholds = {
        'best_f1': {'threshold': 0.5},
        'high_recall': {'threshold': 0.3}
    }
    
    print("Testing examples...")
    
    # Example 1: Potential crisis (without history)
    test_example(
        llm, 
        "We're here to support you during this difficult time.",
        "I don't know how to keep going. Everything reminds me of him.",
        thresholds
    )
    
    # Example 2: Positive sentiment (without history)
    test_example(
        llm, 
        "Remember that grief has no timeline. Be patient with yourself.",
        "Thank you. I've been trying to focus on happy memories and it's starting to help a little.",
        thresholds
    )
    
    # Example 3: Neutral sentiment with history
    test_example(
        llm, 
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
        llm, 
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