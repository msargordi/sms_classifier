import json
import re
import time
import os
import random
from openai import OpenAI
import hashlib
from difflib import SequenceMatcher

# Initialize the OpenAI client
client = OpenAI()

# Dataset generation prompt
PROMPT = """# Grief Support SMS Response Dataset Generation

You are tasked with generating a synthetic dataset of text message conversations for a grief support service. Each conversation will consist of:

1. First supportive message sent by the service
2. First reply from a person experiencing grief
3. Second supportive message responding to the first reply
4. Second reply from the person experiencing grief

## Instructions

Generate {batch_size} conversations with the following structure:

```
CONVERSATION_ID: [number]
TURN_1:
  SUPPORTIVE_MESSAGE: [Initial grief support message]
  RECIPIENT_REPLY: [First reply from the person receiving support]
  LABELS:
    - Sentiment: [Positive/Neutral/Negative]
    - Response Needed: [Yes/No]
    - Crisis Detection: [Crisis/No Crisis]
TURN_2:
  SUPPORTIVE_MESSAGE: [Follow-up support message based on first reply]
  RECIPIENT_REPLY: [Second reply from the person]
  LABELS:
    - Sentiment: [Positive/Neutral/Negative]
    - Response Needed: [Yes/No]
    - Crisis Detection: [Crisis/No Crisis]
```

## Guidelines:

- Create diverse supportive messages addressing different aspects of grief
- Each conversation should be coherent and show natural progression between turns
- Generate realistic replies that vary in:
  - Emotional tone (from grateful to angry)
  - Response necessity (some clearly need follow-up, others don't)
  - Crisis level (most not in crisis, but include some crisis situations)
  - Length (short to multi-sentence)
  - Clarity (some ambiguous responses)
- Include realistic message features like typos, abbreviations, and informal language
- Represent various grief contexts:
  - Recent loss of loved one
  - Anniversary grief
  - Terminal illness (self or loved one)
  - Complicated grief situations
  - Positive memory sharing
  - Progress in grief journey
  - Gratitude for support received
  - Finding meaning after loss
- Ensure representation across grief stages (denial, anger, bargaining, depression, acceptance)

## Distribution Guidelines:
- STRICTLY ENFORCE sentiment distribution in both turns:
  - 33% positive sentiment
  - 33% neutral sentiment
  - 33% negative sentiment
- About 50% requiring response, 50% not requiring response
- Include 10-15% crisis situations, 85-90% non-crisis

## First Reply Guidelines:
For first replies (TURN_1), ensure equal distribution across:
- Positive responses: expressions of gratitude, sharing positive memories, reporting progress
- Neutral responses: information-seeking, acknowledgments, factual statements
- Negative responses: expressions of distress, anger, despair, or overwhelming emotions

## Second Reply Guidelines:
For second replies (TURN_2), ensure similar balanced distribution while maintaining conversation coherence. Include examples of:
- Improvement (negative → positive/neutral)
- Deterioration (positive/neutral → negative)
- Stability (maintaining similar sentiment)

## Important: Include Ambiguous Examples

Include examples of ambiguous responses that would be challenging to classify:
- Responses like "thanks" which could be positive acknowledgment or dismissive/disengaging
- Responses like "I just want to be with him" which could indicate either normal grief or suicidal ideation
- Messages where the meaning changes based on conversation context

## Example Conversations for Each Sentiment Pattern:

```
CONVERSATION_ID: 1
TURN_1:
  SUPPORTIVE_MESSAGE: Remember that grief has no timeline. Be patient with yourself as you navigate these difficult emotions.
  RECIPIENT_REPLY: Thank you for saying that. I actually had a good day yesterday - looked at old photos and mostly smiled instead of cried.
  LABELS:
    - Sentiment: Positive
    - Response Needed: Yes
    - Crisis Detection: No Crisis
TURN_2:
  SUPPORTIVE_MESSAGE: That's wonderful to hear about your good day. Those small moments of peace are precious. What photos brought you the most comfort?
  RECIPIENT_REPLY: Our trip to the beach last summer. It was the last big family vacation. I'm going to frame one of those photos for my desk.
  LABELS:
    - Sentiment: Positive
    - Response Needed: No
    - Crisis Detection: No Crisis
```

```
CONVERSATION_ID: 2
TURN_1:
  SUPPORTIVE_MESSAGE: Grief can feel overwhelming at times. Remember that we're here whenever you need someone to listen.
  RECIPIENT_REPLY: It's been 3 months since my dad died. Does it ever get easier? I still can't sleep well.
  LABELS:
    - Sentiment: Neutral
    - Response Needed: Yes
    - Crisis Detection: No Crisis
TURN_2:
  SUPPORTIVE_MESSAGE: Many people find that grief changes over time rather than disappearing. Sleep difficulties are common. Have you found any small things that help, even if just a little?
  RECIPIENT_REPLY: I've been listening to his favorite music before bed. Not sure if it helps but it feels important somehow.
  LABELS:
    - Sentiment: Neutral
    - Response Needed: Yes
    - Crisis Detection: No Crisis
```

```
CONVERSATION_ID: 3
TURN_1:
  SUPPORTIVE_MESSAGE: On difficult days, try to be as gentle with yourself as your loved one would want you to be.
  RECIPIENT_REPLY: I'm trying but it's been 2 years and I still can't look at her photos without breaking down. Is this normal?
  LABELS:
    - Sentiment: Negative
    - Response Needed: Yes
    - Crisis Detection: No Crisis
TURN_2:
  SUPPORTIVE_MESSAGE: Yes, that's completely normal. Many people experience intense emotions years after a loss. Photos can be powerful triggers. Would it help to talk about a happy memory of her?
  RECIPIENT_REPLY: Maybe. I'm just so tired of feeling this way. Sometimes I think about what it would be like to join her.
  LABELS:
    - Sentiment: Negative
    - Response Needed: Yes
    - Crisis Detection: Crisis
```

## IMPORTANT: Make each conversation distinct and unique.
Ensure each conversation has its own unique scenario, wording, and emotional context.
Avoid similar phrasings or situations between conversations.

Please generate the dataset in the exact format specified above, with clear delimiters between each conversation entry.
"""

def calculate_conversation_signature(conversation):
    """
    Calculate a signature for a conversation to help identify duplicates
    
    Args:
        conversation: Dictionary containing a single conversation
    
    Returns:
        String hash signature of the conversation content
    """
    # Create a string containing all message content
    content = ""
    for turn in conversation.get('turns', []):
        content += turn.get('support_message', '') + turn.get('recipient_reply', '')
    
    # Create a hash signature
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def calculate_similarity(conv1, conv2):
    """
    Calculate text similarity between two conversations
    
    Args:
        conv1, conv2: Conversation dictionaries to compare
    
    Returns:
        Float between 0-1 representing similarity (1 being identical)
    """
    # Create combined text for each conversation
    text1 = ""
    text2 = ""
    
    for turn in conv1.get('turns', []):
        text1 += turn.get('support_message', '') + " " + turn.get('recipient_reply', '') + " "
        
    for turn in conv2.get('turns', []):
        text2 += turn.get('support_message', '') + " " + turn.get('recipient_reply', '') + " "
    
    # Calculate similarity using SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()

def is_duplicate(conversation, existing_conversations, similarity_threshold=0.7):
    """
    Check if a conversation is too similar to any existing conversation
    
    Args:
        conversation: Dictionary containing a conversation to check
        existing_conversations: List of existing conversation dictionaries
        similarity_threshold: Threshold above which conversations are considered too similar
        
    Returns:
        Boolean indicating whether the conversation is a duplicate
    """
    # Calculate signature for quick hash-based comparison
    conv_signature = calculate_conversation_signature(conversation)
    
    for existing in existing_conversations:
        # Quick check - if hash signatures match, it's definitely a duplicate
        existing_signature = calculate_conversation_signature(existing)
        if conv_signature == existing_signature:
            return True
        
        # For conversations with different signatures, check content similarity
        similarity = calculate_similarity(conversation, existing)
        if similarity > similarity_threshold:
            return True
            
    return False

def generate_dataset_batch(batch_size=10, total_size=100, existing_data=None):
    """
    Generate the grief support dataset in batches to avoid token limits
    
    Args:
        batch_size: Number of examples to generate in each API call
        total_size: Total dataset size desired
        existing_data: List of previously generated conversations
    
    Returns:
        List of conversation dictionaries
    """
    dataset = []
    all_conversations = existing_data or []
    duplicate_count = 0
    total_attempts = 0
    max_attempts = total_size * 3  # Set a maximum number of attempts to prevent infinite loops
    
    while len(dataset) < total_size and total_attempts < max_attempts:
        current_batch_size = min(batch_size, total_size - len(dataset))
        print(f"Generating batch {len(dataset)//batch_size + 1}, targeting {current_batch_size} examples...")
        
        try:
            # Replace the batch_size placeholder in the prompt
            current_prompt = PROMPT.format(batch_size=current_batch_size)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant creating synthetic data."},
                    {"role": "user", "content": current_prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            # Extract the generated content
            content = response.choices[0].message.content
            
            # Parse the content to extract conversations
            batch_data = parse_generated_content(content)
            
            # Check for duplicates within this batch
            unique_batch = []
            for conversation in batch_data:
                total_attempts += 1
                
                # Check if the conversation is a duplicate
                if not is_duplicate(conversation, all_conversations):
                    unique_batch.append(conversation)
                    all_conversations.append(conversation)
                else:
                    duplicate_count += 1
            
            # Add unique conversations to the dataset
            dataset.extend(unique_batch)
            
            print(f"Added {len(unique_batch)} unique examples in this batch. " +
                  f"Progress: {len(dataset)}/{total_size} ({duplicate_count} duplicates detected)")
            
            # Add a small delay to avoid rate limiting
            time.sleep(3)
                
        except Exception as e:
            print(f"Error in batch: {str(e)}")
            time.sleep(5)  # Longer delay in case of error
    
    if total_attempts >= max_attempts:
        print(f"Reached maximum attempts ({max_attempts}). Generated {len(dataset)} unique examples.")
    
    return dataset

def parse_generated_content(content):
    """
    Parse the raw text response from GPT-4o into structured conversation data
    
    Args:
        content: Raw text from the API response
    
    Returns:
        List of conversation dictionaries
    """
    conversations = []
    
    # Split the content into individual conversation blocks
    # Look for blocks that start with "CONVERSATION_ID:" 
    conversation_blocks = re.split(r'(?=CONVERSATION_ID:)', content)
    
    for block in conversation_blocks:
        if "CONVERSATION_ID:" not in block:
            continue
            
        try:
            # Extract the conversation ID
            id_match = re.search(r'CONVERSATION_ID:(.*?)(?=TURN_1:|$)', block, re.DOTALL)
            if not id_match:
                continue
            conversation_id = id_match.group(1).strip()
            
            # TURN 1
            # Extract the first supportive message
            turn1_support_match = re.search(r'TURN_1:.*?SUPPORTIVE_MESSAGE:(.*?)RECIPIENT_REPLY:', block, re.DOTALL)
            if not turn1_support_match:
                continue
            turn1_supportive_message = turn1_support_match.group(1).strip()
            
            # Extract the first recipient reply
            turn1_reply_match = re.search(r'TURN_1:.*?RECIPIENT_REPLY:(.*?)LABELS:', block, re.DOTALL)
            if not turn1_reply_match:
                continue
            turn1_recipient_reply = turn1_reply_match.group(1).strip()
            
            # Extract the first turn labels - FIXED REGEX PATTERNS
            turn1_sentiment_match = re.search(r'TURN_1:.*?Sentiment:\s*(Positive|Negative|Neutral)', block, re.DOTALL | re.IGNORECASE)
            turn1_response_needed_match = re.search(r'TURN_1:.*?Response Needed:\s*(Yes|No)', block, re.DOTALL | re.IGNORECASE)
            turn1_crisis_match = re.search(r'TURN_1:.*?Crisis Detection:\s*(Crisis|No Crisis)', block, re.DOTALL | re.IGNORECASE)
            
            if not (turn1_sentiment_match and turn1_response_needed_match and turn1_crisis_match):
                continue
                
            turn1_sentiment = turn1_sentiment_match.group(1).strip()
            turn1_response_needed = turn1_response_needed_match.group(1).strip()
            turn1_crisis_detection = turn1_crisis_match.group(1).strip()
            
            # TURN 2
            # Extract the second supportive message
            turn2_support_match = re.search(r'TURN_2:.*?SUPPORTIVE_MESSAGE:(.*?)RECIPIENT_REPLY:', block, re.DOTALL)
            if not turn2_support_match:
                continue
            turn2_supportive_message = turn2_support_match.group(1).strip()
            
            # Extract the second recipient reply
            turn2_reply_match = re.search(r'TURN_2:.*?RECIPIENT_REPLY:(.*?)LABELS:', block, re.DOTALL)
            if not turn2_reply_match:
                continue
            turn2_recipient_reply = turn2_reply_match.group(1).strip()
            
            # Extract the second turn labels - FIXED REGEX PATTERNS
            turn2_sentiment_match = re.search(r'TURN_2:.*?Sentiment:\s*(Positive|Negative|Neutral)', block, re.DOTALL | re.IGNORECASE)
            turn2_response_needed_match = re.search(r'TURN_2:.*?Response Needed:\s*(Yes|No)', block, re.DOTALL | re.IGNORECASE)
            turn2_crisis_match = re.search(r'TURN_2:.*?Crisis Detection:\s*(Crisis|No Crisis)', block, re.DOTALL | re.IGNORECASE)
            
            if not (turn2_sentiment_match and turn2_response_needed_match and turn2_crisis_match):
                continue
                
            turn2_sentiment = turn2_sentiment_match.group(1).strip()
            turn2_response_needed = turn2_response_needed_match.group(1).strip()
            turn2_crisis_detection = turn2_crisis_match.group(1).strip()
            
            # Create a conversation dictionary
            conversation = {
                "conversation_id": conversation_id,
                "turns": [
                    {
                        "support_message": turn1_supportive_message,
                        "recipient_reply": turn1_recipient_reply,
                        "labels": {
                            "sentiment": turn1_sentiment,
                            "response_needed": turn1_response_needed,
                            "crisis_detection": turn1_crisis_detection
                        }
                    },
                    {
                        "support_message": turn2_supportive_message,
                        "recipient_reply": turn2_recipient_reply,
                        "labels": {
                            "sentiment": turn2_sentiment,
                            "response_needed": turn2_response_needed,
                            "crisis_detection": turn2_crisis_detection
                        }
                    }
                ]
            }
            
            conversations.append(conversation)
            
        except Exception as e:
            print(f"Error parsing a conversation block: {str(e)}")
            continue
    
    return conversations

def split_dataset(dataset, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """
    Split the dataset into training, validation, and test sets
    
    Args:
        dataset: List of conversation dictionaries
        train_ratio: Proportion for training set
        valid_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        
    Returns:
        Dictionary containing the split datasets
    """
    # Ensure ratios sum to 1
    total = train_ratio + valid_ratio + test_ratio
    train_ratio = train_ratio / total
    valid_ratio = valid_ratio / total
    test_ratio = test_ratio / total
    
    # Shuffle the dataset
    random.shuffle(dataset)
    
    # Calculate split indices
    n = len(dataset)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    
    # Split the dataset
    train_set = dataset[:train_end]
    valid_set = dataset[train_end:valid_end]
    test_set = dataset[valid_end:]
    
    return {
        "train": train_set,
        "validation": valid_set,
        "test": test_set
    }

def save_dataset(dataset_splits, output_dir="grief_support_dataset"):
    """
    Save the dataset splits to JSON files
    
    Args:
        dataset_splits: Dictionary containing train, validation, and test datasets
        output_dir: Directory to save the files
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split to a separate file
    for split_name, split_data in dataset_splits.items():
        filename = os.path.join(output_dir, f"{split_name}.json")
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        
        print(f"{split_name.capitalize()} set saved to {filename} with {len(split_data)} examples.")
    
    # Save the complete dataset
    all_data = []
    for split_data in dataset_splits.values():
        all_data.extend(split_data)
        
    complete_filename = os.path.join(output_dir, "complete_dataset.json")
    with open(complete_filename, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    
    print(f"Complete dataset saved to {complete_filename} with {len(all_data)} examples.")

def print_dataset_stats(dataset_splits):
    """
    Print statistics about the dataset
    
    Args:
        dataset_splits: Dictionary containing train, validation, and test datasets
    """
    total_conversations = sum(len(split) for split in dataset_splits.values())
    
    print(f"\nDataset Statistics:")
    print(f"Total conversations: {total_conversations}")
    
    for split_name, split_data in dataset_splits.items():
        print(f"\n{split_name.capitalize()} set: {len(split_data)} conversations ({len(split_data)/total_conversations*100:.1f}%)")
        
        # Count labels for both turns
        for turn_idx in [0, 1]:
            turn_name = f"Turn {turn_idx+1}"
            
            # Initialize counters for expected values only (no "Other" category)
            sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
            response_needed_counts = {"Yes": 0, "No": 0}
            crisis_counts = {"Crisis": 0, "No Crisis": 0}
            
            for item in split_data:
                try:
                    # Get label values and normalize them
                    sentiment = item["turns"][turn_idx]["labels"]["sentiment"].strip().capitalize()
                    response = item["turns"][turn_idx]["labels"]["response_needed"].strip().capitalize()
                    crisis = item["turns"][turn_idx]["labels"]["crisis_detection"].strip().capitalize()
                    
                    # Fix capitalization for "No Crisis"
                    if crisis.lower() == "no crisis":
                        crisis = "No Crisis"
                    
                    # Count only if value matches one of our expected categories
                    if sentiment in sentiment_counts:
                        sentiment_counts[sentiment] += 1
                    
                    if response in response_needed_counts:
                        response_needed_counts[response] += 1
                        
                    if crisis in crisis_counts:
                        crisis_counts[crisis] += 1
                except (KeyError, IndexError):
                    # Skip this item if there's a problem
                    continue
            
            print(f"\n  {turn_name} Distribution:")
            print(f"  Sentiment: {sentiment_counts}")
            print(f"  Response Needed: {response_needed_counts}")
            print(f"  Crisis Detection: {crisis_counts}")

def main(total_examples=100, batch_size=10, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """
    Main function to generate and save the dataset
    
    Args:
        total_examples: Total number of conversations to generate
        batch_size: Number of conversations to generate in each batch
        train_ratio: Proportion for training set
        valid_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    print(f"Generating a dataset with {total_examples} unique conversations in batches of {batch_size}...")
    
    # Generate the dataset
    dataset = generate_dataset_batch(batch_size, total_examples)
    
    # Perform basic validation
    print(f"Generated {len(dataset)} valid and unique conversations out of {total_examples} requested.")
    
    # Split the dataset
    dataset_splits = split_dataset(dataset, train_ratio, valid_ratio, test_ratio)
    
    # Print statistics
    print_dataset_stats(dataset_splits)
    
    # Save the dataset
    save_dataset(dataset_splits)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate a unique grief support conversation dataset')
    parser.add_argument('--total', type=int, default=1000, help='Total number of conversations to generate')
    parser.add_argument('--batch-size', type=int, default=10, help='Number of conversations per batch')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Proportion for training set')
    parser.add_argument('--valid-ratio', type=float, default=0.15, help='Proportion for validation set')
    parser.add_argument('--test-ratio', type=float, default=0.15, help='Proportion for test set')
    
    args = parser.parse_args()
    
    main(
        total_examples=args.total,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio
    )