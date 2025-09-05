import re
import joblib
from collections import Counter

# --- ALL HELPER FUNCTIONS MUST BE IN THIS FILE ---

def improve_training_labels(train_data, test_data):
    """
    Improves BIO labels to better capture multi-word events by applying regex patterns.
    """
    
    # Patterns for multi-word events that should be labeled together
    multi_word_patterns = [
        # Adjective + Event
        (r'\b(final|initial|first|last|next|upcoming|scheduled|planned)\s+(review|meeting|presentation|call|demo|session|discussion|interview)\b', 'EVENT'),
        (r'\b(team|company|client|project|weekly|monthly|daily|quarterly|annual)\s+(meeting|review|session|call|dinner|lunch|party)\b', 'EVENT'),
        (r'\b(code|design|performance|budget|status|progress)\s+(review|meeting|session|discussion)\b', 'EVENT'),
        
        # Compound events
        (r'\b(kick-off|kickoff|follow-up|followup|stand-up|standup)\s*(meeting|session|call)?\b', 'EVENT'),
        (r'\b(brainstorm|brainstorming|planning|training|onboarding)\s+(session|meeting|workshop|call)\b', 'EVENT'),
        
        # Event with descriptor
        (r'\b(quick|brief|short|long|important|urgent)\s+(call|meeting|chat|sync|discussion)\b', 'EVENT'),
        (r'\b(one-on-one|1-on-1|all-hands|all hands on deck)\s*(meeting|call|session)?\b', 'EVENT'),
        
        # Department/role + event
        (r'\b(hr|marketing|sales|engineering|product|design)\s+(meeting|review|session|interview)\b', 'EVENT'),
    ]
    
    def enhance_sentence_labels(tokens, labels):
        """Enhance BIO labels for a single sentence"""
        text = ' '.join(tokens).lower()
        new_labels = labels.copy()
        
        # Apply each pattern
        for pattern, label_type in multi_word_patterns:
            for match in re.finditer(pattern, text):
                match_text = match.group().strip()
                match_words = match_text.split()
                
                start_idx = find_token_span(tokens, match_words)
                
                if start_idx != -1:
                    if len(match_words) == 1:
                        new_labels[start_idx] = f'B-{label_type}'
                    else:
                        new_labels[start_idx] = f'B-{label_type}'
                        for i in range(1, len(match_words)):
                            if start_idx + i < len(new_labels):
                                new_labels[start_idx + i] = f'I-{label_type}'
        return new_labels
    
    def find_token_span(tokens, target_words):
        """Find where a sequence of words appears in the token list (case-insensitive)"""
        tokens_lower = [t.lower() for t in tokens]
        target_lower = [w.lower() for w in target_words]
        
        for i in range(len(tokens_lower) - len(target_lower) + 1):
            if tokens_lower[i:i+len(target_lower)] == target_lower:
                return i
        return -1

    print("Enhancing training data labels...")
    enhanced_train_data = []
    train_improvements = 0
    for tokens, labels in train_data:
        new_labels = enhance_sentence_labels(tokens, labels)
        if new_labels != labels:
            train_improvements += 1
        enhanced_train_data.append((tokens, new_labels))
    print(f"Improved labels in {train_improvements} training sentences.")
    
    print("Enhancing test data labels...")
    enhanced_test_data = []
    test_improvements = 0
    for tokens, labels in test_data:
        new_labels = enhance_sentence_labels(tokens, labels)
        if new_labels != labels:
            test_improvements += 1
        enhanced_test_data.append((tokens, new_labels))
    print(f"Improved labels in {test_improvements} test sentences.")

    return enhanced_train_data, enhanced_test_data

def analyze_label_distribution(data, name):
    """Analyzes the distribution of BIO labels."""
    print(f"\n--- {name} Label Analysis ---")
    label_counts = Counter(label for _, labels in data for label in labels)
    for label, count in label_counts.most_common():
        percentage = (count / sum(label_counts.values())) * 100
        print(f"  {label:<10} | {count:5,} ({percentage:4.1f}%)")

def create_better_training_examples():
    """Creates additional high-quality, manually labeled training examples."""
    return [
        # Multi-word events
        (["Final", "review", "of", "the", "project", "with", "the", "team", "."], 
         ["B-EVENT", "I-EVENT", "I-EVENT", "I-EVENT", "I-EVENT", "O", "O", "O", "O"]),
        
        (["Dinner", "with", "Alex", "and", "Sam", "."], 
         ["B-EVENT", "O", "O", "O", "O", "O"]),

        (["A", "meeting", "with", "Jessica", "tomorrow", "."], 
         ["O", "B-EVENT", "O", "O", "B-DATE", "O"]),

        (["Sync", "up", "with", "Chris", "on", "Friday", "."],
         ["B-EVENT", "I-EVENT", "O", "O", "O", "B-DATE", "O"]),
        
        (["Lunch", "with", "the", "marketing", "team", "and", "David", "."],
         ["B-EVENT", "O", "O", "O", "O", "O", "O", "O"]),

        (["Company", "party", "tonight", "at", "8", "pm", "."], 
         ["B-EVENT", "I-EVENT", "B-DATE", "O", "B-TIME", "I-TIME", "O"]),
        
        (["Schedule", "a", "team", "meeting", "for", "next", "Friday", "."], 
         ["O", "O", "B-EVENT", "I-EVENT", "O", "B-DATE", "I-DATE", "O"]),
         
        (["Quick", "call", "with", "the", "client", "tomorrow", "morning", "."], 
         ["B-EVENT", "I-EVENT", "O", "O", "O", "B-DATE", "I-DATE", "O"]),
         
        (["One-on-one", "meeting", "with", "my", "manager", "at", "2", "pm", "."], 
         ["B-EVENT", "I-EVENT", "O", "O", "O", "O", "B-TIME", "I-TIME", "O"]),
         
        (["Code", "review", "session", "scheduled", "for", "this", "afternoon", "."], 
         ["B-EVENT", "I-EVENT", "I-EVENT", "O", "O", "B-DATE", "I-DATE", "O"]), # Assuming "this afternoon" is a date
         
        (["All-hands", "meeting", "on", "Friday", "at", "10", "am", "."], 
         ["B-EVENT", "I-EVENT", "O", "B-DATE", "O", "B-TIME", "I-TIME", "O"]),
    ]

# --- MAIN FUNCTION ---

def enhance_your_training_data():
    """Main function to load, enhance, and save your training data"""
    print("--- Starting Data Enhancement Process ---")
    try:
        train_data = joblib.load("xp/xp_train_data.pkl") 
        test_data = joblib.load("xp/xp_test_data.pkl")
    except FileNotFoundError:
        print("\nERROR: Could not find 'xp/xp_train_data.pkl' or 'xp/xp_test_data.pkl'.")
        print("Please ensure your data files are in the 'xp' folder before running.")
        return

    analyze_label_distribution(train_data, "Original Training Data")
    
    enhanced_train, enhanced_test = improve_training_labels(train_data, test_data)
    
    additional_examples = create_better_training_examples()
    enhanced_train.extend(additional_examples)
    
    analyze_label_distribution(enhanced_train, "Enhanced Training Data")
    
    # Save the enhanced data back, overwriting the old files
    joblib.dump(enhanced_train, "xp/xp_train_data.pkl")
    joblib.dump(enhanced_test, "xp/xp_test_data.pkl")
    
    print("\n\n--- Data Enhancement Complete ---")
    print("Enhanced training data has been saved, overwriting the old files!")

if __name__ == "__main__":
    enhance_your_training_data()