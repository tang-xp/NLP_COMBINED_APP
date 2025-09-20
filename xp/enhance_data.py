import re, joblib
from collections import Counter

def improve_training_labels(train_data, test_data):
    """Enhanced regex-based relabeling to capture multi-word events including standalone event words."""
    patterns = [
        # Original patterns
        (r'\b(final|initial|first|last|next|upcoming|scheduled|planned)\s+(review|meeting|presentation|call|demo|session|discussion|interview)\b','EVENT'),
        (r'\b(team|company|client|project|weekly|monthly|daily|quarterly|annual)\s+(meeting|review|session|call|dinner|lunch|party)\b','EVENT'),
        (r'\b(code|design|performance|budget|status|progress)\s+(review|meeting|session|discussion)\b','EVENT'),
        (r'\b(kick[- ]?off|follow[- ]?up|stand[- ]?up)\s*(meeting|session|call)?\b','EVENT'),
        (r'\b(brainstorm(?:ing)?|planning|training|onboarding)\s+(session|meeting|workshop|call)\b','EVENT'),
        (r'\b(quick|brief|short|long|important|urgent)\s+(call|meeting|chat|sync|discussion)\b','EVENT'),
        (r'\b(one-on-one|1-on-1|all-hands|all hands on deck)\s*(meeting|call|session)?\b','EVENT'),
        (r'\b(hr|marketing|sales|engineering|product|design)\s+(meeting|review|session|interview)\b','EVENT'),
        
        # NEW: Patterns for standalone event words followed by "with"
        (r'\b(dinner|lunch|breakfast|meeting|call|session|interview|presentation|demo)\s+with\s+\w+(?:\s+and\s+\w+)*\b','EVENT'),
        (r'\b(coffee|drinks|chat|sync|catchup|catch-up)\s+with\s+\w+(?:\s+and\s+\w+)*\b','EVENT'),
        
        # NEW: More flexible event patterns
        (r'\b(conference|workshop|seminar|webinar|training)\s+(?:on|about|with|for)\s+\w+\b','EVENT'),
        (r'\b(party|celebration|gathering|meetup)\s+(?:with|for|at)\s+\w+\b','EVENT'),
        
        # NEW: Event words that can stand alone or with prepositions
        (r'\b(appointment|consultation|checkup|visit)\s+(?:with|at|for)\s+\w+\b','EVENT'),
        (r'\b(rehearsal|practice|drill|exercise)\s+(?:with|for|at)\s+\w+\b','EVENT'),
        
        # NEW: Time-based events
        (r'\b(morning|afternoon|evening|tonight)\s+(meeting|call|session|dinner|lunch)\b','EVENT'),
        
        # NEW: Catch broader event contexts
        (r'\b(group|team|staff|board)\s+(meeting|dinner|lunch|session|retreat)\b','EVENT'),
    ]

    def find_span(tokens, target):
        """Find the starting index of target words in tokens (case-insensitive)."""
        low_toks = [t.lower() for t in tokens]
        low_tgt = [w.lower() for w in target]
        for i in range(len(low_toks) - len(low_tgt) + 1):
            if low_toks[i:i+len(low_tgt)] == low_tgt:
                return i
        return -1

    def enhance(tokens, labels):
        """Apply regex patterns to enhance labels."""
        txt = ' '.join(tokens).lower()
        new = labels.copy()
        
        for pat, lbl in patterns:
            for m in re.finditer(pat, txt):
                # Get the matched phrase and split into words
                matched_phrase = m.group().strip()
                words = matched_phrase.split()
                
                # Find where this phrase starts in our token list
                s = find_span(tokens, words)
                if s != -1:
                    # Label the entire matched phrase
                    new[s] = f"B-{lbl}"
                    for i in range(1, len(words)):
                        if s + i < len(new):
                            new[s + i] = f"I-{lbl}"
        
        return new

    def process(data, name):
        print(f"Enhancing {name} data...")
        improved, count = [], 0
        for toks, labs in data:
            new_labs = enhance(toks, labs)
            if new_labs != labs: 
                count += 1
            improved.append((toks, new_labs))
        print(f"Improved {count} sentences.")
        return improved

    return process(train_data, "training"), process(test_data, "test")


def analyze_distribution(data, name):
    print(f"\n--- {name} Label Analysis ---")
    c = Counter(l for _, labs in data for l in labs)
    total = sum(c.values())
    for lbl, cnt in c.most_common():
        print(f"{lbl:<10} | {cnt:5} ({cnt/total*100:4.1f}%)")


def create_manual_examples():
    """Create manual training examples for common multi-word events."""
    return [
        # Fixed: "Dinner with John" should be fully labeled as EVENT
        (["Dinner","with","John","."],
         ["B-EVENT","I-EVENT","I-EVENT","O"]),
        
        (["Lunch","with","Sarah","and","Mike","tomorrow","."],
         ["B-EVENT","I-EVENT","I-EVENT","I-EVENT","I-EVENT","B-DATE","O"]),
         
        (["Coffee","with","the","team","this","afternoon","."],
         ["B-EVENT","I-EVENT","I-EVENT","I-EVENT","B-TIME","I-TIME","O"]),
        
        (["Final","review","of","the","project","with","the","team","."],
         ["B-EVENT","I-EVENT","I-EVENT","I-EVENT","I-EVENT","I-EVENT","I-EVENT","I-EVENT","O"]),
        
        (["A","meeting","with","Jessica","tomorrow","."],
         ["O","B-EVENT","I-EVENT","I-EVENT","B-DATE","O"]),
         
        (["Staff","party","with","everyone","next","Friday","."],
         ["B-EVENT","I-EVENT","I-EVENT","I-EVENT","B-DATE","I-DATE","O"]),
         
        # Additional examples for better coverage
        (["Morning","standup","with","the","development","team","."],
         ["B-EVENT","I-EVENT","I-EVENT","I-EVENT","I-EVENT","I-EVENT","O"]),
         
        (["Client","presentation","scheduled","for","next","week","."],
         ["B-EVENT","I-EVENT","O","O","B-DATE","I-DATE","O"]),
         
        (["One-on-one","with","my","manager","at","3","PM","."],
         ["B-EVENT","I-EVENT","I-EVENT","I-EVENT","O","B-TIME","I-TIME","O"]),
    ]


def enhance_your_training_data():
    print("--- Starting Data Enhancement ---")
    try:
        train = joblib.load("xp/xp_train_data.pkl")
        test  = joblib.load("xp/xp_test_data.pkl")
    except FileNotFoundError:
        print("ERROR: Missing xp_train_data.pkl or xp_test_data.pkl")
        return

    analyze_distribution(train, "Original Training")
    enhanced_train, enhanced_test = improve_training_labels(train, test)
    
    # Add manual examples
    manual_examples = create_manual_examples()
    enhanced_train.extend(manual_examples)
    print(f"Added {len(manual_examples)} manual training examples.")
    
    analyze_distribution(enhanced_train, "Enhanced Training")

    joblib.dump(enhanced_train, "xp/xp_train_data.pkl")
    joblib.dump(enhanced_test,  "xp/xp_test_data.pkl")
    print("✔ Enhancement complete. Files overwritten.")


if __name__ == "__main__":
    enhance_your_training_data()