import spacy
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import joblib
import os
from collections import Counter
import warnings
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
import numpy as np

# --- Setup and Data Loading ---
warnings.filterwarnings('ignore')
script_dir = os.path.dirname(os.path.abspath(__file__))
from .xp_features import FeatureExtractor
from sklearn.metrics import make_scorer

def load_data_safely():
    """Load training and test data with proper error handling"""
    try:
        print("Loading official train and test datasets...")
        train_path = os.path.join(script_dir, "xp_train_data.pkl")
        test_path = os.path.join(script_dir, "xp_test_data.pkl")
        train_data = joblib.load(train_path)
        test_data = joblib.load(test_path)
        
        if not train_data or not test_data:
            raise ValueError("Empty dataset loaded")
            
        print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples.")
        return train_data, test_data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Load data and initialize components
train_data, test_data = load_data_safely()

# Load spaCy model with error handling
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please install it with:")
    print("python -m spacy download en_core_web_sm")
    raise

feature_extractor = FeatureExtractor()

def sent2features(sent_tokens):
    """
    Convert sentence tokens to features, with robust handling for empty or mismatched inputs.
    """
    # --- FIX 1: Handle empty input gracefully ---
    if not sent_tokens:
        return [] # Always return a list

    doc = nlp(" ".join(sent_tokens))
    
    # --- FIX 2: Correctly handle tokenization mismatches ---
    # If spaCy's tokenization doesn't match the original, we cannot safely create features.
    # The safest action is to return an empty list, which will cause the sentence to be skipped.
    if len(doc) != len(sent_tokens):
        return [] # Return an empty list to signal a mismatch

    # If everything is fine, proceed with the full feature extraction
    chunk_info = [{'tag': 'O', 'is_event_chunk': False} for _ in doc]
    for chunk in doc.noun_chunks:
        has_event_keyword = any(token.text.lower() in feature_extractor.event_words for token in chunk)
        if len(chunk) == 1:
            chunk_info[chunk.start]['tag'] = 'S-CHUNK'
        else:
            chunk_info[chunk.start]['tag'] = 'B-CHUNK'
            for i in range(chunk.start + 1, chunk.end - 1):
                chunk_info[i]['tag'] = 'I-CHUNK'
            chunk_info[chunk.end - 1]['tag'] = 'E-CHUNK'
        if has_event_keyword:
            for i in range(chunk.start, chunk.end):
                chunk_info[i]['is_event_chunk'] = True

    return [feature_extractor.enhanced_word_features(doc, i, chunk_info) for i in range(len(doc))]

    # Step 3: E
# --- Feature Preparation ---
print("\nPreparing features...")
X_train_tokens = [sent for sent, labels in train_data]
y_train = [labels for sent, labels in train_data]

X_test_tokens = [sent for sent, labels in test_data]
y_test = [labels for sent, labels in test_data]

X_train_features = []
y_train_clean = []

# Process training data with progress tracking
for i, (sent_tokens, sent_labels) in enumerate(zip(X_train_tokens, y_train)):
    features = sent2features(sent_tokens)
    
    # Check for tokenization mismatches
    if len(features) != len(sent_labels):
        print(f"\n--- WARNING: Mismatch found in training sentence {i+1} ---")
        print(f"  Original Tokens ({len(sent_tokens)}): {sent_tokens}")
        print(f"  spaCy Tokens ({len(features)}): {[f['word.lower()'] for f in features]}")
        print(f"  Labels ({len(sent_labels)}): {sent_labels}")
        print("  --> This sentence will be SKIPPED.")
        continue # Skip this sentence
    
    X_train_features.append(features)
    y_train_clean.append(sent_labels)

X_test_features = []
y_test_clean = []

for i, (sent_tokens, sent_labels) in enumerate(zip(X_test_tokens, y_test)):
    features = sent2features(sent_tokens)
    
    if len(features) != len(sent_labels):
        print(f"\n--- WARNING: Mismatch found in test sentence {i+1} ---")
        print(f"  --> This sentence will be SKIPPED.")
        continue
        
    X_test_features.append(features)
    y_test_clean.append(sent_labels)

# Use the new feature lists for the rest of the script
X_train = X_train_features
X_test = X_test_features


print(f"\nSuccessfully processed {len(X_train)}/{len(train_data)} training sentences")
print(f"Successfully processed {len(X_test)}/{len(test_data)} test sentences")

# --- Data Analysis for NER ---
print("\nAnalyzing label distribution...")
label_counts = Counter()
for labels in y_train_clean:
    for label in labels:
        label_counts[label] += 1

print("Label distribution:")
for label, count in label_counts.most_common():
    percentage = (count / sum(label_counts.values())) * 100
    print(f"  {label}: {count:,} ({percentage:.1f}%)")

# --- Smart Data Balancing for NER ---
def balance_ner_data(X_train, y_train, max_ratio=3):
    """
    Balance training data specifically for NER tasks.
    Gives more weight to sentences with rare entity types.
    """
    balanced_X, balanced_y = [], []
    
    # Calculate entity rarity
    entity_counts = Counter()
    for labels in y_train:
        entities = [label for label in labels if label != "O"]
        for entity in entities:
            entity_counts[entity] += 1
    
    print(f"Entity distribution: {entity_counts}")
    
    for x_sent, y_sent in zip(X_train, y_train):
        balanced_X.append(x_sent)
        balanced_y.append(y_sent)
        
        # Find rare entities in this sentence
        sentence_entities = [label for label in y_sent if label != "O"]
        if sentence_entities:
            # Calculate augmentation factor based on rarity
            min_entity_count = min(entity_counts[entity] for entity in set(sentence_entities))
            max_entity_count = max(entity_counts.values())
            
            if min_entity_count < max_entity_count / 10:  # If very rare
                augment_factor = min(max_ratio, 2)
                for _ in range(augment_factor):
                    balanced_X.append(x_sent)
                    balanced_y.append(y_sent)
    
    print(f"Balanced training size: {len(balanced_X)} (from {len(y_train)})")
    return balanced_X, balanced_y

print("\nBalancing training data for NER...")
X_train_balanced, y_train_balanced = balance_ner_data(X_train, y_train_clean)

# --- HYPERPARAMETER TUNING WITH RANDOMIZED SEARCH ---
print("\nTraining CRF model with Hyperparameter Search...")

# Define the CRF model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)

# Define parameter search space optimized for NER
# c1: L1 regularization (for feature sparsity)
# c2: L2 regularization (for feature smoothness)
params_space = {
    'c1': uniform(0, 0.5),    # Slightly wider range for NER
    'c2': uniform(0, 0.3)     # L2 regularization for stability
}

# Use F1 score weighted by support (better for imbalanced NER data)
f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')

# Set up randomized search with cross-validation
print("Setting up RandomizedSearchCV...")
rs = RandomizedSearchCV(
    crf, 
    params_space,
    cv=3,                    # 3-fold cross-validation
    verbose=1,               # Show progress
    n_jobs=-1,              # Use all CPU cores
    n_iter=20,              # Try 20 random combinations
    scoring=f1_scorer,
    random_state=42         # For reproducibility
)

# Run the hyperparameter search
print("Running hyperparameter search (this may take several minutes)...")
rs.fit(X_train_balanced, y_train_balanced)

# Get the best model
crf = rs.best_estimator_
print(f"\nBest parameters found: {rs.best_params_}")
print(f"Best cross-validation F1-score: {rs.best_score_:.4f}")

# --- Final Evaluation on Test Set ---
print("\n--- Final Model Evaluation on Test Set ---")
y_pred = crf.predict(X_test)

# Get all entity labels (excluding 'O')
all_labels = set()
for labels in y_train_clean + y_test_clean:
    all_labels.update(labels)
all_labels.discard('O')
sorted_labels = sorted(all_labels, key=lambda name: (name[1:], name[0]))

if sorted_labels:
    # Detailed classification report
    report = metrics.flat_classification_report(
        y_test_clean, y_pred, 
        labels=sorted_labels, 
        digits=3
    )
    print("Classification Report:")
    print(report)
    
    # Overall metrics
    overall_f1 = metrics.flat_f1_score(y_test_clean, y_pred, average='weighted')
    overall_precision = metrics.flat_precision_score(y_test_clean, y_pred, average='weighted')
    overall_recall = metrics.flat_recall_score(y_test_clean, y_pred, average='weighted')
    
    print(f"\nOverall NER Performance:")
    print(f"Precision: {overall_precision:.3f}")
    print(f"Recall: {overall_recall:.3f}")
    print(f"F1-Score: {overall_f1:.3f}")
    
    # Entity-specific analysis
    print(f"\nDetailed Entity Analysis:")
    for label in sorted_labels:
        label_f1 = metrics.flat_f1_score(y_test_clean, y_pred, labels=[label], average='weighted')
        print(f"  {label}: F1 = {label_f1:.3f}")

else:
    print("Warning: No named entity labels found in the dataset!")

# --- Feature Importance Analysis ---
print("\n--- Most Informative Features Per Entity Type ---")
if hasattr(crf, 'state_features_'):
    try:
        for label in sorted_labels:
            weights = crf.state_features_.get(label, {})
            
            if not weights:
                print(f"\n{label}: No specific features found")
                continue

            # Top positive and negative features
            top_positive = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
            top_negative = sorted(weights.items(), key=lambda x: x[1])[:5]

            print(f"\n{label}:")
            print("  Features that PROMOTE this entity:")
            for feature, weight in top_positive:
                print(f"    {weight:+.3f} {feature}")
            
            print("  Features that DISCOURAGE this entity:")
            for feature, weight in top_negative:
                print(f"    {weight:+.3f} {feature}")

    except Exception as e:
        print(f"Could not extract feature weights: {e}")

# --- Save the Optimized Model ---
MODEL_FILE = os.path.join(script_dir, "xp_crf_model.joblib")
print(f"\nSaving optimized model to {MODEL_FILE}...")

try:
    joblib.dump(crf, MODEL_FILE)
    print(f"Model saved successfully!")
    print(f"Best parameters: {rs.best_params_}")
    print(f"Cross-validation F1: {rs.best_score_:.4f}")
    print(f"Test F1: {overall_f1:.4f}")
    
except Exception as e:
    print(f"Error saving model: {e}")

print("\n=== NER Training Complete ===")
print("Your model is now optimized with proper hyperparameter search!")