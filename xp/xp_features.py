import re

class FeatureExtractor:
    def __init__(self):
        # Expanded event words with multi-word context clues
        self.event_words = {
            "meeting", "conference", "appointment", "deadline", "review", "lunch", "dinner", "call", "presentation",
            "workshop", "sync", "flight", "debrief", "demo", "planning", "session", "kick-off", "kickoff",
            "bbq", "party", "report", "barbecue", "fishing", "run", "groceries", "stand-up", "standup",
            "interview", "training", "seminar", "celebration", "ceremony", "event", "gathering", "ceremony",
            "discussion", "brainstorm", "brainstorming", "huddle", "catchup", "catch-up", "check-in", "checkin"
        }
        
        # Words that often precede events (determiners, adjectives, etc.)
        self.event_modifiers = {
            "final", "initial", "first", "last", "next", "upcoming", "scheduled", "planned", "urgent",
            "weekly", "monthly", "daily", "annual", "quarterly", "company", "team", "client", "project",
            "quick", "brief", "long", "important", "mandatory", "optional", "follow-up", "followup"
        }
        
        # Action words that often indicate events
        self.action_words = {
            "schedule", "plan", "organize", "arrange", "book", "reserve", "set", "meet", "discuss", "present"
        }

        self.common_names = {
            "alex", "sam", "david", "jane", "jessica", "mike", "sarah", "chris", 
            "john", "emily", "michael", "lisa", "robert", "mary", "james", "linda",
            "william", "susan", "richard", "karen", "charles", "patricia", "joseph"
        }


    def enhanced_word_features(self, sent, i, chunk_info):
        token = sent[i]
        word_lower = token.text.lower()
        
        token_chunk_info = chunk_info[i] 
        
        chunk_tag = token_chunk_info['tag']
        is_event_chunk = token_chunk_info['is_event_chunk']

        # Basic features
        features = {
            "bias": 1.0,
            "word.lower()": word_lower,
            "word.isupper()": token.is_upper,
            "word.istitle()": token.is_title,
            "word.isdigit()": token.is_digit,
            "word_shape": "".join(['X' if c.isupper() else 'x' if c.islower() else 'd' if c.isdigit() else c for c in token.text]),
            "postag": token.pos_,
            "dep": token.dep_,
            "chunk_tag": chunk_tag,
            'is_event_chunk': is_event_chunk,
            "is_event_word": word_lower in self.event_words,
            "is_event_modifier": word_lower in self.event_modifiers,
            "is_action_word": word_lower in self.action_words,
            "is_phrase_head": token.head == token,
            "lemma": token.lemma_.lower(),
        }

        # === MULTI-WORD EVENT FEATURES ===
        
        # 1. Bigram features (current + next word)
        if i < len(sent) - 1:
            next_token = sent[i + 1]
            next_word_lower = next_token.text.lower()
            
            # Bigram combinations that often form events
            bigram = f"{word_lower}_{next_word_lower}"
            features["word_bigram"] = bigram
            features["next_postag"] = next_token.pos_
            features["next_word.lower()"] = next_word_lower
            features["next_is_title"] = next_token.is_title
            features["next_is_event_word"] = next_word_lower in self.event_words
            
            # Check if this forms a common event pattern
            features["forms_event_bigram"] = (
                (word_lower in self.event_modifiers and next_word_lower in self.event_words) or
                (word_lower in self.event_words and next_word_lower in self.event_words) or
                (word_lower in self.action_words and next_word_lower in self.event_words)
            )
        else:
            features["EOS"] = True

        # 2. Previous word context
        if i > 0:
            prev_token = sent[i - 1]
            prev_word_lower = prev_token.text.lower()
            
            features["prev_postag"] = prev_token.pos_
            features["prev_word.lower()"] = prev_word_lower
            features["prev_is_title"] = prev_token.is_title
            features["prev_is_event_word"] = prev_word_lower in self.event_words
            features["prev_is_event_modifier"] = prev_word_lower in self.event_modifiers
            
            # Check if previous word suggests this should be part of an event
            features["follows_event_modifier"] = prev_word_lower in self.event_modifiers
            features["follows_event_word"] = prev_word_lower in self.event_words
        else:
            features["BOS"] = True

        # 3. Trigram features (for longer event phrases)
        if i > 0 and i < len(sent) - 1:
            prev_word = sent[i - 1].text.lower()
            next_word = sent[i + 1].text.lower()
            trigram = f"{prev_word}_{word_lower}_{next_word}"
            features["trigram"] = trigram

        # 4. Dependency-based features for multi-word events
        # Check if this token modifies or is modified by event words
        features["modifies_event_word"] = False
        features["modified_by_event_word"] = False
        
        for child in token.children:
            if child.text.lower() in self.event_words:
                features["modifies_event_word"] = True
                break
                
        if token.head != token and token.head.text.lower() in self.event_words:
            features["modified_by_event_word"] = True

        # 5. Position within noun chunks (critical for multi-word events!)
        if hasattr(token, 'doc'):
            for chunk in token.doc.noun_chunks:
                if token in chunk:
                    chunk_position = list(chunk).index(token)
                    chunk_length = len(list(chunk))
                    
                    features["chunk_position"] = chunk_position
                    features["chunk_length"] = chunk_length
                    features["is_chunk_start"] = chunk_position == 0
                    features["is_chunk_end"] = chunk_position == chunk_length - 1
                    features["is_chunk_middle"] = 0 < chunk_position < chunk_length - 1
                    
                    # Check if chunk contains event words
                    chunk_words = [t.text.lower() for t in chunk]
                    features["chunk_has_event_word"] = any(w in self.event_words for w in chunk_words)
                    features["chunk_has_modifier"] = any(w in self.event_modifiers for w in chunk_words)
                    break

        # 6. Capitalization patterns (important for event titles)
        if i > 0 and i < len(sent) - 1:
            prev_cap = sent[i - 1].is_title
            curr_cap = token.is_title  
            next_cap = sent[i + 1].is_title
            
            # Sequential capitalization often indicates event titles
            features["title_sequence"] = prev_cap and curr_cap
            features["title_sequence_3"] = prev_cap and curr_cap and next_cap

        # 7. Word length and complexity (longer phrases tend to be events)
        features["word_length"] = len(token.text)
        features["is_long_word"] = len(token.text) > 6

        return features

    def get_extended_context_features(self, sent, i, window=2):
        """Get extended context features for better sequence modeling"""
        features = {}
        
        # Look at wider context window
        for j in range(max(0, i - window), min(len(sent), i + window + 1)):
            if j != i:
                offset = j - i
                context_token = sent[j]
                context_word = context_token.text.lower()
                
                features[f"context_{offset}_word"] = context_word
                features[f"context_{offset}_pos"] = context_token.pos_
                features[f"context_{offset}_is_event"] = context_word in self.event_words
                features[f"context_{offset}_is_modifier"] = context_word in self.event_modifiers
        
        return features