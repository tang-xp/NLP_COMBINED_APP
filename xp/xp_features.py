import spacy
from collections import Counter

class FeatureExtractor:
    def __init__(self):
        # A set of words that typically indicate an event
        self.event_words = {
            "meeting", "conference", "appointment", "deadline", "review", "lunch",
            "dinner", "call", "presentation", "workshop", "sync", "flight",
            "debrief", "demo", "planning", "session", "kick-off", "kickoff",
            "bbq", "party", "report", "barbecue", "fishing", "run", "groceries",
            "stand-up", "standup", "interview", "training", "seminar",
            "celebration", "ceremony", "event", "gathering", "discussion",
            "brainstorm", "brainstorming", "huddle", "catchup", "catch-up",
            "check-in", "checkin"
        }
        # A set of words that modify events
        self.event_modifiers = {
            "final", "initial", "first", "last", "next", "upcoming",
            "scheduled", "planned", "urgent", "weekly", "monthly", "daily",
            "annual", "quarterly", "company", "team", "client", "project",
            "quick", "brief", "long", "important", "mandatory", "optional",
            "follow-up", "followup"
        }
        # A set of action verbs that can precede an event
        self.action_words = {
            "schedule", "plan", "organize", "arrange", "book", "reserve",
            "set", "meet", "discuss", "present"
        }
        # A set of common names to identify people
        self.common_names = {
            "alex", "sam", "david", "jane", "jessica", "mike", "sarah",
            "chris", "john", "emily", "michael", "lisa", "robert", "mary",
            "james", "linda", "william", "susan", "richard", "karen",
            "charles", "patricia", "joseph"
        }
        # New set of words for relative time expressions
        self.relative_time_words = {"hour", "hours", "minute", "minutes", "later"}

    def enhanced_word_features(self, sent, i, chunk_info):
        """
        Extracts a dictionary of features for the token at index `i` in the sentence.
        
        Args:
            sent (spacy.tokens.doc.Doc): The spacy Doc object for the sentence.
            i (int): The index of the current token.
            chunk_info (list): A list of dictionaries containing noun chunk tags and event chunk info.
            
        Returns:
            dict: A dictionary of features for the current token.
        """
        token = sent[i]
        w = token.text.lower()
        info = chunk_info[i]

        features = {
            "bias": 1.0,
            "word.lower()": w,
            "word.isupper()": token.is_upper,
            "word.istitle()": token.is_title,
            "word.isdigit()": token.is_digit,
            "word_shape": "".join(
                "X" if c.isupper() else "x" if c.islower()
                else "d" if c.isdigit() else c for c in token.text
            ),
            "postag": token.pos_,
            "dep": token.dep_,
            "chunk_tag": info["tag"],
            "is_event_chunk": info["is_event_chunk"],
            "is_event_word": w in self.event_words,
            "is_event_modifier": w in self.event_modifiers,
            "is_action_word": w in self.action_words,
            "is_phrase_head": token.head == token,
            "lemma": token.lemma_.lower(),
            # New feature for relative time words
            "is_relative_time_word": w in self.relative_time_words,
        }

        # Features for the next word
        if i < len(sent) - 1:
            nxt = sent[i + 1]
            nxt_w = nxt.text.lower()
            features.update({
                "word_bigram": f"{w}_{nxt_w}",
                "next_postag": nxt.pos_,
                "next_word.lower()": nxt_w,
                "next_is_title": nxt.is_title,
                "next_is_event_word": nxt_w in self.event_words,
                "forms_event_bigram":
                    (w in self.event_modifiers and nxt_w in self.event_words)
                    or (w in self.event_words and nxt_w in self.event_words)
                    or (w in self.action_words and nxt_w in self.event_words)
            })
        else:
            features["EOS"] = True

        # Features for the previous word
        if i > 0:
            prev = sent[i - 1]
            prev_w = prev.text.lower()
            features.update({
                "prev_postag": prev.pos_,
                "prev_word.lower()": prev_w,
                "prev_is_title": prev.is_title,
                "prev_is_event_word": prev_w in self.event_words,
                "prev_is_event_modifier": prev_w in self.event_modifiers,
                # New: a feature to capture the "number + relative time word" pattern
                "prev_is_digit": prev.is_digit, 
            })
            if prev.is_digit and w in self.relative_time_words:
                features["is_time_relative_phrase"] = True
        else:
            features["BOS"] = True

        return features