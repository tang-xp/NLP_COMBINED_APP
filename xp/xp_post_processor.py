# xp/xp_post_processor.py

import re
from collections import defaultdict


class EnhancedEventExtractor:
    def __init__(self):
        self.event_patterns = [
            r"\b(final|initial|first|last|next|upcoming)\s+(review|meeting|presentation|call|demo|session)\b",
            r"\b(team|company|client|project)\s+(meeting|party|dinner|lunch|review|session)\b",
            r"\b(code|design|performance|quarterly|annual)\s+(review|meeting|session)\b",
            r"\b(kick-off|follow-up|stand-up)\s+(meeting|session|call)?\b",
            r"\b(brainstorm|planning|training)\s+(session|meeting|workshop)\b",
            r"\b(quick|brief|short)\s+(call|meeting|chat|sync)\b",
        ]
        self.expansion_words = {
            "final", "initial", "first", "last", "next", "upcoming", "scheduled", "planned",
            "team", "company", "client", "project", "quarterly", "annual", "monthly", "weekly",
            "quick", "brief", "short", "long", "important", "urgent", "mandatory", "optional",
            "code", "design", "performance", "marketing", "sales", "hr", "product",
        }

    def bio_to_structured_enhanced(self, tokens, bio_labels, return_list=False):
        """
        Converts BIO labels into a structured list or dictionary of entities.
        """
        structured_list = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, bio_labels)):
            if label.startswith('B-'):
                # Start of a new entity
                if current_entity:
                    structured_list.append(current_entity)
                current_entity = {
                    'type': label[2:],
                    'text': token,
                }
            elif label.startswith('I-'):
                # Continuation of an entity
                if current_entity and label[2:] == current_entity['type']:
                    current_entity['text'] += f" {token}"
                else:
                    # Mismatched I- tag, treat as new entity or ignore
                    if current_entity:
                        structured_list.append(current_entity)
                    current_entity = {
                        'type': label[2:],
                        'text': token,
                    }
            else: # 'O' tag
                if current_entity:
                    structured_list.append(current_entity)
                current_entity = None
                
        # Append the last entity if it exists
        if current_entity:
            structured_list.append(current_entity)
        
        if return_list:
            return structured_list
        else:
            # Convert the list to your original dictionary format for backward compatibility
            structured_dict = {}
            for entity in structured_list:
                if entity['type'] not in structured_dict:
                    structured_dict[entity['type']] = []
                structured_dict[entity['type']].append(entity['text'])
            return structured_dict

    def expand_events(self, events, tokens, tags):
        return [self.expand_single_event(e, tokens, tags) for e in events]

    def expand_single_event(self, event, tokens, tags):
        event_words = event.split()
        event_start = next(
            (i for i in range(len(tokens) - len(event_words) + 1)
             if tokens[i:i + len(event_words)] == event_words),
            -1,
        )
        if event_start == -1:
            return event

        event_end = event_start + len(event_words)
        expanded_start, expanded_end = event_start, event_end

        # Expand backward
        for i in range(event_start - 1, -1, -1):
            word, tag = tokens[i].lower(), tags[i]
            if not tag.startswith("O") or word in [".", ",", "!", "?"]:
                break
            if word in self.expansion_words:
                expanded_start = i
            else:
                break

        # Expand forward
        for i in range(event_end, len(tokens)):
            word, tag = tokens[i].lower(), tags[i]
            if not tag.startswith("O") or word in [".", ",", "!", "?"]:
                break
            if word in ["session", "meeting", "call", "review"] and i == event_end:
                expanded_end = i + 1
            else:
                break

        return " ".join(tokens[expanded_start:expanded_end]).strip()

    def apply_pattern_matching(self, text, structured_data):
        text_lower = text.lower()
        for pattern in self.event_patterns:
            for match in re.finditer(pattern, text_lower):
                matched = match.group().strip()
                if "EVENT" not in structured_data:
                    structured_data["EVENT"] = []
                if not any(matched in e.lower() or e.lower() in matched
                           for e in structured_data["EVENT"]):
                    structured_data["EVENT"].append(matched)
        return structured_data

    def post_process_events(self, structured_data, original_text):
        structured_data = self.apply_pattern_matching(original_text, structured_data)
        if "EVENT" in structured_data and structured_data["EVENT"]:
            cleaned = []
            for e in structured_data["EVENT"]:
                e = e.strip()
                if e and e not in cleaned:
                    cleaned.append(e)
            structured_data["EVENT"] = cleaned if cleaned else None
        return structured_data