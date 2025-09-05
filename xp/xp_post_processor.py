import re
from collections import defaultdict

class EnhancedEventExtractor:
    def __init__(self):
        # Multi-word event patterns
        self.event_patterns = [
            r'\b(final|initial|first|last|next|upcoming)\s+(review|meeting|presentation|call|demo|session)\b',
            r'\b(team|company|client|project)\s+(meeting|party|dinner|lunch|review|session)\b', 
            r'\b(code|design|performance|quarterly|annual)\s+(review|meeting|session)\b',
            r'\b(kick-off|follow-up|stand-up)\s+(meeting|session|call)?\b',
            r'\b(brainstorm|planning|training)\s+(session|meeting|workshop)\b',
            r'\b(quick|brief|short)\s+(call|meeting|chat|sync)\b',
        ]
        
        # Event expansion words (should be included in events)
        self.expansion_words = {
            'final', 'initial', 'first', 'last', 'next', 'upcoming', 'scheduled', 'planned',
            'team', 'company', 'client', 'project', 'quarterly', 'annual', 'monthly', 'weekly',
            'quick', 'brief', 'short', 'long', 'important', 'urgent', 'mandatory', 'optional',
            'code', 'design', 'performance', 'marketing', 'sales', 'hr', 'product'
        }

    def bio_to_structured_enhanced(self, tokens, bio_tags):
        """Enhanced BIO tag processing with multi-word event reconstruction"""
        entities = defaultdict(list)
        current_entity = None
        current_tokens = []
        
        for token, tag in zip(tokens, bio_tags):
            if tag.startswith('B-'):
                # Save previous entity if exists
                if current_entity and current_tokens:
                    entity_text = ' '.join(current_tokens).strip()
                    entities[current_entity].append(entity_text)
                
                # Start new entity
                current_entity = tag[2:]  # Remove 'B-'
                current_tokens = [token]
                
            elif tag.startswith('I-'):
                if current_entity == tag[2:]:  # Same entity type
                    current_tokens.append(token)
                else:
                    # Entity type changed, save previous and start new
                    if current_entity and current_tokens:
                        entity_text = ' '.join(current_tokens).strip()
                        entities[current_entity].append(entity_text)
                    
                    current_entity = tag[2:]
                    current_tokens = [token]
                    
            else:  # 'O' tag
                # Save current entity if exists
                if current_entity and current_tokens:
                    entity_text = ' '.join(current_tokens).strip()
                    entities[current_entity].append(entity_text)
                
                current_entity = None
                current_tokens = []
        
        # Don't forget the last entity
        if current_entity and current_tokens:
            entity_text = ' '.join(current_tokens).strip()
            entities[current_entity].append(entity_text)
        
        # Post-process to expand events
        if 'EVENT' in entities:
            entities['EVENT'] = self.expand_events(entities['EVENT'], tokens, bio_tags)
            
        return dict(entities)

    def expand_events(self, events, original_tokens, original_tags):
        """Expand single-word events to include modifying words"""
        expanded_events = []
        
        for event in events:
            expanded_event = self.expand_single_event(event, original_tokens, original_tags)
            expanded_events.append(expanded_event)
            
        return expanded_events

    def expand_single_event(self, event, tokens, tags):
        """Expand a single event by looking at context"""
        event_words = event.split()
        
        # Find the position of this event in the original tokens
        event_start = -1
        for i in range(len(tokens) - len(event_words) + 1):
            if tokens[i:i+len(event_words)] == event_words:
                event_start = i
                break
        
        if event_start == -1:
            return event  # Couldn't find event in original tokens
            
        event_end = event_start + len(event_words)
        
        # Look for expansion words before the event
        expanded_start = event_start
        for i in range(event_start - 1, -1, -1):
            word = tokens[i].lower()
            tag = tags[i]
            
            # Stop if we hit another entity or punctuation
            if not tag.startswith('O') or word in ['.', ',', '!', '?']:
                break
                
            # Include expansion words
            if word in self.expansion_words:
                expanded_start = i
            else:
                break  # Stop at first non-expansion word
        
        # Look for expansion words after the event (less common but possible)
        expanded_end = event_end
        for i in range(event_end, len(tokens)):
            word = tokens[i].lower()
            tag = tags[i]
            
            # Stop if we hit another entity or punctuation
            if not tag.startswith('O') or word in ['.', ',', '!', '?']:
                break
                
            # Include certain expansion words (be more conservative here)
            if word in ['session', 'meeting', 'call', 'review'] and i == event_end:
                expanded_end = i + 1
            else:
                break
                
        # Return the expanded event
        expanded_tokens = tokens[expanded_start:expanded_end]
        return ' '.join(expanded_tokens).strip()

    def apply_pattern_matching(self, text, structured_data):
        """Apply regex patterns to catch missed multi-word events"""
        text_lower = text.lower()
        
        # Check for pattern matches
        for pattern in self.event_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                matched_text = match.group().strip()
                
                # Only add if not already captured
                if 'EVENT' not in structured_data:
                    structured_data['EVENT'] = []
                
                # Check if this event is already captured (or a subset of it)
                already_captured = False
                for existing_event in structured_data['EVENT']:
                    if matched_text in existing_event.lower() or existing_event.lower() in matched_text:
                        already_captured = True
                        break
                
                if not already_captured:
                    structured_data['EVENT'].append(matched_text)
        
        return structured_data

    def post_process_events(self, structured_data, original_text):
        """Complete post-processing pipeline"""
        # Apply pattern matching first
        structured_data = self.apply_pattern_matching(original_text, structured_data)
        
        # Clean up events
        if 'EVENT' in structured_data and structured_data['EVENT']:
            cleaned_events = []
            for event in structured_data['EVENT']:
                # Remove empty events
                if event and event.strip():
                    cleaned_event = event.strip()
                    # Remove duplicates
                    if cleaned_event not in cleaned_events:
                        cleaned_events.append(cleaned_event)
            
            structured_data['EVENT'] = cleaned_events if cleaned_events else None
        
        return structured_data
