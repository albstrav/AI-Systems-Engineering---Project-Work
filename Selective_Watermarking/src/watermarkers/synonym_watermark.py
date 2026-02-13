#!/usr/bin/env python3

import hashlib
import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# GRUPPI DI SINONIMI CURATI
# I sinonimi sono curati manualmente per garantire intercambiabilità.
# Criteri: stessa parte del discorso, significato simile, registro simile.

CURATED_SYNONYM_GROUPS: Dict[str, List[str]] = {
    # === AGGETTIVI ===
    'big': ['big', 'large', 'huge', 'enormous', 'vast', 'massive'],
    'small': ['small', 'little', 'tiny', 'minute', 'compact'],
    'good': ['good', 'excellent', 'great', 'fine', 'superb'],
    'bad': ['bad', 'poor', 'terrible', 'awful', 'dreadful'],
    'important': ['important', 'significant', 'crucial', 'vital', 'essential'],
    'difficult': ['difficult', 'hard', 'challenging', 'tough', 'demanding'],
    'easy': ['easy', 'simple', 'straightforward', 'effortless'],
    'fast': ['fast', 'quick', 'rapid', 'swift', 'speedy'],
    'slow': ['slow', 'gradual', 'unhurried', 'leisurely'],
    'new': ['new', 'recent', 'fresh', 'modern', 'novel'],
    'old': ['old', 'ancient', 'aged', 'elderly', 'vintage'],
    'happy': ['happy', 'joyful', 'cheerful', 'delighted', 'pleased'],
    'sad': ['sad', 'unhappy', 'sorrowful', 'melancholy', 'dejected'],
    'different': ['different', 'distinct', 'diverse', 'various', 'unique'],
    'similar': ['similar', 'alike', 'comparable', 'analogous'],
    'clear': ['clear', 'obvious', 'evident', 'apparent', 'plain'],
    
    # === VERBI ===
    'say': ['say', 'state', 'declare', 'mention', 'express'],
    'tell': ['tell', 'inform', 'notify', 'advise'],
    'ask': ['ask', 'inquire', 'question', 'query'],
    'answer': ['answer', 'respond', 'reply', 'react'],
    'explain': ['explain', 'clarify', 'describe', 'illustrate'],
    'think': ['think', 'consider', 'believe', 'suppose', 'assume'],
    'know': ['know', 'understand', 'recognize', 'realize'],
    'see': ['see', 'observe', 'notice', 'perceive', 'view'],
    'find': ['find', 'discover', 'locate', 'identify', 'detect'],
    'make': ['make', 'create', 'produce', 'construct', 'build'],
    'give': ['give', 'provide', 'offer', 'supply', 'grant'],
    'take': ['take', 'grab', 'seize', 'obtain', 'acquire'],
    'use': ['use', 'utilize', 'employ', 'apply'],
    'change': ['change', 'modify', 'alter', 'adjust', 'transform'],
    'improve': ['improve', 'enhance', 'upgrade', 'refine', 'boost'],
    'reduce': ['reduce', 'decrease', 'diminish', 'lower', 'cut'],
    'increase': ['increase', 'grow', 'expand', 'rise', 'escalate'],
    'help': ['help', 'assist', 'aid', 'support'],
    'need': ['need', 'require', 'demand'],
    'want': ['want', 'desire', 'wish', 'seek'],
    'try': ['try', 'attempt', 'endeavor', 'strive'],
    'start': ['start', 'begin', 'commence', 'initiate'],
    'stop': ['stop', 'cease', 'halt', 'end', 'terminate'],
    'show': ['show', 'demonstrate', 'display', 'exhibit', 'reveal'],
    
    # === SOSTANTIVI ===
    'problem': ['problem', 'issue', 'challenge', 'difficulty', 'obstacle'],
    'solution': ['solution', 'answer', 'resolution', 'remedy'],
    'result': ['result', 'outcome', 'consequence', 'effect'],
    'reason': ['reason', 'cause', 'motive', 'rationale'],
    'way': ['way', 'method', 'approach', 'means', 'technique'],
    'part': ['part', 'portion', 'section', 'segment', 'component'],
    'place': ['place', 'location', 'site', 'spot', 'area'],
    'work': ['work', 'job', 'task', 'assignment', 'project'],
    'group': ['group', 'team', 'collection', 'assembly'],
    
    # === AVVERBI ===
    'very': ['very', 'extremely', 'highly', 'greatly', 'remarkably'],
    'quickly': ['quickly', 'rapidly', 'swiftly', 'speedily', 'promptly'],
    'slowly': ['slowly', 'gradually', 'steadily', 'leisurely'],
    'often': ['often', 'frequently', 'regularly', 'commonly'],
}

# Lookup inverso: sinonimo -> parola base
SYNONYM_TO_BASE: Dict[str, str] = {}
for base, synonyms in CURATED_SYNONYM_GROUPS.items():
    for syn in synonyms:
        SYNONYM_TO_BASE[syn.lower()] = base


# DATACLASSES
@dataclass
class SynonymApplyResult:
    """Risultato dell'applicazione del synonym watermark."""
    text: str
    original_text: str
    substitutions_made: int
    words_with_synonyms: int
    total_words: int
    details: List[Dict[str, str]]


@dataclass
class SynonymDetectResult:
    """Risultato della detection del synonym watermark."""
    detected: bool
    confidence: float
    match_ratio: float
    matches: int
    total_checkable: int
    details: List[Dict]


# CLASSE SYNONYM WATERMARKER
class SynonymWatermarker:

    def __init__(
        self,
        secret_key: int,
        context_window: int = 2
    ):
        self.secret_key = secret_key
        self.context_window = context_window
        self.synonym_groups = CURATED_SYNONYM_GROUPS
        self.word_to_base = SYNONYM_TO_BASE
        
        logger.info(f"SynonymWatermarker: key={secret_key}, window={context_window}")
    
    def _tokenize(self, text: str) -> List[Tuple[str, int, int]]:
        return [(m.group(), m.start(), m.end()) 
                for m in re.finditer(r'\b\w+\b', text)]
    
    def _get_context(self, words: List[str], index: int) -> str:
        start = max(0, index - self.context_window)
        end = min(len(words), index + self.context_window + 1)
        context_words = words[start:index] + words[index+1:end]
        return " ".join(context_words).lower()
    
    def _compute_hash(self, context: str, base_word: str) -> int:
        input_str = f"{self.secret_key}:{context}:{base_word.lower()}"
        hash_bytes = hashlib.sha256(input_str.encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder='big')
    
    def _select_synonym(self, base_word: str, context: str) -> str:
        synonyms = sorted(self.synonym_groups.get(base_word, [base_word]))
        hash_val = self._compute_hash(context, base_word)
        return synonyms[hash_val % len(synonyms)]
    
    def _preserve_case(self, original: str, replacement: str) -> str:
        """Preserva la capitalizzazione."""
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement.capitalize()
        return replacement.lower()
    
    def apply(self, text: str) -> SynonymApplyResult:
        tokens = self._tokenize(text)
        words = [t[0] for t in tokens]
        
        # Prima passa: converti tutte le parole nelle loro forme base
        base_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.word_to_base:
                base_words.append(self.word_to_base[word_lower])
            else:
                base_words.append(word.lower())
        
        result_text = text
        offset = 0
        substitutions = []
        words_with_synonyms = 0
        substitutions_made = 0
        
        for i, (word, start, end) in enumerate(tokens):
            word_lower = word.lower()
            
            if word_lower in self.word_to_base:
                words_with_synonyms += 1
                base_word = self.word_to_base[word_lower]
                
                # Usa le parole base per il contesto
                ctx_start = max(0, i - self.context_window)
                ctx_end = min(len(base_words), i + self.context_window + 1)
                context_words = base_words[ctx_start:i] + base_words[i+1:ctx_end]
                context = " ".join(context_words)
                
                selected = self._select_synonym(base_word, context)
                selected_cased = self._preserve_case(word, selected)
                
                if selected_cased != word:
                    substitutions_made += 1
                    substitutions.append({
                        'original': word,
                        'replacement': selected_cased,
                        'position': start,
                        'base_word': base_word,
                        'context': context
                    })
                
                # Applica sostituzione
                adj_start = start + offset
                adj_end = end + offset
                result_text = result_text[:adj_start] + selected_cased + result_text[adj_end:]
                offset += len(selected_cased) - len(word)
        
        return SynonymApplyResult(
            text=result_text,
            original_text=text,
            substitutions_made=substitutions_made,
            words_with_synonyms=words_with_synonyms,
            total_words=len(tokens),
            details=substitutions
        )
    
    def detect(self, text: str) -> SynonymDetectResult:
        tokens = self._tokenize(text)
        words = [t[0] for t in tokens]
        
        # Prima passa: converti tutte le parole nelle loro forme base
        # per avere un contesto consistente
        base_words = []
        for word in words:
            word_lower = word.lower()
            if word_lower in self.word_to_base:
                base_words.append(self.word_to_base[word_lower])
            else:
                base_words.append(word.lower())
        
        matches = 0
        total_checkable = 0
        details = []
        
        for i, (word, _, _) in enumerate(tokens):
            word_lower = word.lower()
            
            if word_lower in self.word_to_base:
                total_checkable += 1
                base_word = self.word_to_base[word_lower]
                
                # Usa le parole base per il contesto
                start = max(0, i - self.context_window)
                end = min(len(base_words), i + self.context_window + 1)
                context_words = base_words[start:i] + base_words[i+1:end]
                context = " ".join(context_words)
                
                # Calcola sinonimo atteso
                expected = self._select_synonym(base_word, context)
                
                is_match = word_lower == expected.lower()
                if is_match:
                    matches += 1
                
                details.append({
                    'word': word,
                    'base': base_word,
                    'expected': expected,
                    'match': is_match,
                    'context': context
                })
        
        if total_checkable == 0:
            return SynonymDetectResult(
                detected=False,
                confidence=0.0,
                match_ratio=0.0,
                matches=0,
                total_checkable=0,
                details=[]
            )
        
        match_ratio = matches / total_checkable
        
        # Confidence considera anche il numero di campioni
        sample_factor = min(1.0, total_checkable / 10)
        confidence = match_ratio * sample_factor
        
        # Soglia: random sarebbe ~1/|sinonimi| ≈ 0.2, usiamo 0.5
        detected = match_ratio > 0.5 and total_checkable >= 3
        
        return SynonymDetectResult(
            detected=detected,
            confidence=confidence,
            match_ratio=match_ratio,
            matches=matches,
            total_checkable=total_checkable,
            details=details
        )


# TEST
if __name__ == "__main__":
    print("=" * 70)
    print("TEST: Synonym Watermarking")
    print("=" * 70)
    
    wm = SynonymWatermarker(secret_key=12345)
    
    test_text = """
    The big problem is very important. We need to find a good solution quickly.
    Many people think it is difficult, but I believe we can make significant
    improvements if we change our approach. The team needs to help each other.
    """
    
    print(f"\n1. TESTO ORIGINALE:\n{test_text.strip()}")
    
    # Apply
    result = wm.apply(test_text)
    print(f"\n2. TESTO WATERMARKED:\n{result.text.strip()}")
    print(f"\n   Sostituzioni: {result.substitutions_made}/{result.words_with_synonyms}")
    
    # Detect con chiave corretta
    detection = wm.detect(result.text)
    print(f"\n3. DETECTION (chiave corretta):")
    print(f"   Rilevato: {detection.detected}")
    print(f"   Match ratio: {detection.match_ratio:.2%}")
    print(f"   Confidence: {detection.confidence:.2%}")
    
    # Detect con chiave sbagliata
    wm_wrong = SynonymWatermarker(secret_key=99999)
    detection_wrong = wm_wrong.detect(result.text)
    print(f"\n4. DETECTION (chiave sbagliata):")
    print(f"   Rilevato: {detection_wrong.detected}")
    print(f"   Match ratio: {detection_wrong.match_ratio:.2%}")
    
    print("\n" + "=" * 70)
