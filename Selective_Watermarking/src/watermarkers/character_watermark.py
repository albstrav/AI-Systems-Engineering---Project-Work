#!/usr/bin/env python3

import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# MAPPATURA LOOKALIKES
# Caratteri ASCII e i loro equivalenti visivi in altri alfabeti Unicode
LOOKALIKES: Dict[str, str] = {
    # Minuscole - Cirillico
    'a': 'а',  # U+0061 → U+0430
    'c': 'с',  # U+0063 → U+0441
    'e': 'е',  # U+0065 → U+0435
    'o': 'о',  # U+006F → U+043E
    'p': 'р',  # U+0070 → U+0440
    's': 'ѕ',  # U+0073 → U+0455
    'x': 'х',  # U+0078 → U+0445
    'y': 'у',  # U+0079 → U+0443
    'i': 'і',  # U+0069 → U+0456
    
    # Maiuscole - Cirillico
    'A': 'А',  # U+0041 → U+0410
    'B': 'В',  # U+0042 → U+0412
    'C': 'С',  # U+0043 → U+0421
    'E': 'Е',  # U+0045 → U+0415
    'H': 'Н',  # U+0048 → U+041D
    'K': 'К',  # U+004B → U+041A
    'M': 'М',  # U+004D → U+041C
    'O': 'О',  # U+004F → U+041E
    'P': 'Р',  # U+0050 → U+0420
    'T': 'Т',  # U+0054 → U+0422
    'X': 'Х',  # U+0058 → U+0425
}

# Lookup inverso
REVERSE_LOOKALIKES: Dict[str, str] = {v: k for k, v in LOOKALIKES.items()}
LOOKALIKE_SET = set(LOOKALIKES.values())


def is_lookalike(char: str) -> bool:
    """Verifica se un carattere è un lookalike Unicode."""
    return char in LOOKALIKE_SET


def get_original(char: str) -> str:
    """Ottiene il carattere ASCII originale da un lookalike."""
    return REVERSE_LOOKALIKES.get(char, char)


def normalize_text(text: str) -> str:
    """Normalizza il testo convertendo tutti i lookalikes in ASCII."""
    return ''.join(get_original(c) for c in text)


# DATACLASSES
@dataclass
class CharacterApplyResult:
    """Risultato dell'applicazione del character watermark."""
    text: str
    original_text: str
    total_chars: int
    eligible_chars: int
    substituted_chars: int


@dataclass
class CharacterDetectResult:
    """Risultato della detection del character watermark."""
    detected: bool
    confidence: float
    match_ratio: float
    correct_matches: int
    total_eligible: int


# CLASSE CHARACTER WATERMARKER
class CharacterWatermarker:
    def __init__(
        self,
        secret_key: int,
        substitution_rate: float = 0.25
    ):
        self.secret_key = secret_key
        self.substitution_rate = substitution_rate
        
        logger.info(f"CharacterWatermarker: key={secret_key}, rate={substitution_rate}")
    
    def _get_rng(self) -> random.Random:
        """Crea un nuovo RNG con seed dalla chiave."""
        rng = random.Random()
        rng.seed(self.secret_key)
        return rng
    
    def apply(self, text: str) -> CharacterApplyResult:
        rng = self._get_rng()
        result_chars = []
        
        total_chars = len(text)
        eligible_chars = 0
        substituted_chars = 0
        
        for char in text:
            if char in LOOKALIKES:
                eligible_chars += 1
                if rng.random() < self.substitution_rate:
                    result_chars.append(LOOKALIKES[char])
                    substituted_chars += 1
                else:
                    result_chars.append(char)
            else:
                result_chars.append(char)
                # Consuma random per sincronizzazione
                if char.isalpha():
                    rng.random()
        
        return CharacterApplyResult(
            text=''.join(result_chars),
            original_text=text,
            total_chars=total_chars,
            eligible_chars=eligible_chars,
            substituted_chars=substituted_chars
        )
    
    def detect(self, text: str) -> CharacterDetectResult:
        rng = self._get_rng()
        
        total_eligible = 0
        correct_matches = 0
        
        for char in text:
            original_char = get_original(char)
            
            if original_char in LOOKALIKES:
                total_eligible += 1
                expected_sub = rng.random() < self.substitution_rate
                actual_sub = is_lookalike(char)
                
                if expected_sub == actual_sub:
                    correct_matches += 1
            else:
                if original_char.isalpha():
                    rng.random()
        
        if total_eligible == 0:
            return CharacterDetectResult(
                detected=False,
                confidence=0.0,
                match_ratio=0.0,
                correct_matches=0,
                total_eligible=0
            )
        
        match_ratio = correct_matches / total_eligible
        confidence = max(0.0, min(1.0, (match_ratio - 0.5) * 2))
        detected = match_ratio > 0.65 and total_eligible >= 5
        
        return CharacterDetectResult(
            detected=detected,
            confidence=confidence,
            match_ratio=match_ratio,
            correct_matches=correct_matches,
            total_eligible=total_eligible
        )


# TEST
if __name__ == "__main__":
    print("=" * 70)
    print("TEST: Character Watermarking")
    print("=" * 70)
    
    # Mostra lookalikes
    print("\n1. ESEMPI LOOKALIKES:")
    print(f"   {'ASCII':<6} {'Unicode':<6} {'Code ASCII':<12} {'Code Unicode'}")
    for ascii_c in ['a', 'e', 'o', 'c', 'p', 'A', 'E', 'O']:
        if ascii_c in LOOKALIKES:
            uni_c = LOOKALIKES[ascii_c]
            print(f"   '{ascii_c}'    '{uni_c}'    U+{ord(ascii_c):04X}       U+{ord(uni_c):04X}")
    
    wm = CharacterWatermarker(secret_key=12345, substitution_rate=0.30)
    
    test_text = "The quick brown fox jumps over the lazy dog. AI is transforming science."
    
    print(f"\n2. TESTO ORIGINALE:\n   {test_text}")
    
    result = wm.apply(test_text)
    print(f"\n3. TESTO WATERMARKED:\n   {result.text}")
    print(f"   (Visivamente identico, ma {result.substituted_chars} caratteri sostituiti)")
    
    # Detection
    detection = wm.detect(result.text)
    print(f"\n4. DETECTION (chiave corretta):")
    print(f"   Rilevato: {detection.detected}")
    print(f"   Match ratio: {detection.match_ratio:.2%}")
    print(f"   Confidence: {detection.confidence:.2%}")
    
    # Chiave sbagliata
    wm_wrong = CharacterWatermarker(secret_key=99999)
    detection_wrong = wm_wrong.detect(result.text)
    print(f"\n5. DETECTION (chiave sbagliata):")
    print(f"   Rilevato: {detection_wrong.detected}")
    print(f"   Match ratio: {detection_wrong.match_ratio:.2%}")
    
    # Normalizzazione
    normalized = normalize_text(result.text)
    print(f"\n6. DOPO NORMALIZZAZIONE:")
    print(f"   {normalized}")
    print(f"   Uguale all'originale: {normalized == test_text}")
    
    print("\n" + "=" * 70)
