#!/usr/bin/env python3

import math #radici quadrate
import hmac #crittografia
import hashlib #crittografia
import random #generazione primo blocco di testo
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GPT2_VOCAB_SIZE = 50257
BITS_PER_TOKEN = 16


@dataclass
class WatermarkGenerationResult:
    """Risultato della generazione con watermark."""
    text: str
    tokens: List[int]
    bits: List[int]
    total_bits: int
    empirical_entropy: float
    blocks_created: int
    watermark_active_from_bit: int
    block_boundaries: List[int]


@dataclass 
class WatermarkDetectionResult:
    """Risultato della detection del watermark."""
    detected: bool
    confidence: float
    best_score: float
    threshold: float
    best_block_start: int
    best_block_length: int
    details: Dict[str, Any]


class PRF:
    """Pseudo-Random Function basata su HMAC-SHA256."""
    
    def __init__(self, secret_key: int):
        self._key_bytes = secret_key.to_bytes(32, byteorder='big', signed=False)
        self._cache = {}
    
    def evaluate(self, r: str, offset: int) -> float:
        """Calcola F_sk(r, offset) ∈ [0, 1]."""
        key = (r, offset)
        if key not in self._cache:
            input_str = repr(key).encode('utf-8')
            h = hmac.new(self._key_bytes, input_str, hashlib.sha256)
            z = int.from_bytes(h.digest()[:8], byteorder='big', signed=False)
            self._cache[key] = z / (2**64)
        return self._cache[key]
    
    def clear_cache(self):
        """Svuota la cache."""
        self._cache.clear()


class BinaryEncoder:
    """Codifica/decodifica token in rappresentazione binaria."""
    
    def __init__(self, vocab_size: int = GPT2_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.bits_per_token = math.ceil(math.log2(vocab_size))
    
    def encode_token(self, token_id: int) -> List[int]:
        """Converte un token ID in lista di bit."""
        if token_id < 0 or token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} fuori range")
        bits = []
        for i in range(self.bits_per_token - 1, -1, -1):
            bits.append((token_id >> i) & 1)
        return bits
    
    def decode_bits(self, bits: List[int]) -> int:
        """Converte una lista di bit in token ID."""
        if len(bits) != self.bits_per_token:
            raise ValueError(f"Attesi {self.bits_per_token} bit")
        token_id = 0
        for bit in bits:
            token_id = (token_id << 1) | bit
        if token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} >= vocab_size")
        return token_id
    
    def compute_bit_probability(self, token_probs: List[float], previous_bits: List[int], bit_index: int) -> float:
        """Calcola P(bit=1 | bit precedenti, distribuzione token)."""
        prefix_int = 0
        for bit in previous_bits:
            prefix_int = (prefix_int << 1) | bit
        remaining_bits = self.bits_per_token - bit_index
        prob_bit_1 = 0.0
        prob_total = 0.0
        for token_id in range(self.vocab_size):
            token_prefix = token_id >> remaining_bits
            if token_prefix == prefix_int:
                prob_total += token_probs[token_id]
                if (token_id >> (remaining_bits - 1)) & 1:
                    prob_bit_1 += token_probs[token_id]
        return prob_bit_1 / prob_total if prob_total > 1e-10 else 0.5


class ChristWatermarker:
   
    def __init__(self, secret_key: int, security_param: float = 2.0, vocab_size: int = GPT2_VOCAB_SIZE):
        self.secret_key = secret_key
        self.security_param = security_param
        self.vocab_size = vocab_size
        self.prf = PRF(secret_key)
        self.encoder = BinaryEncoder(vocab_size)
        self._entropy_constant = 2.0 / math.log(2)
        self._last_generation: Optional[WatermarkGenerationResult] = None
        logger.info(f"ChristWatermarker: λ={security_param}, vocab={vocab_size}")
    
    def _compute_entropy_threshold(self, ell: int) -> float:
        """Calcola la soglia di entropia per creare un nuovo blocco."""
        return self._entropy_constant * self.security_param * math.sqrt(max(ell, 1))
    
    def generate(self, model, tokenizer, prompt: str, max_new_tokens: int = 100) -> WatermarkGenerationResult:
        import torch
        device = next(model.parameters()).device
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        r = None
        H = 0.0
        ell = 1
        all_bits = []
        generated_tokens = []
        total_entropy = 0.0
        blocks_created = 0
        watermark_active_from_bit = -1
        block_start_bit = 0
        block_boundaries = []
        eos_token_id = tokenizer.eos_token_id
        should_create_block = False
        
        self.prf.clear_cache()
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[0, -1, :]
                token_probs = torch.softmax(logits, dim=-1).cpu().tolist()
            
            current_token_bits = []
            
            for bit_idx in range(self.encoder.bits_per_token):
                prob_bit_1 = self.encoder.compute_bit_probability(token_probs, current_token_bits, bit_idx)
                
                if r is None:
                    # Primo blocco: sampling casuale
                    bit = 1 if random.random() < prob_bit_1 else 0
                else:
                    # Watermark attivo: usa PRF
                    u = self.prf.evaluate(r, ell - 1)
                    bit = 1 if u <= prob_bit_1 else 0
                
                current_token_bits.append(bit)
                all_bits.append(bit)
                
                # Accumula entropia
                prob_chosen = prob_bit_1 if bit == 1 else (1 - prob_bit_1)
                bit_entropy = -math.log2(max(prob_chosen, 1e-10))
                H += bit_entropy
                total_entropy += bit_entropy
                
                # Controlla soglia entropia
                if H >= self._compute_entropy_threshold(ell):
                    should_create_block = True
                
                ell += 1
            
            # Fine token: crea blocco se necessario (allineato ai confini token)
            if should_create_block:
                r = ''.join(map(str, all_bits[block_start_bit:]))
                if watermark_active_from_bit == -1:
                    watermark_active_from_bit = len(all_bits)
                block_boundaries.append(len(all_bits))
                blocks_created += 1
                H = 0.0
                ell = 1
                block_start_bit = len(all_bits)
                should_create_block = False
            
            # Decodifica token
            try:
                token_id = self.encoder.decode_bits(current_token_bits)
            except ValueError:
                token_id = eos_token_id
            
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, torch.tensor([[token_id]], device=device)], dim=1)
            
            if token_id == eos_token_id:
                break
        
        result = WatermarkGenerationResult(
            text=tokenizer.decode(generated_tokens, skip_special_tokens=True),
            tokens=generated_tokens,
            bits=all_bits,
            total_bits=len(all_bits),
            empirical_entropy=total_entropy,
            blocks_created=blocks_created,
            watermark_active_from_bit=watermark_active_from_bit,
            block_boundaries=block_boundaries
        )
        
        self._last_generation = result
        logger.info(f"Generato: {len(generated_tokens)} token, {blocks_created} blocchi")
        
        return result
    
    def detect_with_boundaries(
        self, 
        text: str, 
        tokenizer, 
        boundaries: List[int], 
        original_bits: Optional[List[int]] = None
    ) -> WatermarkDetectionResult:
   
        # Ri-codifica il testo in bit
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_bits = []
        for token_id in tokens:
            if token_id < self.vocab_size:
                all_bits.extend(self.encoder.encode_token(token_id))
            else:
                all_bits.extend([0] * self.encoder.bits_per_token)
        
        L = len(all_bits)
        
        # Se forniti i bit originali, usali SEMPRE (il testo potrebbe essere
        # stato modificato dal character watermark dopo la generazione crypto)
        if original_bits is not None and len(original_bits) > 0:
            all_bits = original_bits
            L = len(all_bits)
        
        if len(boundaries) == 0:
            return WatermarkDetectionResult(
                detected=False, confidence=0.0, best_score=0.0, threshold=0.0,
                best_block_start=-1, best_block_length=-1,
                details={'error': 'Nessun boundary fornito'}
            )
        
        self.prf.clear_cache()
        
        best_margin = -float('inf')
        best_score = 0
        best_threshold = 0
        best_block_idx = -1
        best_n_bits = 0
        detected = False
        
        # Testa ogni blocco
        prev_boundary = 0
        for block_idx, boundary in enumerate(boundaries):
            if boundary > L:
                break
            
            block_start = prev_boundary
            block_end = boundary
            r_block = ''.join(map(str, all_bits[block_start:block_end]))
            
            # Bit da verificare: dal boundary alla fine
            n_bits = L - block_end
            if n_bits < 50:  # Richiedi almeno 50 bit
                prev_boundary = boundary
                continue
            
            # Calcola score
            score = 0.0
            for j in range(block_end, L):
                offset = j - block_end
                x_j = all_bits[j]
                u = self.prf.evaluate(r_block, offset)
                v_j = u if x_j == 1 else (1.0 - u)
                v_j = max(min(v_j, 1.0 - 1e-10), 1e-10) #clamping
                score += -math.log(v_j)
            
            threshold = n_bits + self.security_param * math.sqrt(n_bits)
            margin = score - threshold
            
            if margin > best_margin:
                best_margin = margin
                best_score = score
                best_threshold = threshold
                best_block_idx = block_idx
                best_n_bits = n_bits
                if margin > 0:
                    detected = True
            
            prev_boundary = boundary
        
        # Calcola confidence normalizzata
        if best_margin > 0 and best_n_bits > 0:
            normalized_margin = best_margin / math.sqrt(best_n_bits)
            confidence = min(1.0, max(0.0, normalized_margin / self.security_param))
        else:
            confidence = 0.0
        
        return WatermarkDetectionResult(
            detected=detected,
            confidence=confidence,
            best_score=best_score,
            threshold=best_threshold,
            best_block_start=boundaries[best_block_idx] if best_block_idx >= 0 and best_block_idx < len(boundaries) else -1,
            best_block_length=best_n_bits,
            details={
                'best_block_idx': best_block_idx,
                'margin': best_margin,
                'n_bits_verified': best_n_bits,
                'total_blocks_tested': len(boundaries)
            }
        )
    
    def detect(self, text: str, tokenizer) -> WatermarkDetectionResult:
        logger.warning("detect() è deprecato. Usare detect_with_boundaries() per risultati affidabili.")
        
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)
        
        if n_tokens < 10:
            return WatermarkDetectionResult(
                detected=False, confidence=0.0, best_score=0.0, threshold=0.0,
                best_block_start=-1, best_block_length=-1,
                details={'error': 'Testo troppo corto'}
            )
        
        all_bits = []
        for token_id in tokens:
            if token_id < self.vocab_size:
                all_bits.extend(self.encoder.encode_token(token_id))
            else:
                all_bits.extend([0] * self.encoder.bits_per_token)
        
        L = len(all_bits)
        bpt = self.encoder.bits_per_token
        
        min_block = bpt
        max_block = min(L // 2, bpt * 40)
        min_verify = 128
        max_i = L - min_verify
        step = bpt
        
        n_tests = sum(1 for i in range(min_block, max_i, step)
                      for ell in range(min_block, min(i + 1, max_block + 1), step))
        
        bonferroni = math.log(max(n_tests, 1) + 1) * 0.3
        effective_lambda = self.security_param + bonferroni
        
        self.prf.clear_cache()
        
        best_margin = -float('inf')
        best_score = 0
        best_threshold = 0
        best_i, best_ell = -1, -1
        best_n_bits = 0
        detected = False
        
        for i in range(min_block, max_i, step):
            n_bits = L - i
            
            for ell in range(min_block, min(i + 1, max_block + 1), step):
                block_start = i - ell
                if block_start < 0:
                    continue
                
                r_candidate = ''.join(map(str, all_bits[block_start:i]))
                
                score = 0.0
                for j in range(i, L):
                    offset = j - i
                    x_j = all_bits[j]
                    u = self.prf.evaluate(r_candidate, offset)
                    v_j = u if x_j == 1 else (1.0 - u)
                    v_j = max(min(v_j, 1.0 - 1e-10), 1e-10)
                    score += -math.log(v_j)
                
                threshold = n_bits + effective_lambda * math.sqrt(n_bits)
                margin = score - threshold
                
                if margin > best_margin:
                    best_margin = margin
                    best_score = score
                    best_threshold = threshold
                    best_i = i
                    best_ell = ell
                    best_n_bits = n_bits
                    if margin > 0:
                        detected = True
        
        if best_margin > 0:
            normalized_margin = best_margin / math.sqrt(max(best_n_bits, 1))
            confidence = min(1.0, max(0.0, normalized_margin / effective_lambda))
        else:
            confidence = 0.0
        
        return WatermarkDetectionResult(
            detected=detected,
            confidence=confidence,
            best_score=best_score,
            threshold=best_threshold,
            best_block_start=best_i - best_ell if best_i >= 0 else -1,
            best_block_length=best_ell,
            details={
                'total_bits': L,
                'n_tests': n_tests,
                'effective_lambda': effective_lambda,
                'margin': best_margin,
                'best_i': best_i,
                'n_bits_verified': best_n_bits,
                'warning': 'Detection blind - risultati potrebbero essere inaffidabili'
            }
        )


def create_watermarker(secret_key: int, security_param: float = 2.0, vocab_size: int = GPT2_VOCAB_SIZE) -> ChristWatermarker:
    """Factory function per creare un ChristWatermarker."""
    return ChristWatermarker(secret_key=secret_key, security_param=security_param, vocab_size=vocab_size)
