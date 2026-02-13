#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging

from .crypto_watermark_christ import ChristWatermarker, GPT2_VOCAB_SIZE
from .synonym_watermark import SynonymWatermarker
from .character_watermark import CharacterWatermarker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CONFIGURAZIONI
class WatermarkTechnique(Enum):
    """Tecniche di watermarking disponibili."""
    CRYPTO = "crypto"
    SYNONYM = "synonym"
    CHARACTER = "character"


@dataclass
class TaskConfig:
    key: int
    techniques: List[WatermarkTechnique]
    crypto_security_param: float = 2.0
    synonym_context_window: int = 2
    character_substitution_rate: float = 0.25
    detection_weights: Dict[WatermarkTechnique, float] = field(default_factory=lambda: {
        WatermarkTechnique.CRYPTO: 0.5,
        WatermarkTechnique.SYNONYM: 0.3,
        WatermarkTechnique.CHARACTER: 0.2
    })


# Configurazioni default per i task
DEFAULT_TASK_CONFIGS: Dict[str, TaskConfig] = {
    'qa': TaskConfig(
        key=314159265,  #pi greco
        techniques=[WatermarkTechnique.CRYPTO, WatermarkTechnique.CHARACTER],
        crypto_security_param=2.0,
        character_substitution_rate=0.25,
        detection_weights={
            WatermarkTechnique.CRYPTO: 0.6,
            WatermarkTechnique.CHARACTER: 0.4,
        }
    ),
    'summary': TaskConfig(
        key=271828182,  #Eulero)
        techniques=[WatermarkTechnique.CRYPTO, WatermarkTechnique.SYNONYM],
        crypto_security_param=2.0,
        detection_weights={
            WatermarkTechnique.CRYPTO: 0.6,
            WatermarkTechnique.SYNONYM: 0.4,
        }
    ),
    'news': TaskConfig(
        key=161803398,  #sezione aurea)
        techniques=[
            WatermarkTechnique.CRYPTO,
            WatermarkTechnique.SYNONYM,
            WatermarkTechnique.CHARACTER
        ],
        crypto_security_param=2.0,
        character_substitution_rate=0.25,
        detection_weights={
            WatermarkTechnique.CRYPTO: 0.4,
            WatermarkTechnique.SYNONYM: 0.35,
            WatermarkTechnique.CHARACTER: 0.25
        }
    )
}


# DATACLASSES RISULTATI
@dataclass
class MultiLayerApplyResult:
    """Risultato dell'applicazione multi-layer."""
    text: str
    task: str
    techniques_applied: List[str]
    stats: Dict[str, Any]
    #metadata per la detection crypto
    crypto_boundaries: Optional[List[int]] = None
    crypto_bits: Optional[List[int]] = None


@dataclass
class MultiLayerDetectResult:
    """Risultato della detection per un singolo task."""
    detected: bool
    confidence: float
    weighted_score: float
    threshold: float
    technique_results: Dict[str, Dict[str, Any]]


@dataclass
class FullDetectionResult:
    """Risultato della detection su tutti i task."""
    task_results: Dict[str, MultiLayerDetectResult]
    best_match: Optional[str]
    best_confidence: float
    is_watermarked: bool


# MULTI-LAYER WATERMARKER
class MultiLayerWatermarker:
    def __init__(
        self,
        task_configs: Optional[Dict[str, TaskConfig]] = None,
        vocab_size: int = GPT2_VOCAB_SIZE
    ):
        self.task_configs = task_configs or DEFAULT_TASK_CONFIGS
        self.vocab_size = vocab_size
        
        # Cache per i watermarker (creati on-demand)
        self._crypto_cache: Dict[int, ChristWatermarker] = {}
        self._synonym_cache: Dict[int, SynonymWatermarker] = {}
        self._character_cache: Dict[int, CharacterWatermarker] = {}
        
        logger.info(f"MultiLayerWatermarker inizializzato con {len(self.task_configs)} task")
    
    def _get_crypto_watermarker(self, config: TaskConfig) -> ChristWatermarker:
        """Ottiene o crea un ChristWatermarker per la configurazione."""
        if config.key not in self._crypto_cache:
            self._crypto_cache[config.key] = ChristWatermarker(
                secret_key=config.key,
                security_param=config.crypto_security_param,
                vocab_size=self.vocab_size
            )
        return self._crypto_cache[config.key]
    
    def _get_synonym_watermarker(self, config: TaskConfig) -> SynonymWatermarker:
        """Ottiene o crea un SynonymWatermarker per la configurazione."""
        if config.key not in self._synonym_cache:
            self._synonym_cache[config.key] = SynonymWatermarker(
                secret_key=config.key,
                context_window=config.synonym_context_window
            )
        return self._synonym_cache[config.key]
    
    def _get_character_watermarker(self, config: TaskConfig) -> CharacterWatermarker:
        """Ottiene o crea un CharacterWatermarker per la configurazione."""
        if config.key not in self._character_cache:
            self._character_cache[config.key] = CharacterWatermarker(
                secret_key=config.key,
                substitution_rate=config.character_substitution_rate
            )
        return self._character_cache[config.key]
    
    def get_available_tasks(self) -> List[str]:
        """Restituisce la lista dei task disponibili."""
        return list(self.task_configs.keys())
    
    def generate(
        self,
        model,
        tokenizer,
        prompt: str,
        task: str,
        max_new_tokens: int = 100
    ) -> MultiLayerApplyResult:
        if task not in self.task_configs:
            raise ValueError(f"Task '{task}' non trovato. Disponibili: {self.get_available_tasks()}")
        
        config = self.task_configs[task]
        stats = {}
        techniques_applied = []
        crypto_boundaries = None
        crypto_bits = None
        
        # Step 1: Crypto watermarking (durante generazione)
        if WatermarkTechnique.CRYPTO in config.techniques:
            crypto_wm = self._get_crypto_watermarker(config)
            crypto_result = crypto_wm.generate(model, tokenizer, prompt, max_new_tokens)
            text = crypto_result.text
            techniques_applied.append('crypto')
            stats['crypto'] = {
                'total_bits': crypto_result.total_bits,
                'blocks_created': crypto_result.blocks_created,
                'empirical_entropy': crypto_result.empirical_entropy,
                'watermark_active_from_bit': crypto_result.watermark_active_from_bit
            }
            # SALVA BOUNDARIES E BITS per la detection
            crypto_boundaries = crypto_result.block_boundaries
            crypto_bits = crypto_result.bits
        else:
            # Generazione normale senza watermark crypto
            import torch
            device = next(model.parameters()).device
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Step 2: Synonym watermarking (post-hoc)
        if WatermarkTechnique.SYNONYM in config.techniques:
            synonym_wm = self._get_synonym_watermarker(config)
            synonym_result = synonym_wm.apply(text)
            text = synonym_result.text
            techniques_applied.append('synonym')
            stats['synonym'] = {
                'substitutions_made': synonym_result.substitutions_made,
                'words_with_synonyms': synonym_result.words_with_synonyms
            }
        
        # Step 3: Character watermarking (post-hoc)
        if WatermarkTechnique.CHARACTER in config.techniques:
            char_wm = self._get_character_watermarker(config)
            char_result = char_wm.apply(text)
            text = char_result.text
            techniques_applied.append('character')
            stats['character'] = {
                'substituted_chars': char_result.substituted_chars,
                'eligible_chars': char_result.eligible_chars
            }
        
        return MultiLayerApplyResult(
            text=text,
            task=task,
            techniques_applied=techniques_applied,
            stats=stats,
            crypto_boundaries=crypto_boundaries,
            crypto_bits=crypto_bits
        )
    
    def apply_posthoc(self, text: str, task: str) -> MultiLayerApplyResult:
        if task not in self.task_configs:
            raise ValueError(f"Task '{task}' non trovato")
        
        config = self.task_configs[task]
        stats = {}
        techniques_applied = []
        
        # Synonym
        if WatermarkTechnique.SYNONYM in config.techniques:
            synonym_wm = self._get_synonym_watermarker(config)
            synonym_result = synonym_wm.apply(text)
            text = synonym_result.text
            techniques_applied.append('synonym')
            stats['synonym'] = {
                'substitutions_made': synonym_result.substitutions_made
            }
        
        # Character
        if WatermarkTechnique.CHARACTER in config.techniques:
            char_wm = self._get_character_watermarker(config)
            char_result = char_wm.apply(text)
            text = char_result.text
            techniques_applied.append('character')
            stats['character'] = {
                'substituted_chars': char_result.substituted_chars
            }
        
        return MultiLayerApplyResult(
            text=text,
            task=task,
            techniques_applied=techniques_applied,
            stats=stats,
            crypto_boundaries=None,  # Non disponibili per post-hoc
            crypto_bits=None
        )
    
    def detect(
        self,
        text: str,
        task: str,
        tokenizer=None,
        crypto_boundaries: Optional[List[int]] = None,
        crypto_bits: Optional[List[int]] = None
    ) -> MultiLayerDetectResult:

        if task not in self.task_configs:
            raise ValueError(f"Task '{task}' non trovato")
        
        config = self.task_configs[task]
        technique_results = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        # Crypto detection (CON BOUNDARIES)
        if WatermarkTechnique.CRYPTO in config.techniques:
            if tokenizer is None:
                logger.warning("Tokenizer non fornito, skip crypto detection")
            elif crypto_boundaries is None or len(crypto_boundaries) == 0:
                logger.warning("Boundaries non forniti, skip crypto detection")
            else:
                crypto_wm = self._get_crypto_watermarker(config)
                crypto_result = crypto_wm.detect_with_boundaries(
                    text, tokenizer, crypto_boundaries, crypto_bits
                )
                technique_results['crypto'] = {
                    'detected': crypto_result.detected,
                    'confidence': crypto_result.confidence,
                    'score': crypto_result.best_score,
                    'margin': crypto_result.details.get('margin', 0)
                }
                weight = config.detection_weights.get(WatermarkTechnique.CRYPTO, 0.5)
                weighted_sum += weight * crypto_result.confidence
                total_weight += weight
        
        # Synonym detection
        if WatermarkTechnique.SYNONYM in config.techniques:
            synonym_wm = self._get_synonym_watermarker(config)
            synonym_result = synonym_wm.detect(text)
            technique_results['synonym'] = {
                'detected': synonym_result.detected,
                'confidence': synonym_result.confidence,
                'match_ratio': synonym_result.match_ratio
            }
            weight = config.detection_weights.get(WatermarkTechnique.SYNONYM, 0.3)
            weighted_sum += weight * synonym_result.confidence
            total_weight += weight
        
        # Character detection
        if WatermarkTechnique.CHARACTER in config.techniques:
            char_wm = self._get_character_watermarker(config)
            char_result = char_wm.detect(text)
            technique_results['character'] = {
                'detected': char_result.detected,
                'confidence': char_result.confidence,
                'match_ratio': char_result.match_ratio
            }
            weight = config.detection_weights.get(WatermarkTechnique.CHARACTER, 0.2)
            weighted_sum += weight * char_result.confidence
            total_weight += weight
        
        # Calcola score finale
        weighted_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # Soglia per detection (0.4 = moderatamente certo)
        threshold = 0.4
        detected = weighted_score > threshold
        
        return MultiLayerDetectResult(
            detected=detected,
            confidence=weighted_score,
            weighted_score=weighted_score,
            threshold=threshold,
            technique_results=technique_results
        )
    
    def detect_all_tasks(
        self,
        text: str,
        tokenizer=None,
        crypto_boundaries: Optional[List[int]] = None,
        crypto_bits: Optional[List[int]] = None
    ) -> FullDetectionResult:

        task_results = {}
        best_match = None
        best_confidence = 0.0
        
        for task in self.task_configs:
            result = self.detect(
                text, task, tokenizer, 
                crypto_boundaries, crypto_bits
            )
            task_results[task] = result
            
            if result.confidence > best_confidence:
                best_confidence = result.confidence
                best_match = task
        
        # Determina se è watermarked (best confidence > soglia)
        is_watermarked = best_confidence > 0.4
        
        return FullDetectionResult(
            task_results=task_results,
            best_match=best_match if is_watermarked else None,
            best_confidence=best_confidence,
            is_watermarked=is_watermarked
        )


# TEST
if __name__ == "__main__":
    print("=" * 70)
    print("TEST: Multi-Layer Watermarking System")
    print("=" * 70)
    
    system = MultiLayerWatermarker()
    
    print(f"\n1. TASK DISPONIBILI: {system.get_available_tasks()}")
    
    print("\n2. CONFIGURAZIONI:")
    for task, config in DEFAULT_TASK_CONFIGS.items():
        techniques = [t.value for t in config.techniques]
        print(f"   {task}: key={config.key}, techniques={techniques}")
    
    # Test post-hoc
    test_text = """
    The big problem with artificial intelligence is very important.
    We need to find a good solution quickly. Many researchers think
    it is difficult to make significant improvements without changing
    the fundamental approach to machine learning.
    """
    
    print(f"\n3. TESTO ORIGINALE:\n{test_text.strip()}")
    
    # Applica watermark post-hoc per task 'news'
    result = system.apply_posthoc(test_text, 'news')
    print(f"\n4. TESTO WATERMARKED (task='news'):\n{result.text.strip()}")
    print(f"   Tecniche: {result.techniques_applied}")
    print(f"   Crypto boundaries: {result.crypto_boundaries}")  # None per post-hoc
    
    # Detection (senza crypto perché post-hoc)
    print("\n5. DETECTION PER TASK (solo synonym/character):")
    for task in ['qa', 'summary', 'news']:
        det = system.detect(result.text, task)
        print(f"   {task}: detected={det.detected}, confidence={det.confidence:.2%}")
    
    print("\n" + "=" * 70)
    print("NOTA: Per test completo con crypto, eseguire scripts/demo.py")
    print("=" * 70)
