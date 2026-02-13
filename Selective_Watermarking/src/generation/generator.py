#!/usr/bin/env python3

import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    text: str
    prompt: str
    task: str
    techniques_applied: List[str]
    tokens_generated: int
    generation_time: float
    stats: Dict[str, Any]
    # NUOVO: metadata per detection crypto
    crypto_boundaries: Optional[List[int]] = None
    crypto_bits: Optional[List[int]] = None


class SelectiveWatermarkGenerator:
    def __init__(
        self,
        model_name: str = 'gpt2-medium',
        device: Optional[str] = None,
        load_model: bool = True
    ):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.watermarker = None
        
        # Auto-detect device
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"SelectiveWatermarkGenerator: model={model_name}, device={self.device}")
        
        if load_model:
            self._load_model()
            self._init_watermarker()
    
    def _load_model(self):
        """Carica il modello GPT-2 e il tokenizer."""
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        logger.info(f"Caricamento modello {self.model_name}...")
        start_time = time.time()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        
        # Imposta padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Sposta su device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"Modello caricato in {load_time:.2f}s")
    
    def _init_watermarker(self):
        """Inizializza il sistema di watermarking multi-layer."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        
        from watermarkers.multilayer import MultiLayerWatermarker
        
        self.watermarker = MultiLayerWatermarker(
            vocab_size=self.tokenizer.vocab_size
        )
        
        logger.info(f"Watermarker inizializzato con task: {self.watermarker.get_available_tasks()}")
    
    def is_ready(self) -> bool:
        """Verifica se il generatore è pronto per l'uso."""
        return (
            self.model is not None and 
            self.tokenizer is not None and 
            self.watermarker is not None
        )
    
    def get_available_tasks(self) -> List[str]:
        """Restituisce i task disponibili."""
        if self.watermarker is None:
            return []
        return self.watermarker.get_available_tasks()
    
    def generate(
        self,
        prompt: str,
        task: str,
        max_tokens: int = 100,
        use_watermark: bool = True
    ) -> GenerationResult:
        
        if not self.is_ready():
            raise RuntimeError("Generator non inizializzato. Chiama _load_model() prima.")
        
        if task not in self.get_available_tasks():
            raise ValueError(f"Task '{task}' non valido. Disponibili: {self.get_available_tasks()}")
        
        start_time = time.time()
        
        crypto_boundaries = None
        crypto_bits = None
        
        if use_watermark:
            # Generazione con watermark
            result = self.watermarker.generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                task=task,
                max_new_tokens=max_tokens
            )
            
            text = result.text
            techniques = result.techniques_applied
            stats = result.stats
            crypto_boundaries = result.crypto_boundaries
            crypto_bits = result.crypto_bits
            
            # Conta token generati (il text dal watermarker NON include il prompt)
            tokens_generated = len(self.tokenizer.encode(text))
        else:
            # Generazione normale senza watermark
            import torch
            
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            techniques = []
            stats = {}
            tokens_generated = output.shape[1] - input_ids.shape[1]
        
        generation_time = time.time() - start_time
        
        return GenerationResult(
            text=text,
            prompt=prompt,
            task=task,
            techniques_applied=techniques,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            stats=stats,
            crypto_boundaries=crypto_boundaries,
            crypto_bits=crypto_bits
        )
    
    def detect(
        self,
        text: str,
        task: str,
        crypto_boundaries: Optional[List[int]] = None,
        crypto_bits: Optional[List[int]] = None
    ):
    
        if not self.is_ready():
            raise RuntimeError("Generator non inizializzato.")
        
        return self.watermarker.detect(
            text=text,
            task=task,
            tokenizer=self.tokenizer,
            crypto_boundaries=crypto_boundaries,
            crypto_bits=crypto_bits
        )
    
    def detect_all_tasks(
        self,
        text: str,
        crypto_boundaries: Optional[List[int]] = None,
        crypto_bits: Optional[List[int]] = None
    ):
      
        if not self.is_ready():
            raise RuntimeError("Generator non inizializzato.")
        
        return self.watermarker.detect_all_tasks(
            text=text,
            tokenizer=self.tokenizer,
            crypto_boundaries=crypto_boundaries,
            crypto_bits=crypto_bits
        )
    
    def generate_batch(
        self,
        prompts: List[str],
        task: str,
        max_tokens: int = 100,
        use_watermark: bool = True,
        show_progress: bool = True
    ) -> List[GenerationResult]:
     
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(prompts, desc=f"Generating ({task})")
            except ImportError:
                iterator = prompts
                logger.info(f"Generazione di {len(prompts)} testi per task '{task}'...")
        else:
            iterator = prompts
        
        for prompt in iterator:
            try:
                result = self.generate(
                    prompt=prompt,
                    task=task,
                    max_tokens=max_tokens,
                    use_watermark=use_watermark
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Errore generazione per prompt '{prompt[:50]}...': {e}")
                # Crea risultato vuoto per mantenere allineamento
                results.append(GenerationResult(
                    text="[ERROR]",
                    prompt=prompt,
                    task=task,
                    techniques_applied=[],
                    tokens_generated=0,
                    generation_time=0.0,
                    stats={'error': str(e)},
                    crypto_boundaries=None,
                    crypto_bits=None
                ))
        
        return results


# FUNZIONI DI UTILITÀ
def create_generator(
    model_name: str = 'gpt2-medium',
    device: Optional[str] = None
) -> SelectiveWatermarkGenerator:
  
    return SelectiveWatermarkGenerator(model_name=model_name, device=device)


# TEST
if __name__ == "__main__":
    print("=" * 70)
    print("TEST: SelectiveWatermarkGenerator")
    print("=" * 70)
    
    # Test senza caricare il modello (per verificare la struttura)
    print("\n1. Inizializzazione (senza modello)...")
    
    try:
        gen = SelectiveWatermarkGenerator(load_model=False)
        print(f"   Device: {gen.device}")
        print(f"   Ready: {gen.is_ready()}")
        
        print("\n2. Per test completo con generazione, eseguire:")
        print("   python scripts/demo.py")
        
    except Exception as e:
        print(f"   Errore: {e}")
    
    print("\n" + "=" * 70)
