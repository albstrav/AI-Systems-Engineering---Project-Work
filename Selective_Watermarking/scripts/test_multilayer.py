#!/usr/bin/env python3

import sys
import os
import logging

# Silenzia log HTTP
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.generator import SelectiveWatermarkGenerator

print("=" * 60)
print("TEST: Multilayer Watermark + Generator")
print("=" * 60)

print('\nCaricamento modello (puo richiedere ~30 secondi)...')
gen = SelectiveWatermarkGenerator(model_name='gpt2-medium')
print('Modello caricato!')

all_passed = True

for task in ['qa', 'summary', 'news']:
    print(f'\n{"-"*60}')
    print(f'TASK: {task.upper()}')
    print('-'*60)
    
    result = gen.generate(
        prompt='Artificial intelligence is',
        task=task,
        max_tokens=100,
        use_watermark=True
    )
    
    print(f'Testo: {result.text[:80]}...')
    print(f'Token generati: {result.tokens_generated}')
    print(f'Tecniche: {result.techniques_applied}')
    print(f'Blocchi crypto: {len(result.crypto_boundaries) if result.crypto_boundaries else 0}')
    
    # Detection per tutti i task - passa testo e boundaries separatamente
    det_all = gen.detect_all_tasks(
        text=result.text,
        crypto_boundaries=result.crypto_boundaries,
        crypto_bits=result.crypto_bits
    )
    
    print(f'\nConfidence per task:')
    for t in ['qa', 'summary', 'news']:
        conf = det_all.task_results[t].confidence
        marker = ' --> CORRETTO' if t == task else ''
        print(f'  {t.upper()}: {conf:.2%}{marker}')
    
    # Verifica
    if det_all.best_match == task:
        print(f'\n[OK] Task identificato correttamente!')
    else:
        print(f'\n[FAIL] Atteso {task}, ottenuto {det_all.best_match}')
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("TUTTI I TEST PASSED")
else:
    print("ALCUNI TEST FAILED")
print("=" * 60)
