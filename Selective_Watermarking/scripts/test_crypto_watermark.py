#!/usr/bin/env python3

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    parser = argparse.ArgumentParser(description='Test watermark crittografico')
    parser.add_argument('--security-param', type=float, default=2.0)
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--model', type=str, default='gpt2-medium')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("TEST: Christ et al. Watermarking con GPT-2")
    print("=" * 70)
    
    # Import
    print("\n[1/5] Importazione librerie...")
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        from watermarkers.crypto_watermark_christ import ChristWatermarker, BITS_PER_TOKEN
        
        print(f"    ✓ PyTorch version: {torch.__version__}")
        print(f"    ✓ CUDA disponibile: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"    ✗ Errore importazione: {e}")
        return 1
    
    # Carica modello
    print(f"\n[2/5] Caricamento {args.model}...")
    start_time = time.time()
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"    ✓ Modello caricato in {time.time() - start_time:.2f}s")
    print(f"    ✓ Vocab size: {tokenizer.vocab_size}")
    print(f"    ✓ EOS token: {tokenizer.eos_token_id}")
    
    # Watermarker
    print(f"\n[3/5] Inizializzazione watermarker (λ={args.security_param})...")
    secret_key = 314159265
    watermarker = ChristWatermarker(
        secret_key=secret_key,
        security_param=args.security_param,
        vocab_size=tokenizer.vocab_size
    )
    print(f"    ✓ Chiave segreta: {secret_key}")
    print(f"    ✓ Bits per token: {BITS_PER_TOKEN}")
    
    # Generazione
    print(f"\n[4/5] Generazione testo watermarked (max {args.max_tokens} token)...")
    prompt = "Artificial intelligence is"
    print(f"    Prompt: '{prompt}'")
    
    start_time = time.time()
    result = watermarker.generate(model, tokenizer, prompt, max_new_tokens=args.max_tokens)
    gen_time = time.time() - start_time
    
    print(f"\n    ✓ Generazione completata in {gen_time:.2f}s")
    print(f"\n    TESTO GENERATO:")
    print("    " + "-" * 60)
    print(f"    {result.text[:500]}...")
    print("    " + "-" * 60)
    
    print(f"\n    STATISTICHE:")
    print(f"    - Token generati: {len(result.tokens)}")
    print(f"    - Bit generati: {result.total_bits}")
    print(f"    - Entropia empirica: {result.empirical_entropy:.2f}")
    print(f"    - Blocchi creati: {result.blocks_created}")
    print(f"    - Watermark attivo da bit: {result.watermark_active_from_bit}")
    print(f"    - Boundaries: {result.block_boundaries[:5]}...")
    
    # Detection
    print("\n[5/5] Detection...")
    
    # A) Chiave corretta CON boundaries
    print("\n    A) Con chiave CORRETTA e boundaries:")
    start_time = time.time()
    detection_correct = watermarker.detect_with_boundaries(
        result.text, tokenizer,
        boundaries=result.block_boundaries,
        original_bits=result.bits
    )
    det_time = time.time() - start_time
    
    print(f"    - Rilevato: {detection_correct.detected}")
    print(f"    - Confidence: {detection_correct.confidence:.2%}")
    print(f"    - Best score: {detection_correct.best_score:.2f}")
    print(f"    - Threshold: {detection_correct.threshold:.2f}")
    print(f"    - Margin: {detection_correct.details.get('margin', 0):.2f}")
    print(f"    - Tempo: {det_time:.2f}s")
    
    # B) Chiave sbagliata CON boundaries
    print("\n    B) Con chiave SBAGLIATA e boundaries:")
    watermarker_wrong = ChristWatermarker(
        secret_key=999999999,
        security_param=args.security_param,
        vocab_size=tokenizer.vocab_size
    )
    
    detection_wrong = watermarker_wrong.detect_with_boundaries(
        result.text, tokenizer,
        boundaries=result.block_boundaries,
        original_bits=result.bits
    )
    
    print(f"    - Rilevato: {detection_wrong.detected}")
    print(f"    - Confidence: {detection_wrong.confidence:.2%}")
    print(f"    - Best score: {detection_wrong.best_score:.2f}")
    print(f"    - Margin: {detection_wrong.details.get('margin', 0):.2f}")
    
    # C) Testo normale (non watermarked)
    print("\n    C) Con testo NON watermarked:")
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    normal_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Per testo normale non abbiamo boundaries, usiamo detect() deprecato
    detection_normal = watermarker.detect(normal_text, tokenizer)
    
    print(f"    Testo: '{normal_text[:60]}...'")
    print(f"    - Rilevato: {detection_normal.detected}")
    print(f"    - Confidence: {detection_normal.confidence:.2%}")
    
    # Riepilogo
    print("\n" + "=" * 70)
    print("RIEPILOGO TEST")
    print("=" * 70)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Generazione produce testo
    if len(result.text) > len(prompt):
        print("[✓] Generazione produce testo")
        tests_passed += 1
    else:
        print("[✗] Generazione non produce testo")
    
    # Test 2: Blocchi creati
    if result.blocks_created > 0:
        print(f"[✓] Blocchi watermark creati: {result.blocks_created}")
        tests_passed += 1
    else:
        print("[✗] Nessun blocco watermark creato")
    
    # Test 3: Detection chiave corretta
    if detection_correct.detected and detection_correct.confidence > 0.3:
        print(f"[✓] Detection con chiave corretta (conf={detection_correct.confidence:.2%})")
        tests_passed += 1
    else:
        print(f"[✗] Detection con chiave corretta (conf={detection_correct.confidence:.2%})")
    
    # Test 4: Chiave sbagliata ha confidence minore
    if detection_correct.confidence > detection_wrong.confidence:
        print("[✓] Chiave sbagliata ha confidence minore")
        tests_passed += 1
    else:
        print("[✗] Chiave sbagliata NON ha confidence minore")
    
    # Test 5: Testo normale non rilevato (o confidence bassa)
    if not detection_normal.detected or detection_normal.confidence < 0.3:
        print("[✓] Testo normale ha confidence bassa")
        tests_passed += 1
    else:
        print(f"[✗] Testo normale rilevato (conf={detection_normal.confidence:.2%})")
    
    print("\n" + "=" * 70)
    if tests_passed == total_tests:
        print("TUTTI I TEST SUPERATI ✓")
    else:
        print(f"TEST SUPERATI: {tests_passed}/{total_tests}")
    print("=" * 70)
    
    return 0 if tests_passed == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
