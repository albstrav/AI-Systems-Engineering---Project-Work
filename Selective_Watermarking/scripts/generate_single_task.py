#!/usr/bin/env python3

import os
import sys
import csv
import json
import time
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Genera dataset per un singolo task',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
    python generate_single_task.py --task qa
    python generate_single_task.py --task summary --samples 30
    python generate_single_task.py --task news --lengths 100 200
        """
    )
    
    parser.add_argument(
        '--task', type=str, required=True,
        choices=['qa', 'summary', 'news'],
        help='Task da generare'
    )
    
    parser.add_argument(
        '--output', type=str, default=None,
        help='File CSV di output (default: data/{task}_dataset.csv)'
    )
    
    parser.add_argument(
        '--boundaries-output', type=str, default=None,
        help='File JSON boundaries (default: data/{task}_boundaries.json)'
    )
    
    parser.add_argument(
        '--model', type=str, default='gpt2-medium',
        choices=['gpt2', 'gpt2-medium', 'gpt2-large'],
        help='Modello GPT-2 (default: gpt2-medium)'
    )
    
    parser.add_argument(
        '--samples', type=int, default=50,
        help='Campioni per lunghezza (default: 50)'
    )
    
    parser.add_argument(
        '--lengths', type=int, nargs='+', default=[50, 100, 200],
        help='Lunghezze token (default: 50 100 200)'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        os.makedirs('data', exist_ok=True)
        args.output = f'data/{args.task}_dataset.csv'
    
    if args.boundaries_output is None:
        os.makedirs('data', exist_ok=True)
        args.boundaries_output = f'data/{args.task}_boundaries.json'
    
    return args


def main():
    args = parse_args()
    
    print("=" * 70)
    print(f"GENERAZIONE TASK: {args.task.upper()}")
    print("=" * 70)
    
    # Info
    n_watermarked = args.samples * len(args.lengths)
    
    print(f"\nConfigurazione:")
    print(f"  Task:           {args.task}")
    print(f"  Model:          {args.model}")
    print(f"  Lengths:        {args.lengths}")
    print(f"  Samples/length: {args.samples}")
    print(f"  Output CSV:     {args.output}")
    print(f"  Output JSON:    {args.boundaries_output}")
    print(f"\n  TOTALE:         {n_watermarked} campioni")
    
    print(f"\nStima tempo: ~{n_watermarked * 5 / 60:.1f} minuti")
    response = input("Procedere? [y/N]: ")
    if response.lower() != 'y':
        print("Annullato.")
        return 1
    
    # Carica moduli
    print("\nCaricamento moduli...")
    from generation.generator import SelectiveWatermarkGenerator
    from generation import prompts as prompts_module
    
    print(f"\nCaricamento modello {args.model}...")
    generator = SelectiveWatermarkGenerator(model_name=args.model)
    
    # Genera
    print(f"\nGenerazione campioni watermarked...")
    
    prompts = prompts_module.get_prompts_for_task(args.task)
    samples = []
    boundaries_data = {}
    errors = 0
    
    import random
    random.seed(args.seed)
    
    start_time = time.time()
    
    for length in args.lengths:
        print(f"\n  Lunghezza: {length} token")
        
        for i in range(args.samples):
            prompt_idx = i % len(prompts)
            prompt = prompts[prompt_idx]
            
            try:
                result = generator.generate(
                    prompt=prompt,
                    task=args.task,
                    max_tokens=length,
                    use_watermark=True
                )
                
                sample_id = f"{args.task}_{length}tok_{i+1:03d}"
                
                sample = {
                    'id': sample_id,
                    'task': args.task,
                    'target_length': length,
                    'actual_tokens': result.tokens_generated,
                    'prompt': prompt,
                    'text': result.text,
                    'techniques': ','.join(result.techniques_applied),
                    'generation_time': result.generation_time
                }
                samples.append(sample)
                
                # Salva boundaries
                if result.crypto_boundaries:
                    boundaries_data[sample_id] = {
                        'boundaries': result.crypto_boundaries,
                        'bits': result.crypto_bits
                    }
                
                if (i + 1) % 10 == 0:
                    print(f"    Completati: {i+1}/{args.samples}")
                    
            except Exception as e:
                print(f"    ERRORE campione {i+1}: {e}")
                errors += 1
    
    total_time = time.time() - start_time
    
    # Salva CSV
    print(f"\nSalvataggio {args.output}...")
    fieldnames = ['id', 'task', 'target_length', 'actual_tokens', 'prompt', 
                  'text', 'techniques', 'generation_time']
    
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    
    # Salva boundaries JSON
    print(f"Salvataggio {args.boundaries_output}...")
    with open(args.boundaries_output, 'w') as f:
        json.dump(boundaries_data, f)
    
    # Riepilogo
    print("\n" + "=" * 70)
    print("COMPLETATO")
    print("=" * 70)
    print(f"\nStatistiche:")
    print(f"  Campioni generati: {len(samples)}")
    print(f"  Errori:            {errors}")
    print(f"  Tempo totale:      {total_time:.1f}s")
    print(f"\nOutput:")
    print(f"  CSV:  {args.output}")
    print(f"  JSON: {args.boundaries_output}")
    
    # Metadata
    metadata = {
        'task': args.task,
        'generated_at': datetime.now().isoformat(),
        'model': args.model,
        'samples_per_length': args.samples,
        'lengths': args.lengths,
        'total_samples': len(samples),
        'errors': errors,
        'generation_time': total_time
    }
    
    metadata_path = args.output.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")
    
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
