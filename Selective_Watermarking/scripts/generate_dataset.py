#!/usr/bin/env python3

import os
import sys
import json
import argparse
import time
from datetime import datetime
from typing import List, Dict, Any
import csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Genera dataset per valutazione watermarking')
    
    parser.add_argument('--model', type=str, default='gpt2-medium',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large'])
    parser.add_argument('--tasks', type=str, nargs='+', default=['qa', 'summary', 'news'])
    parser.add_argument('--lengths', type=int, nargs='+', default=[50, 100, 200])
    parser.add_argument('--samples-per-length', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='data/generated')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-individual', action='store_true', default=False)
    
    return parser.parse_args()


def setup_output_dir(output_dir: str) -> None:
    """Crea directory di output se non esistono."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples'), exist_ok=True)


def generate_dataset(
    generator,
    prompts_module,
    tasks: List[str],
    lengths: List[int],
    samples_per_length: int,
    output_dir: str,
    save_individual: bool,
    seed: int
) -> Dict[str, Any]:
    """Genera il dataset completo."""
    import random
    random.seed(seed)
    
    all_samples = []
    all_boundaries = {}
    
    stats = {
        'total_samples': 0,
        'by_task': {},
        'by_length': {},
        'generation_time': 0,
        'errors': 0
    }
    
    start_time = time.time()
    
    # Genera per ogni task
    for task in tasks:
        print(f"\n{'='*60}")
        print(f"TASK: {task.upper()}")
        print(f"{'='*60}")
        
        prompts = prompts_module.get_prompts_for_task(task)
        stats['by_task'][task] = {'total': 0, 'by_length': {}}
        
        for length in lengths:
            print(f"\n  Lunghezza: {length} token")
            print(f"  Campioni: {samples_per_length}")
            
            stats['by_task'][task]['by_length'][length] = 0
            if length not in stats['by_length']:
                stats['by_length'][length] = 0
            
            for i in range(samples_per_length):
                prompt_idx = i % len(prompts)
                prompt = prompts[prompt_idx]
                
                try:
                    result = generator.generate(
                        prompt=prompt,
                        task=task,
                        max_tokens=length,
                        use_watermark=True
                    )
                    
                    sample_id = f"{task}_{length}tok_{i+1:03d}"
                    
                    sample = {
                        'id': sample_id,
                        'task': task,
                        'target_length': length,
                        'actual_tokens': result.tokens_generated,
                        'prompt': prompt,
                        'text': result.text,
                        'techniques': ','.join(result.techniques_applied),
                        'generation_time': result.generation_time
                    }
                    
                    # Salva boundaries
                    if result.crypto_boundaries:
                        all_boundaries[sample_id] = {
                            'boundaries': result.crypto_boundaries,
                            'bits': result.crypto_bits
                        }
                    
                    all_samples.append(sample)
                    
                    stats['total_samples'] += 1
                    stats['by_task'][task]['total'] += 1
                    stats['by_task'][task]['by_length'][length] += 1
                    stats['by_length'][length] += 1
                    
                    if save_individual:
                        filepath = os.path.join(output_dir, 'samples', f"{sample_id}.txt")
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(result.text)
                    
                    if (i + 1) % 10 == 0:
                        print(f"    Generati: {i+1}/{samples_per_length}")
                        
                except Exception as e:
                    print(f"    ERRORE campione {i+1}: {e}")
                    stats['errors'] += 1
    
    stats['generation_time'] = time.time() - start_time
    
    # Salva dataset CSV
    csv_path = os.path.join(output_dir, 'dataset.csv')
    fieldnames = ['id', 'task', 'target_length', 'actual_tokens', 'prompt', 
                  'text', 'techniques', 'generation_time']
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_samples)
    
    print(f"\nDataset salvato: {csv_path}")
    
    # Salva boundaries JSON
    boundaries_path = os.path.join(output_dir, 'boundaries.json')
    with open(boundaries_path, 'w') as f:
        json.dump(all_boundaries, f)
    
    print(f"Boundaries salvati: {boundaries_path}")
    
    return stats


def save_metadata(stats: Dict[str, Any], args, output_dir: str) -> None:
    """Salva metadati della generazione."""
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'model': args.model,
        'tasks': args.tasks,
        'lengths': args.lengths,
        'samples_per_length': args.samples_per_length,
        'seed': args.seed,
        'statistics': stats
    }
    
    filepath = os.path.join(output_dir, 'metadata.json')
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadati salvati: {filepath}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("DATASET GENERATION - Selective Watermarking")
    print("=" * 70)
    print(f"\nConfigurazione:")
    print(f"  Model:             {args.model}")
    print(f"  Tasks:             {args.tasks}")
    print(f"  Lengths:           {args.lengths}")
    print(f"  Samples/length:    {args.samples_per_length}")
    print(f"  Output dir:        {args.output_dir}")
    
    n_total = len(args.tasks) * len(args.lengths) * args.samples_per_length
    
    print(f"\n  TOTALE CAMPIONI:   {n_total}")
    
    print(f"\nStima tempo: ~{n_total * 5 / 60:.1f} minuti (con GPU)")
    response = input("Procedere? [y/N]: ")
    if response.lower() != 'y':
        print("Annullato.")
        return
    
    setup_output_dir(args.output_dir)
    
    print("\nCaricamento moduli...")
    from generation.generator import SelectiveWatermarkGenerator
    from generation import prompts as prompts_module
    
    print(f"\nCaricamento modello {args.model}...")
    generator = SelectiveWatermarkGenerator(model_name=args.model)
    
    stats = generate_dataset(
        generator=generator,
        prompts_module=prompts_module,
        tasks=args.tasks,
        lengths=args.lengths,
        samples_per_length=args.samples_per_length,
        output_dir=args.output_dir,
        save_individual=args.save_individual,
        seed=args.seed
    )
    
    save_metadata(stats, args, args.output_dir)
    
    print("\n" + "=" * 70)
    print("GENERAZIONE COMPLETATA")
    print("=" * 70)
    print(f"\nStatistiche:")
    print(f"  Campioni totali:  {stats['total_samples']}")
    print(f"  Errori:           {stats['errors']}")
    print(f"  Tempo totale:     {stats['generation_time']:.1f} secondi")
    print(f"\nPer task:")
    for task, task_stats in stats['by_task'].items():
        print(f"  {task}: {task_stats['total']} campioni")
    print(f"\nOutput salvato in: {args.output_dir}/")
    print("  - dataset.csv (testi)")
    print("  - boundaries.json (boundaries crypto)")
    print("  - metadata.json (metadati)")
    print("=" * 70)


if __name__ == "__main__":
    main()
