#!/usr/bin/env python3

import os
import sys
import csv
import json
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Unisce i dataset dei task')
    
    parser.add_argument(
        '--qa', type=str, default='data/qa_dataset.csv',
        help='Dataset QA (default: data/qa_dataset.csv)'
    )
    
    parser.add_argument(
        '--summary', type=str, default='data/summary_dataset.csv',
        help='Dataset Summary (default: data/summary_dataset.csv)'
    )
    
    parser.add_argument(
        '--news', type=str, default='data/news_dataset.csv',
        help='Dataset News (default: data/news_dataset.csv)'
    )
    
    parser.add_argument(
        '--output', type=str, default='data/generated/dataset.csv',
        help='File output unificato (default: data/generated/dataset.csv)'
    )
    
    parser.add_argument(
        '--boundaries-output', type=str, default='data/generated/boundaries.json',
        help='File JSON boundaries output (default: data/generated/boundaries.json)'
    )
    
    return parser.parse_args()


def load_csv(filepath):
    """Carica un file CSV e restituisce lista di dizionari."""
    if not os.path.exists(filepath):
        return None
    
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def load_boundaries(csv_path):
    """Carica il file JSON boundaries associato al CSV."""
    json_path = csv_path.replace('.csv', '_boundaries.json')
    if not os.path.exists(json_path):
        return {}
    
    with open(json_path, 'r') as f:
        return json.load(f)


def main():
    args = parse_args()
    
    print("=" * 70)
    print("MERGE DATASETS")
    print("=" * 70)
    
    # Carica tutti i dataset
    datasets = {
        'qa': (args.qa, load_csv(args.qa)),
        'summary': (args.summary, load_csv(args.summary)),
        'news': (args.news, load_csv(args.news))
    }
    
    # Verifica quali sono disponibili
    print("\nVerifica file:")
    all_samples = []
    all_boundaries = {}
    stats = {}
    
    for task, (path, data) in datasets.items():
        if data is None:
            print(f"  ✗ {task.upper()}: {path} NON TROVATO")
            stats[task] = 0
        else:
            print(f"  ✓ {task.upper()}: {path} ({len(data)} campioni)")
            all_samples.extend(data)
            stats[task] = len(data)
            
            # Carica boundaries
            boundaries = load_boundaries(path)
            if boundaries:
                all_boundaries.update(boundaries)
                print(f"    └─ Boundaries: {len(boundaries)} entries")
    
    if not all_samples:
        print("\nERRORE: Nessun dataset trovato!")
        print("Genera prima i dataset con:")
        print("  python scripts/generate_single_task.py --task qa")
        print("  python scripts/generate_single_task.py --task summary")
        print("  python scripts/generate_single_task.py --task news")
        return 1
    
    # Determina le colonne
    fieldnames = list(all_samples[0].keys())
    
    # Crea directory output
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Salva dataset unificato
    print(f"\nSalvataggio: {args.output}")
    
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_samples)
    
    # Salva boundaries unificati
    print(f"Salvataggio: {args.boundaries_output}")
    with open(args.boundaries_output, 'w') as f:
        json.dump(all_boundaries, f)
    
    # Salva metadata
    metadata_path = args.output.replace('.csv', '_metadata.json')
    metadata = {
        'merged_at': datetime.now().isoformat(),
        'source_files': {
            'qa': args.qa if stats.get('qa', 0) > 0 else None,
            'summary': args.summary if stats.get('summary', 0) > 0 else None,
            'news': args.news if stats.get('news', 0) > 0 else None,
        },
        'samples_per_task': stats,
        'total_samples': len(all_samples),
        'total_boundaries': len(all_boundaries)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Riepilogo
    print("\n" + "=" * 70)
    print("MERGE COMPLETATO!")
    print("=" * 70)
    print(f"\nStatistiche:")
    for task, count in stats.items():
        print(f"  {task.upper():10s}: {count:4d} campioni")
    print(f"  {'TOTALE':10s}: {len(all_samples):4d} campioni")
    print(f"  {'BOUNDARIES':10s}: {len(all_boundaries):4d} entries")
    
    print(f"\nFile generati:")
    print(f"  Dataset:    {args.output}")
    print(f"  Boundaries: {args.boundaries_output}")
    print(f"  Metadata:   {metadata_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
