#!/usr/bin/env python3

import os
import sys
import json
import argparse
import csv
from datetime import datetime
from typing import Dict, List, Any
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def parse_args():
    parser = argparse.ArgumentParser(description='Valuta sistema watermarking')
    
    parser.add_argument('--dataset', type=str, default='data/generated/dataset.csv')
    parser.add_argument('--boundaries', type=str, default='data/generated/boundaries.json')
    parser.add_argument('--output-dir', type=str, default='data/results')
    parser.add_argument('--model', type=str, default='gpt2-medium')
    
    return parser.parse_args()


def load_dataset(csv_path: str) -> List[Dict]:
    """Carica il dataset da CSV."""
    samples = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['target_length'] = int(row['target_length'])
            row['actual_tokens'] = int(row['actual_tokens'])
            row['generation_time'] = float(row['generation_time'])
            samples.append(row)
    return samples


def load_boundaries(json_path: str) -> Dict:
    """Carica i boundaries da JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def evaluate_detection(
    samples: List[Dict],
    boundaries_data: Dict,
    watermarker,
    tokenizer
) -> List[Dict]:
    """Esegue detection su tutti i campioni."""
    results = []
    
    print(f"\nValutazione detection su {len(samples)} campioni...")
    
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  Processati: {i+1}/{len(samples)}")
        
        sample_id = sample['id']
        text = sample['text']
        
        # Recupera boundaries per questo sample
        sample_boundaries = boundaries_data.get(sample_id, {})
        crypto_boundaries = sample_boundaries.get('boundaries')
        crypto_bits = sample_boundaries.get('bits')
        
        # Esegui detection con boundaries
        detection = watermarker.detect_all_tasks(
            text, tokenizer,
            crypto_boundaries=crypto_boundaries,
            crypto_bits=crypto_bits
        )
        
        result = {
            'id': sample_id,
            'true_task': sample['task'],
            'target_length': sample['target_length'],
            'predicted_task': detection.best_match,
            'is_detected': detection.is_watermarked,
            'confidence': detection.best_confidence,
            'task_confidences': {
                task: det.confidence
                for task, det in detection.task_results.items()
            }
        }
        
        results.append(result)
    
    return results


def compute_task_accuracy(results: List[Dict]) -> Dict[str, Any]:
    """Calcola accuratezza per task."""
    accuracy = {}
    
    # Overall
    correct = sum(1 for r in results if r['predicted_task'] == r['true_task'])
    accuracy['overall'] = correct / len(results) if results else 0
    accuracy['correct'] = correct
    accuracy['total'] = len(results)
    
    # Per task
    for task in ['qa', 'summary', 'news']:
        task_samples = [r for r in results if r['true_task'] == task]
        if task_samples:
            task_correct = sum(1 for r in task_samples if r['predicted_task'] == task)
            accuracy[task] = {
                'accuracy': task_correct / len(task_samples),
                'correct': task_correct,
                'total': len(task_samples)
            }
        else:
            accuracy[task] = {'accuracy': 0, 'correct': 0, 'total': 0}
    
    return accuracy


def compute_confusion_matrix(results: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Calcola matrice di confusione per task."""
    tasks = ['qa', 'summary', 'news']
    matrix = {t1: {t2: 0 for t2 in tasks + ['none']} for t1 in tasks}
    
    for r in results:
        true_task = r['true_task']
        pred_task = r['predicted_task'] if r['predicted_task'] else 'none'
        if true_task in matrix:
            matrix[true_task][pred_task] += 1
    
    return matrix


def compute_length_impact(results: List[Dict]) -> Dict[int, Dict[str, float]]:
    """Calcola accuratezza per lunghezza."""
    by_length = defaultdict(list)
    for r in results:
        by_length[r['target_length']].append(r)
    
    impact = {}
    for length, samples in sorted(by_length.items()):
        correct = sum(1 for r in samples if r['predicted_task'] == r['true_task'])
        avg_conf = sum(r['confidence'] for r in samples) / len(samples)
        
        impact[length] = {
            'accuracy': correct / len(samples),
            'confidence_avg': avg_conf,
            'correct': correct,
            'total': len(samples)
        }
    
    return impact


def compute_detection_rate(results: List[Dict]) -> Dict[str, float]:
    """Calcola quanti testi vengono rilevati come watermarked."""
    detected = sum(1 for r in results if r['is_detected'])
    return {
        'detection_rate': detected / len(results) if results else 0,
        'detected': detected,
        'total': len(results)
    }


def save_results(
    results: List[Dict],
    task_accuracy: Dict,
    confusion_matrix: Dict,
    length_impact: Dict,
    detection_rate: Dict,
    output_dir: str
) -> None:
    """Salva tutti i risultati."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Risultati completi JSON
    full_results = {
        'evaluated_at': datetime.now().isoformat(),
        'n_samples': len(results),
        'task_accuracy': task_accuracy,
        'detection_rate': detection_rate,
        'confusion_matrix': confusion_matrix,
        'length_impact': {str(k): v for k, v in length_impact.items()}
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(full_results, f, indent=2)
    
    # 2. Confusion matrix CSV
    tasks = ['qa', 'summary', 'news', 'none']
    with open(os.path.join(output_dir, 'confusion_matrix.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true\\predicted'] + tasks)
        for true_task in ['qa', 'summary', 'news']:
            row = [true_task] + [confusion_matrix[true_task].get(t, 0) for t in tasks]
            writer.writerow(row)
    
    # 3. Length impact CSV
    with open(os.path.join(output_dir, 'length_impact.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['length', 'accuracy', 'confidence_avg', 'correct', 'total'])
        for length, data in sorted(length_impact.items()):
            writer.writerow([length, f"{data['accuracy']:.4f}", f"{data['confidence_avg']:.4f}", 
                           data['correct'], data['total']])
    
    # 4. Detailed predictions CSV
    with open(os.path.join(output_dir, 'detailed_predictions.csv'), 'w', newline='') as f:
        fieldnames = ['id', 'true_task', 'target_length', 'predicted_task', 'is_detected', 'confidence']
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nRisultati salvati in: {output_dir}/")


def print_summary(
    task_accuracy: Dict,
    confusion_matrix: Dict,
    length_impact: Dict,
    detection_rate: Dict
) -> None:
    """Stampa riepilogo dei risultati."""
    
    print("\n" + "=" * 70)
    print("RISULTATI VALUTAZIONE")
    print("=" * 70)
    
    # Detection Rate
    print("\n1. DETECTION RATE")
    print("-" * 40)
    print(f"   Testi rilevati: {detection_rate['detected']}/{detection_rate['total']}")
    print(f"   Detection Rate: {detection_rate['detection_rate']*100:.1f}%")
    
    # Task Accuracy
    print("\n2. TASK ACCURACY")
    print("-" * 40)
    print(f"   Overall: {task_accuracy['overall']*100:.1f}% ({task_accuracy['correct']}/{task_accuracy['total']})")
    for task in ['qa', 'summary', 'news']:
        t = task_accuracy[task]
        print(f"   {task.upper():8s}: {t['accuracy']*100:.1f}% ({t['correct']}/{t['total']})")
    
    # Confusion Matrix
    print("\n3. CONFUSION MATRIX")
    print("-" * 40)
    tasks = ['qa', 'summary', 'news', 'none']
    header = 'True\\Pred'
    print(f"   {header:<10}", end='')
    for t in tasks:
        print(f"{t:>8}", end='')
    print()
    for true_task in ['qa', 'summary', 'news']:
        print(f"   {true_task:<10}", end='')
        for pred_task in tasks:
            count = confusion_matrix[true_task].get(pred_task, 0)
            print(f"{count:>8}", end='')
        print()
    
    # Length Impact
    print("\n4. LENGTH IMPACT")
    print("-" * 40)
    print(f"   {'Length':<10} {'Accuracy':<12} {'Confidence':<12} {'Samples':<10}")
    for length, data in sorted(length_impact.items()):
        print(f"   {length:<10} {data['accuracy']*100:>6.1f}%      {data['confidence_avg']:>6.2f}        {data['total']}")
    
    print("\n" + "=" * 70)


def main():
    args = parse_args()
    
    print("=" * 70)
    print("EVALUATION - Selective Watermarking")
    print("=" * 70)
    
    # Verifica files
    if not os.path.exists(args.dataset):
        print(f"\nERRORE: Dataset non trovato: {args.dataset}")
        print("Eseguire prima: python scripts/generate_dataset.py")
        return 1
    
    if not os.path.exists(args.boundaries):
        print(f"\nERRORE: Boundaries non trovati: {args.boundaries}")
        print("Eseguire prima: python scripts/generate_dataset.py")
        return 1
    
    # Carica dati
    print(f"\nCaricamento dataset: {args.dataset}")
    samples = load_dataset(args.dataset)
    print(f"Campioni caricati: {len(samples)}")
    
    print(f"\nCaricamento boundaries: {args.boundaries}")
    boundaries_data = load_boundaries(args.boundaries)
    print(f"Boundaries caricati: {len(boundaries_data)}")
    
    # Carica moduli
    print("\nCaricamento moduli...")
    from transformers import GPT2Tokenizer
    from watermarkers.multilayer import MultiLayerWatermarker
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    watermarker = MultiLayerWatermarker(vocab_size=tokenizer.vocab_size)
    
    # Esegui detection
    results = evaluate_detection(samples, boundaries_data, watermarker, tokenizer)
    
    # Calcola metriche
    print("\nCalcolo metriche...")
    task_accuracy = compute_task_accuracy(results)
    confusion_matrix = compute_confusion_matrix(results)
    length_impact = compute_length_impact(results)
    detection_rate = compute_detection_rate(results)
    
    # Salva risultati
    save_results(
        results=results,
        task_accuracy=task_accuracy,
        confusion_matrix=confusion_matrix,
        length_impact=length_impact,
        detection_rate=detection_rate,
        output_dir=args.output_dir
    )
    
    # Stampa riepilogo
    print_summary(task_accuracy, confusion_matrix, length_impact, detection_rate)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
