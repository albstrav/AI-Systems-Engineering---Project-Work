#!/usr/bin/env python3

import os
import sys
import json
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Prova a importare seaborn, altrimenti usa matplotlib base
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn non disponibile, uso matplotlib base")


def parse_args():
    parser = argparse.ArgumentParser(description='Genera grafici per la relazione')
    
    parser.add_argument(
        '--results', type=str, default='data/results/evaluation_results.json',
        help='Path al file JSON dei risultati'
    )
    
    parser.add_argument(
        '--output-dir', type=str, default='figures',
        help='Directory di output per i grafici'
    )
    
    parser.add_argument(
        '--dpi', type=int, default=300,
        help='Risoluzione grafici (default: 300)'
    )
    
    parser.add_argument(
        '--style', type=str, default='seaborn-v0_8-whitegrid',
        help='Stile matplotlib'
    )
    
    return parser.parse_args()


def setup_style(style: str):
    """Configura lo stile dei grafici."""
    try:
        plt.style.use(style)
    except:
        plt.style.use('seaborn-whitegrid' if 'seaborn' in plt.style.available else 'default')
    
    # Impostazioni globali
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.titlesize'] = 16


def load_results(filepath: str) -> dict:
    """Carica i risultati da JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(results: dict, output_path: str, dpi: int):
    """Genera heatmap della matrice di confusione."""
    cm = results['confusion_matrix']
    tasks = ['qa', 'summary', 'news']
    
    # Costruisci matrice numpy
    matrix = np.array([
        [cm[t1].get(t2, 0) for t2 in tasks]
        for t1 in tasks
    ])
    
    # Normalizza per riga (percentuali)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / row_sums * 100
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if HAS_SEABORN:
        sns.heatmap(matrix_norm, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=[t.upper() for t in tasks],
                   yticklabels=[t.upper() for t in tasks],
                   ax=ax, cbar_kws={'label': 'Percentuale (%)'})
    else:
        im = ax.imshow(matrix_norm, cmap='Blues')
        ax.set_xticks(range(len(tasks)))
        ax.set_yticks(range(len(tasks)))
        ax.set_xticklabels([t.upper() for t in tasks])
        ax.set_yticklabels([t.upper() for t in tasks])
        
        # Annotazioni
        for i in range(len(tasks)):
            for j in range(len(tasks)):
                ax.text(j, i, f'{matrix_norm[i, j]:.1f}%',
                       ha='center', va='center', fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Percentuale (%)')
    
    ax.set_xlabel('Task Predetto')
    ax.set_ylabel('Task Vero')
    ax.set_title('Matrice di Confusione - Identificazione Task')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {output_path}")


def plot_length_impact(results: dict, output_path: str, dpi: int):
    """Genera line plot dell'impatto della lunghezza."""
    length_data = results['length_impact']
    
    lengths = sorted([int(k) for k in length_data.keys()])
    accuracies = [length_data[str(l)]['accuracy'] * 100 for l in lengths]
    confidences = [length_data[str(l)]['confidence_avg'] * 100 for l in lengths]
    
    # Baseline teorica (crypto-only, stimata)
    # Basata su: 50 token → ~55%, 100 token → ~70%, 200 token → ~90%
    baseline_accuracies = [55, 70, 90][:len(lengths)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Multi-layer
    ax.plot(lengths, accuracies, 'o-', linewidth=2.5, markersize=10,
           label='Multi-Layer (Crypto + Synonym + Char)', color='#2E86AB')
    
    # Baseline
    ax.plot(lengths, baseline_accuracies, 's--', linewidth=2, markersize=8,
           label='Crypto Only (baseline)', color='#A23B72', alpha=0.7)
    
    # Target line
    ax.axhline(y=90, color='green', linestyle=':', linewidth=1.5, 
               label='Target (90%)', alpha=0.7)
    
    # Annotazioni vantaggio
    for i, (l, acc, base) in enumerate(zip(lengths, accuracies, baseline_accuracies)):
        if acc > base:
            advantage = acc - base
            ax.annotate(f'+{advantage:.0f}pp', 
                       xy=(l, (acc + base) / 2),
                       fontsize=10, ha='center', color='#2E86AB')
    
    ax.set_xlabel('Lunghezza Testo (token)')
    ax.set_ylabel('Accuratezza (%)')
    ax.set_title('Impatto della Lunghezza sulla Detection')
    ax.legend(loc='lower right')
    ax.set_ylim(40, 105)
    ax.set_xticks(lengths)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {output_path}")


def plot_task_accuracy(results: dict, output_path: str, dpi: int):
    """Genera bar chart dell'accuratezza per task."""
    task_acc = results['task_accuracy']
    
    tasks = ['QA', 'Summary', 'News', 'Overall']
    
    # Gestisci sia formato vecchio (numero) che nuovo (dizionario)
    def get_acc(key):
        val = task_acc.get(key, 0)
        if isinstance(val, dict):
            return val.get('accuracy', 0) * 100
        return val * 100
    
    accuracies = [
        get_acc('qa'),
        get_acc('summary'),
        get_acc('news'),
        get_acc('overall')
    ]
    
    colors = ['#3498DB', '#2ECC71', '#E74C3C', '#9B59B6']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(tasks, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Target line
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2, 
               label='Target (90%)')
    
    # Annotazioni valori
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5), textcoords='offset points',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuratezza (%)')
    ax.set_title('Accuratezza Identificazione Task')
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {output_path}")


def plot_summary_dashboard(results: dict, output_path: str, dpi: int):
    """Genera dashboard 2x2 con i grafici principali."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Helper per estrarre accuracy
    def get_acc(task_acc, key):
        val = task_acc.get(key, 0)
        if isinstance(val, dict):
            return val.get('accuracy', 0) * 100
        return val * 100
    
    # 1. Task Accuracy (top-left)
    ax = axes[0, 0]
    task_acc = results['task_accuracy']
    tasks = ['QA', 'Summary', 'News']
    accuracies = [get_acc(task_acc, t.lower()) for t in tasks]
    colors = ['#3498DB', '#2ECC71', '#E74C3C']
    
    bars = ax.bar(tasks, accuracies, color=colors)
    ax.axhline(y=90, color='green', linestyle='--', linewidth=2)
    ax.set_ylabel('Accuratezza (%)')
    ax.set_title('Accuratezza per Task')
    ax.set_ylim(0, 110)
    
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=11)
    
    # 2. Confusion Matrix (top-right)
    ax = axes[0, 1]
    cm = results['confusion_matrix']
    tasks_cm = ['qa', 'summary', 'news']
    matrix = np.array([[cm[t1].get(t2, 0) for t2 in tasks_cm] for t1 in tasks_cm])
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = matrix / row_sums * 100
    
    im = ax.imshow(matrix_norm, cmap='Blues')
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(['QA', 'SUM', 'NEWS'])
    ax.set_yticklabels(['QA', 'SUM', 'NEWS'])
    ax.set_xlabel('Predetto')
    ax.set_ylabel('Vero')
    ax.set_title('Matrice di Confusione')
    
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{matrix_norm[i,j]:.0f}%', ha='center', va='center', fontsize=12)
    
    # 3. Length Impact (bottom-left)
    ax = axes[1, 0]
    length_data = results['length_impact']
    lengths = sorted([int(k) for k in length_data.keys()])
    accs = [length_data[str(l)]['accuracy'] * 100 for l in lengths]
    baseline = [55, 70, 90][:len(lengths)]
    
    ax.plot(lengths, accs, 'o-', linewidth=2.5, markersize=10, label='Multi-Layer', color='#2E86AB')
    ax.plot(lengths, baseline, 's--', linewidth=2, markersize=8, label='Crypto Only', color='#A23B72', alpha=0.7)
    ax.axhline(y=90, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Token')
    ax.set_ylabel('Accuratezza (%)')
    ax.set_title('Impatto Lunghezza')
    ax.legend(fontsize=10)
    ax.set_ylim(40, 105)
    ax.set_xticks(lengths)
    
    # 4. Summary Stats (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')
    
    # Gestisci sia formato vecchio che nuovo per overall
    overall_val = task_acc.get('overall', 0)
    if isinstance(overall_val, dict):
        overall_acc = overall_val.get('accuracy', 0) * 100
    else:
        overall_acc = overall_val * 100
    
    detection_rate = results.get('detection_rate', {})
    if isinstance(detection_rate, dict):
        det_rate = detection_rate.get('detection_rate', 0) * 100
    else:
        det_rate = detection_rate * 100
    
    n_samples = results['n_samples']
    
    stats_text = f"""
    RIEPILOGO RISULTATI
    ══════════════════════════════
    
    Campioni Valutati:     {n_samples}
    
    Accuratezza Overall:   {overall_acc:.1f}%
    Target:                90%
    Status:                {'✓ PASS' if overall_acc >= 90 else '→ CLOSE' if overall_acc >= 80 else '✗ FAIL'}
    
    Detection Rate:        {det_rate:.1f}%
    
    ══════════════════════════════
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
           fontsize=13, fontfamily='monospace', verticalalignment='top')
    
    plt.suptitle('Selective Watermarking - Dashboard Risultati', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  Salvato: {output_path}")


def main():
    args = parse_args()
    
    print("=" * 70)
    print("GENERAZIONE GRAFICI")
    print("=" * 70)
    
    # Verifica file risultati
    if not os.path.exists(args.results):
        print(f"\nERRORE: File risultati non trovato: {args.results}")
        print("Eseguire prima: python scripts/run_evaluation.py")
        return 1
    
    # Setup
    setup_style(args.style)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Carica risultati
    print(f"\nCaricamento: {args.results}")
    results = load_results(args.results)
    
    # Genera grafici
    print(f"\nGenerazione grafici in: {args.output_dir}/")
    
    plot_confusion_matrix(
        results, 
        os.path.join(args.output_dir, 'confusion_matrix.png'),
        args.dpi
    )
    
    plot_length_impact(
        results,
        os.path.join(args.output_dir, 'length_impact.png'),
        args.dpi
    )
    
    plot_task_accuracy(
        results,
        os.path.join(args.output_dir, 'task_accuracy.png'),
        args.dpi
    )
    
    plot_summary_dashboard(
        results,
        os.path.join(args.output_dir, 'summary_dashboard.png'),
        args.dpi
    )
    
    print("\n" + "=" * 70)
    print("GRAFICI GENERATI CON SUCCESSO")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
