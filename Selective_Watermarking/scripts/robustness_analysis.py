#!/usr/bin/env python3

import os
import sys
import json
import random
import argparse
import logging
from typing import List, Dict, Tuple, Optional
import csv

# Silenzia logging verbose
logging.getLogger('watermarkers').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from tqdm import tqdm


# DATA LOADING
def load_dataset(dataset_path: str, boundaries_path: str) -> List[Dict]:
    """Carica dataset e boundaries."""
    samples = []
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'id': row.get('id', ''),
                'text': row['text'],
                'task': row['task'],
                'target_length': int(row.get('target_length', 100))
            })
    
    with open(boundaries_path, 'r', encoding='utf-8') as f:
        boundaries_data = json.load(f)
    
    for sample in samples:
        sample_id = sample['id']
        if sample_id in boundaries_data:
            sample['boundaries'] = boundaries_data[sample_id].get('boundaries', [])
            sample['bits'] = boundaries_data[sample_id].get('bits', [])
        else:
            sample['boundaries'] = []
            sample['bits'] = []
    
    return samples


# PERTURBATION FUNCTIONS
def truncate_text_with_boundaries(text: str, boundaries: List[int], bits: List[int], 
                                   keep_ratio: float, tokenizer) -> Tuple[str, List[int], List[int]]:
    """Tronca il testo e adatta i boundaries."""
    if keep_ratio >= 1.0:
        return text, boundaries, bits
    
    tokens = tokenizer.encode(text)
    keep_count = max(1, int(len(tokens) * keep_ratio))
    truncated_tokens = tokens[:keep_count]
    truncated_text = tokenizer.decode(truncated_tokens)
    
    max_bit = keep_count * 16
    new_boundaries = [b for b in boundaries if b < max_bit]
    
    return truncated_text, new_boundaries, bits


def perturb_characters(text: str, noise_ratio: float, seed: int = 42) -> str:
    """Aggiunge rumore ai caratteri."""
    if noise_ratio <= 0:
        return text
    
    random.seed(seed)
    chars = list(text)
    n_changes = int(len(chars) * noise_ratio)
    
    for _ in range(n_changes):
        if len(chars) == 0:
            break
        idx = random.randint(0, len(chars) - 1)
        action = random.choice(['insert', 'delete', 'replace'])
        
        if action == 'insert':
            chars.insert(idx, random.choice('abcdefghijklmnopqrstuvwxyz '))
        elif action == 'delete' and len(chars) > 1:
            chars.pop(idx)
        else:
            chars[idx] = random.choice('abcdefghijklmnopqrstuvwxyz ')
    
    return ''.join(chars)


def paraphrase_text(text: str, model, tokenizer, device) -> str:
    """Riscrive il testo usando T5 paraphraser."""
    import torch
    
    # Limita lunghezza input
    words = text.split()
    if len(words) > 80:
        text = ' '.join(words[:80])
    
    input_text = f"paraphrase: {text}"
    
    try:
        encoding = tokenizer(
            input_text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=256,
                do_sample=True,
                temperature=1.2,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result if len(result) > 10 else text
        
    except Exception as e:
        return text


# DETECTION
def detect_best_task(text: str, watermarker, tokenizer,
                     boundaries: List[int] = None, bits: List[int] = None) -> Tuple[str, float]:
    """Trova il task con confidence più alta."""
    best_task = None
    best_conf = 0.0
    
    for task in ['qa', 'summary', 'news']:
        try:
            result = watermarker.detect(
                text=text,
                task=task,
                tokenizer=tokenizer,
                crypto_boundaries=boundaries,
                crypto_bits=bits
            )
            if result.confidence > best_conf:
                best_conf = result.confidence
                best_task = task
        except:
            pass
    
    return best_task, best_conf


# ANALYSIS FUNCTIONS
def run_truncation_analysis(samples: List[Dict], watermarker, tokenizer) -> Dict:
    """Analisi robustezza a truncation."""
    print("\n" + "=" * 60)
    print("📏 TRUNCATION ANALYSIS")
    print("=" * 60)
    print("Testing watermark robustness when text is truncated...\n")
    
    ratios = [1.0, 0.75, 0.50, 0.25]
    results = {r: {'correct': 0, 'detected': 0, 'total': 0} for r in ratios}
    
    for sample in tqdm(samples, desc="Analyzing", ncols=70):
        if not sample['boundaries']:
            continue
        
        for ratio in ratios:
            truncated, new_bounds, bits = truncate_text_with_boundaries(
                sample['text'], sample['boundaries'], sample['bits'], ratio, tokenizer
            )
            
            pred_task, conf = detect_best_task(
                truncated, watermarker, tokenizer, 
                new_bounds if new_bounds else None, 
                bits
            )
            
            results[ratio]['total'] += 1
            if conf > 0.4:
                results[ratio]['detected'] += 1
                if pred_task == sample['task']:
                    results[ratio]['correct'] += 1
    
    for r in ratios:
        res = results[r]
        res['detection_rate'] = res['detected'] / res['total'] if res['total'] > 0 else 0
        res['accuracy'] = res['correct'] / res['total'] if res['total'] > 0 else 0
    
    print("\n" + "-" * 50)
    print(f"{'Retention':<12} {'Detection Rate':<18} {'Task Accuracy'}")
    print("-" * 50)
    for r in ratios:
        print(f"{r:<12.0%} {results[r]['detection_rate']:<18.1%} {results[r]['accuracy']:.1%}")
    print("-" * 50)
    
    return results


def run_perturbation_analysis(samples: List[Dict], watermarker, tokenizer) -> Dict:
    """Analisi robustezza a character perturbation."""
    print("\n" + "=" * 60)
    print("🔤 CHARACTER PERTURBATION ANALYSIS")
    print("=" * 60)
    print("Testing watermark robustness against character noise...\n")
    
    noise_levels = [0.0, 0.01, 0.05, 0.10, 0.20]
    results = {n: {'correct': 0, 'detected': 0, 'total': 0} for n in noise_levels}
    
    for sample in tqdm(samples, desc="Analyzing", ncols=70):
        if not sample['boundaries']:
            continue
        
        for noise in noise_levels:
            perturbed = perturb_characters(sample['text'], noise)
            
            pred_task, conf = detect_best_task(
                perturbed, watermarker, tokenizer,
                sample['boundaries'], sample['bits']
            )
            
            results[noise]['total'] += 1
            if conf > 0.4:
                results[noise]['detected'] += 1
                if pred_task == sample['task']:
                    results[noise]['correct'] += 1
    
    for n in noise_levels:
        res = results[n]
        res['detection_rate'] = res['detected'] / res['total'] if res['total'] > 0 else 0
        res['accuracy'] = res['correct'] / res['total'] if res['total'] > 0 else 0
    
    print("\n" + "-" * 50)
    print(f"{'Noise Level':<12} {'Detection Rate':<18} {'Task Accuracy'}")
    print("-" * 50)
    for n in noise_levels:
        print(f"{n:<12.0%} {results[n]['detection_rate']:<18.1%} {results[n]['accuracy']:.1%}")
    print("-" * 50)
    
    return results


def run_paraphrase_analysis(samples: List[Dict], watermarker, tokenizer, 
                            max_samples: int = 450) -> Dict:
    """Analisi robustezza a paraphrasing."""
    print("\n" + "=" * 60)
    print("📝 PARAPHRASE ANALYSIS")
    print("=" * 60)
    
    print("Loading T5 paraphrase model...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "humarin/chatgpt_paraphraser_on_T5_base"
        t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t5_model = t5_model.to(device)
        t5_model.eval()
        
        print(f"   ✓ Model loaded on {device.type.upper()}")
        if device.type == "cuda":
            print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        print("   Install: pip install transformers sentencepiece")
        return None
    
    print(f"\nTesting on {min(max_samples, len(samples))} samples...")
    print("Note: Paraphrasing rewrites text completely, destroying watermarks.\n")
    
    test_samples = [s for s in samples if s['boundaries']][:max_samples]
    
    results = {
        'original': {'correct': 0, 'detected': 0, 'total': 0},
        'paraphrased': {'correct': 0, 'detected': 0, 'total': 0}
    }
    examples = []
    
    for sample in tqdm(test_samples, desc="Paraphrasing", ncols=70):
        # Original (con boundaries)
        pred_orig, conf_orig = detect_best_task(
            sample['text'], watermarker, tokenizer,
            sample['boundaries'], sample['bits']
        )
        results['original']['total'] += 1
        if conf_orig > 0.4:
            results['original']['detected'] += 1
            if pred_orig == sample['task']:
                results['original']['correct'] += 1
        
        # Paraphrased (SENZA pre-processing, testo originale con watermark)
        paraphrased = paraphrase_text(sample['text'], t5_model, t5_tokenizer, device)
        
        # Detection senza boundaries (il testo è completamente nuovo)
        pred_para, conf_para = detect_best_task(
            paraphrased, watermarker, tokenizer, None, None
        )
        results['paraphrased']['total'] += 1
        if conf_para > 0.4:
            results['paraphrased']['detected'] += 1
            if pred_para == sample['task']:
                results['paraphrased']['correct'] += 1
        
        # Salva esempi
        if len(examples) < 3:
            examples.append({
                'task': sample['task'],
                'original': sample['text'][:100],
                'paraphrased': paraphrased[:100],
                'conf_orig': conf_orig,
                'conf_para': conf_para
            })
    
    for key in ['original', 'paraphrased']:
        res = results[key]
        res['detection_rate'] = res['detected'] / res['total'] if res['total'] > 0 else 0
        res['accuracy'] = res['correct'] / res['total'] if res['total'] > 0 else 0
    
    print("\n" + "-" * 50)
    print(f"{'Condition':<15} {'Detection Rate':<18} {'Task Accuracy'}")
    print("-" * 50)
    print(f"{'Original':<15} {results['original']['detection_rate']:<18.1%} {results['original']['accuracy']:.1%}")
    print(f"{'Paraphrased':<15} {results['paraphrased']['detection_rate']:<18.1%} {results['paraphrased']['accuracy']:.1%}")
    print("-" * 50)
    
    print("\n📋 Examples:")
    for i, ex in enumerate(examples, 1):
        print(f"\n  [{i}] Task: {ex['task'].upper()}")
        print(f"      Original ({ex['conf_orig']:.0%}): {ex['original']}...")
        print(f"      Paraphrased ({ex['conf_para']:.0%}): {ex['paraphrased']}...")
    
    results['examples'] = examples
    return results


# PLOTTING
def create_plots(trunc_res: Optional[Dict], pert_res: Optional[Dict], 
                 para_res: Optional[Dict], output_dir: str) -> List[str]:
    """Crea grafici."""
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        pass
    
    plots_created = []
    
    # Truncation
    if trunc_res:
        fig, ax = plt.subplots(figsize=(10, 6))
        ratios = list(trunc_res.keys())
        det_rates = [trunc_res[r]['detection_rate'] * 100 for r in ratios]
        accuracies = [trunc_res[r]['accuracy'] * 100 for r in ratios]
        
        x = np.arange(len(ratios))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, det_rates, width, label='Detection Rate', color='#00d4ff')
        bars2 = ax.bar(x + width/2, accuracies, width, label='Task Accuracy', color='#00ff88')
        
        ax.set_xlabel('Text Retention Ratio', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Robustness to Truncation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{r:.0%}' for r in ratios])
        ax.legend()
        ax.set_ylim(0, 105)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/robustness_truncation.png', dpi=150)
        plt.close()
        plots_created.append('robustness_truncation.png')
    
    # Perturbation
    if pert_res:
        fig, ax = plt.subplots(figsize=(10, 6))
        levels = list(pert_res.keys())
        det_rates = [pert_res[l]['detection_rate'] * 100 for l in levels]
        accuracies = [pert_res[l]['accuracy'] * 100 for l in levels]
        
        x = np.arange(len(levels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, det_rates, width, label='Detection Rate', color='#ff6b6b')
        bars2 = ax.bar(x + width/2, accuracies, width, label='Task Accuracy', color='#ffd93d')
        
        ax.set_xlabel('Noise Level', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Robustness to Character Perturbation', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{l:.0%}' for l in levels])
        ax.legend()
        ax.set_ylim(0, 105)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/robustness_perturbation.png', dpi=150)
        plt.close()
        plots_created.append('robustness_perturbation.png')
    
    # Paraphrase
    if para_res:
        fig, ax = plt.subplots(figsize=(8, 6))
        conditions = ['Original', 'Paraphrased']
        det_rates = [para_res['original']['detection_rate'] * 100,
                     para_res['paraphrased']['detection_rate'] * 100]
        accuracies = [para_res['original']['accuracy'] * 100,
                      para_res['paraphrased']['accuracy'] * 100]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, det_rates, width, label='Detection Rate', color='#a78bfa')
        bars2 = ax.bar(x + width/2, accuracies, width, label='Task Accuracy', color='#f472b6')
        
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Robustness to Paraphrasing', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.legend()
        ax.set_ylim(0, 105)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/robustness_paraphrase.png', dpi=150)
        plt.close()
        plots_created.append('robustness_paraphrase.png')
    
    # Summary (solo se abbiamo tutti e tre)
    if trunc_res and pert_res and para_res:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Truncation
        ratios = list(trunc_res.keys())
        axes[0].plot([r*100 for r in ratios], 
                     [trunc_res[r]['accuracy']*100 for r in ratios],
                     'o-', color='#00d4ff', linewidth=2, markersize=8)
        axes[0].fill_between([r*100 for r in ratios],
                             [trunc_res[r]['accuracy']*100 for r in ratios],
                             alpha=0.3, color='#00d4ff')
        axes[0].set_xlabel('Text Retained (%)')
        axes[0].set_ylabel('Task Accuracy (%)')
        axes[0].set_title('Truncation', fontweight='bold')
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, alpha=0.3)
        
        # Perturbation
        levels = list(pert_res.keys())
        axes[1].plot([l*100 for l in levels],
                     [pert_res[l]['accuracy']*100 for l in levels],
                     'o-', color='#ff6b6b', linewidth=2, markersize=8)
        axes[1].fill_between([l*100 for l in levels],
                             [pert_res[l]['accuracy']*100 for l in levels],
                             alpha=0.3, color='#ff6b6b')
        axes[1].set_xlabel('Noise Level (%)')
        axes[1].set_ylabel('Task Accuracy (%)')
        axes[1].set_title('Character Perturbation', fontweight='bold')
        axes[1].set_ylim(0, 105)
        axes[1].grid(True, alpha=0.3)
        
        # Paraphrase
        conds = ['Original', 'Paraphrased']
        accs = [para_res['original']['accuracy']*100, para_res['paraphrased']['accuracy']*100]
        bars = axes[2].bar(conds, accs, color=['#a78bfa', '#f472b6'])
        axes[2].set_ylabel('Task Accuracy (%)')
        axes[2].set_title('Paraphrasing', fontweight='bold')
        axes[2].set_ylim(0, 105)
        for bar in bars:
            h = bar.get_height()
            axes[2].annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')
        
        plt.suptitle('Robustness Analysis Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/robustness_summary.png', dpi=150, bbox_inches='tight')
        plt.close()
        plots_created.append('robustness_summary.png')
    
    return plots_created


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description='Robustness Analysis for Selective Watermarking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python robustness_analysis.py                     # Run all analyses
  python robustness_analysis.py --only-truncation   # Run only truncation
  python robustness_analysis.py --only-perturbation # Run only perturbation  
  python robustness_analysis.py --only-paraphrase   # Run only paraphrasing
        """
    )
    parser.add_argument('--dataset', type=str, default='data/generated/dataset.csv')
    parser.add_argument('--boundaries', type=str, default='data/generated/boundaries.json')
    parser.add_argument('--output', type=str, default='data/results')
    parser.add_argument('--only-truncation', action='store_true', help='Run only truncation analysis')
    parser.add_argument('--only-perturbation', action='store_true', help='Run only perturbation analysis')
    parser.add_argument('--only-paraphrase', action='store_true', help='Run only paraphrase analysis')
    parser.add_argument('--paraphrase-samples', type=int, default=450, help='Max samples for paraphrasing')
    
    args = parser.parse_args()
    
    # Determina quali analisi eseguire
    run_all = not (args.only_truncation or args.only_perturbation or args.only_paraphrase)
    do_truncation = run_all or args.only_truncation
    do_perturbation = run_all or args.only_perturbation
    do_paraphrase = run_all or args.only_paraphrase
    
    # Header
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " ROBUSTNESS ANALYSIS - Selective Watermarking ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Carica dati
    print(f"\n📂 Loading dataset...")
    samples = load_dataset(args.dataset, args.boundaries)
    samples_valid = [s for s in samples if s['boundaries']]
    print(f"   ✓ {len(samples_valid)} samples with boundaries")
    
    # Carica watermarker
    print("\n🔧 Loading watermarker...")
    from watermarkers.multilayer import MultiLayerWatermarker
    from transformers import AutoTokenizer
    
    watermarker = MultiLayerWatermarker()
    tokenizer = AutoTokenizer.from_pretrained('gpt2-medium')
    print("   ✓ Ready")
    
    # Esegui analisi
    trunc_res = None
    pert_res = None
    para_res = None
    
    if do_truncation:
        trunc_res = run_truncation_analysis(samples_valid, watermarker, tokenizer)
    
    if do_perturbation:
        pert_res = run_perturbation_analysis(samples_valid, watermarker, tokenizer)
    
    if do_paraphrase:
        para_res = run_paraphrase_analysis(samples_valid, watermarker, tokenizer, args.paraphrase_samples)
    
    # Salva risultati
    os.makedirs(args.output, exist_ok=True)
    results = {}
    if trunc_res:
        results['truncation'] = {str(k): v for k, v in trunc_res.items()}
    if pert_res:
        results['perturbation'] = {str(k): v for k, v in pert_res.items()}
    if para_res:
        para_save = {k: v for k, v in para_res.items() if k != 'examples'}
        results['paraphrase'] = para_save
    
    output_file = os.path.join(args.output, 'robustness_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Crea grafici
    print("\n📊 Creating plots...")
    plots = create_plots(trunc_res, pert_res, para_res, 'figures')
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\n📁 Results: {output_file}")
    if plots:
        print(f"📈 Plots:")
        for p in plots:
            print(f"   - figures/{p}")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
