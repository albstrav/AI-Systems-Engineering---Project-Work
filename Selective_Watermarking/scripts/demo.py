#!/usr/bin/env python3

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gradio as gr

# Variabile globale per il generatore
_generator = None

def get_generator():
    """Carica il generatore (lazy loading)."""
    global _generator
    if _generator is None:
        from generation.generator import SelectiveWatermarkGenerator
        print("Caricamento modello...")
        _generator = SelectiveWatermarkGenerator(model_name='gpt2-medium')
        print("Modello caricato!")
    return _generator


def generate_and_detect(prompt: str, task: str, max_tokens: int):
    """Genera testo con watermark e mostra detection comparativa."""
    try:
        gen = get_generator()
        
        # Genera testo con watermark
        result = gen.generate(
            prompt=prompt,
            task=task.lower(),
            max_tokens=int(max_tokens),
            use_watermark=True
        )
        
        # Detection per tutti i task
        detection_results = {}
        for test_task in ['qa', 'summary', 'news']:
            det = gen.watermarker.detect(
                text=result.text,
                task=test_task,
                tokenizer=gen.tokenizer,
                crypto_boundaries=result.crypto_boundaries,
                crypto_bits=result.crypto_bits
            )
            detection_results[test_task] = det
        
        # Prepara output con prompt visibile
        text_output = f"[PROMPT]: {prompt}\n\n[GENERATED TEXT]:\n{result.text}"
        
        # Statistiche
        techniques = ", ".join(result.techniques_applied)
        n_blocks = len(result.crypto_boundaries) if result.crypto_boundaries else 0
        entropy = result.stats.get('crypto', {}).get('empirical_entropy', 0)
        
        # Token: modello vs post-watermark
        total_bits = result.stats.get('crypto', {}).get('total_bits', 0)
        tokens_model = total_bits // 16 if total_bits > 0 else result.tokens_generated
        tokens_post = result.tokens_generated
        
        # Sostituzioni
        syn_subs = result.stats.get('synonym', {}).get('substitutions_made', 0)
        char_subs = result.stats.get('character', {}).get('substituted_chars', 0)
        
        # Estrai contributi REALI dalla detection del task corretto
        correct_task = task.lower()
        det_correct = detection_results[correct_task]
        tech_results = det_correct.technique_results
        
        # Contributi reali
        crypto_conf = tech_results.get('crypto', {}).get('confidence', 0) * 100
        synonym_conf = tech_results.get('synonym', {}).get('confidence', 0) * 100
        char_conf = tech_results.get('character', {}).get('confidence', 0) * 100
        
        # Match ratio reali
        crypto_detected = tech_results.get('crypto', {}).get('detected', False)
        synonym_match = tech_results.get('synonym', {}).get('match_ratio', 0) * 100
        char_match = tech_results.get('character', {}).get('match_ratio', 0) * 100
        
        stats_text = f"""
<div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 20px 0;">
    <div style="text-align: center; padding: 12px 18px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; min-width: 90px; border: 1px solid #2d3748;">
        <div style="font-size: 1.5em; font-weight: bold; color: #00d4ff;">{result.task.upper()}</div>
        <div style="color: #888; font-size: 0.8em;">Task</div>
    </div>
    <div style="text-align: center; padding: 12px 18px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; min-width: 110px; border: 1px solid #2d3748;">
        <div style="font-size: 1.5em; font-weight: bold; color: #00ff88;">{tokens_model} <span style="color: #666; font-size: 0.6em;">→</span> <span style="color: #a78bfa;">{tokens_post}</span></div>
        <div style="color: #888; font-size: 0.8em;">Token (model→post)</div>
    </div>
    <div style="text-align: center; padding: 12px 18px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; min-width: 90px; border: 1px solid #2d3748;">
        <div style="font-size: 1.5em; font-weight: bold; color: #ff6b6b;">{result.generation_time:.1f}s</div>
        <div style="color: #888; font-size: 0.8em;">Tempo</div>
    </div>
    <div style="text-align: center; padding: 12px 18px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; min-width: 90px; border: 1px solid #2d3748;">
        <div style="font-size: 1.5em; font-weight: bold; color: #ffd93d;">{n_blocks}</div>
        <div style="color: #888; font-size: 0.8em;">Blocchi</div>
    </div>
    <div style="text-align: center; padding: 12px 18px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; min-width: 90px; border: 1px solid #2d3748;">
        <div style="font-size: 1.5em; font-weight: bold; color: #a78bfa;">{entropy:.0f}</div>
        <div style="color: #888; font-size: 0.8em;">Entropia</div>
    </div>
    <div style="text-align: center; padding: 12px 18px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; min-width: 90px; border: 1px solid #2d3748;">
        <div style="font-size: 1.5em; font-weight: bold; color: #ffd93d;">{syn_subs}</div>
        <div style="color: #888; font-size: 0.8em;">Syn. Subs</div>
    </div>
    <div style="text-align: center; padding: 12px 18px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; min-width: 90px; border: 1px solid #2d3748;">
        <div style="font-size: 1.5em; font-weight: bold; color: #a78bfa;">{char_subs}</div>
        <div style="color: #888; font-size: 0.8em;">Char. Subs</div>
    </div>
</div>

<div style="text-align: center; margin: 15px 0;">
    <span style="background: #2d3748; padding: 8px 16px; border-radius: 20px; color: #a0aec0;">
        🔧 Tecniche: <strong style="color: #fff;">{techniques}</strong>
    </span>
</div>
"""
        
        # Detection results
        qa_conf = detection_results['qa'].confidence * 100
        sum_conf = detection_results['summary'].confidence * 100
        news_conf = detection_results['news'].confidence * 100
        total_conf = qa_conf + sum_conf + news_conf
        
        # Percentuali per donut chart
        if total_conf > 0:
            qa_pct = (qa_conf / total_conf) * 100
            sum_pct = (sum_conf / total_conf) * 100
            news_pct = (news_conf / total_conf) * 100
        else:
            qa_pct = sum_pct = news_pct = 33.3
        
        # Best match e gap
        best_task = max(detection_results.keys(), key=lambda t: detection_results[t].confidence)
        best_conf = detection_results[best_task].confidence * 100
        confs_sorted = sorted([qa_conf, sum_conf, news_conf], reverse=True)
        gap = confs_sorted[0] - confs_sorted[1]
        margin = confs_sorted[0] - confs_sorted[2]
        
        # Indicatore affidabilità
        if best_conf >= 80 and gap >= 30 and n_blocks >= 3:
            reliability = ("ALTA", "#00ff88", "Il sistema è molto sicuro di questa identificazione")
        elif best_conf >= 50 and gap >= 15 and n_blocks >= 2:
            reliability = ("MEDIA", "#ffd93d", "Identificazione ragionevolmente affidabile")
        else:
            reliability = ("BASSA", "#ff6b6b", "Risultato incerto, potrebbero servire più token")
        
        # Colori per le barre
        def get_bar_color(task_name, is_best):
            if is_best and task_name == correct_task:
                return "#00ff88"
            elif is_best:
                return "#ff6b6b"
            else:
                return "#4a5568"
        
        # Calcolo gradi per donut chart
        qa_deg = (qa_pct / 100) * 360
        sum_deg = (sum_pct / 100) * 360
        
        detection_text = f"""
<div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap; align-items: flex-start;">
    
    <!-- Barre di confidence -->
    <div style="flex: 1; min-width: 300px; max-width: 450px;">
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold; color: {'#00ff88' if correct_task == 'qa' else '#fff'};">QA {'✓' if correct_task == 'qa' else ''}</span>
                <span style="color: #00d4ff;">{qa_conf:.1f}%</span>
            </div>
            <div style="background: #2d3748; border-radius: 10px; height: 24px; overflow: hidden;">
                <div style="background: {get_bar_color('qa', best_task == 'qa')}; width: {qa_conf}%; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
            </div>
        </div>
        
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold; color: {'#00ff88' if correct_task == 'summary' else '#fff'};">SUMMARY {'✓' if correct_task == 'summary' else ''}</span>
                <span style="color: #00d4ff;">{sum_conf:.1f}%</span>
            </div>
            <div style="background: #2d3748; border-radius: 10px; height: 24px; overflow: hidden;">
                <div style="background: {get_bar_color('summary', best_task == 'summary')}; width: {sum_conf}%; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
            </div>
        </div>
        
        <div style="margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: bold; color: {'#00ff88' if correct_task == 'news' else '#fff'};">NEWS {'✓' if correct_task == 'news' else ''}</span>
                <span style="color: #00d4ff;">{news_conf:.1f}%</span>
            </div>
            <div style="background: #2d3748; border-radius: 10px; height: 24px; overflow: hidden;">
                <div style="background: {get_bar_color('news', best_task == 'news')}; width: {news_conf}%; height: 100%; border-radius: 10px; transition: width 0.5s;"></div>
            </div>
        </div>
    </div>
    
    <!-- Donut Chart -->
    <div style="text-align: center; min-width: 200px;">
        <div style="
            width: 150px; 
            height: 150px; 
            border-radius: 50%; 
            background: conic-gradient(
                #00d4ff 0deg {qa_deg}deg, 
                #ffd93d {qa_deg}deg {qa_deg + sum_deg}deg, 
                #ff6b6b {qa_deg + sum_deg}deg 360deg
            );
            margin: 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div style="width: 90px; height: 90px; border-radius: 50%; background: #1a1a2e; display: flex; align-items: center; justify-content: center; flex-direction: column;">
                <div style="font-size: 1.4em; font-weight: bold; color: #fff;">{best_conf:.0f}%</div>
                <div style="font-size: 0.7em; color: #888;">Best</div>
            </div>
        </div>
        <div style="margin-top: 15px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
            <span style="color: #00d4ff; font-size: 0.85em;">● QA {qa_pct:.0f}%</span>
            <span style="color: #ffd93d; font-size: 0.85em;">● SUM {sum_pct:.0f}%</span>
            <span style="color: #ff6b6b; font-size: 0.85em;">● NEWS {news_pct:.0f}%</span>
        </div>
    </div>
</div>

<!-- Contributi REALI per tecnica -->
<div style="margin: 25px auto; max-width: 700px; background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #2d3748;">
    <div style="text-align: center; color: #888; font-size: 0.9em; margin-bottom: 15px;">📊 Detection Results per Technique (Task: {correct_task.upper()})</div>
    <div style="display: flex; justify-content: center; gap: 25px; flex-wrap: wrap;">
        <div style="text-align: center; min-width: 140px; padding: 10px; background: #2d3748; border-radius: 10px;">
            <div style="font-size: 1.5em; font-weight: bold; color: {'#00ff88' if crypto_detected else '#ff6b6b'};">{crypto_conf:.1f}%</div>
            <div style="font-size: 0.85em; color: #00ff88; font-weight: bold;">Crypto</div>
            <div style="font-size: 0.75em; color: #888; margin-top: 3px;">{'✓ Detected' if crypto_detected else '✗ Not detected'}</div>
        </div>
        <div style="text-align: center; min-width: 140px; padding: 10px; background: #2d3748; border-radius: 10px;">
            <div style="font-size: 1.5em; font-weight: bold; color: #ffd93d;">{synonym_conf:.1f}%</div>
            <div style="font-size: 0.85em; color: #ffd93d; font-weight: bold;">Synonym</div>
            <div style="font-size: 0.75em; color: #888; margin-top: 3px;">Match: {synonym_match:.1f}%</div>
        </div>
        <div style="text-align: center; min-width: 140px; padding: 10px; background: #2d3748; border-radius: 10px;">
            <div style="font-size: 1.5em; font-weight: bold; color: #a78bfa;">{char_conf:.1f}%</div>
            <div style="font-size: 0.85em; color: #a78bfa; font-weight: bold;">Character</div>
            <div style="font-size: 0.75em; color: #888; margin-top: 3px;">Match: {char_match:.1f}%</div>
        </div>
    </div>
</div>

<!-- Indicatore Affidabilità -->
<div style="text-align: center; margin: 20px 0;">
    <div style="display: inline-block; background: #1a1a2e; border-radius: 25px; padding: 12px 25px; border: 2px solid {reliability[1]};">
        <span style="color: {reliability[1]}; font-weight: bold; font-size: 1.1em;">🎯 Affidabilità: {reliability[0]}</span>
        <span style="color: #888; margin-left: 10px; font-size: 0.9em;">- {reliability[2]}</span>
    </div>
</div>
"""
        
        # Verdetto finale
        if best_task == correct_task:
            verdict = f"""
<div style="text-align: center; margin: 25px auto; padding: 25px; background: linear-gradient(135deg, #064e3b, #065f46); border-radius: 15px; border: 2px solid #00ff88; max-width: 600px;">
    <div style="font-size: 1.6em; font-weight: bold; color: #00ff88; margin-bottom: 10px;">
        ✅ IDENTIFICAZIONE CORRETTA
    </div>
    <div style="font-size: 1.05em; color: #d1fae5;">
        Task <strong>{best_task.upper()}</strong> identificato con confidence <strong>{best_conf:.1f}%</strong>
    </div>
    <div style="margin-top: 15px; display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
        <span style="background: #047857; padding: 8px 16px; border-radius: 20px; color: #fff; font-size: 0.9em;">
            📊 Gap: <strong>+{gap:.1f}</strong> punti
        </span>
        <span style="background: #047857; padding: 8px 16px; border-radius: 20px; color: #fff; font-size: 0.9em;">
            📏 Margine: <strong>+{margin:.1f}</strong> punti
        </span>
    </div>
</div>
"""
        else:
            verdict = f"""
<div style="text-align: center; margin: 25px auto; padding: 25px; background: linear-gradient(135deg, #7f1d1d, #991b1b); border-radius: 15px; border: 2px solid #ff6b6b; max-width: 600px;">
    <div style="font-size: 1.6em; font-weight: bold; color: #ff6b6b; margin-bottom: 10px;">
        ❌ IDENTIFICAZIONE ERRATA
    </div>
    <div style="font-size: 1.05em; color: #fecaca;">
        Identificato <strong>{best_task.upper()}</strong> invece di <strong>{correct_task.upper()}</strong>
    </div>
</div>
"""
        
        return text_output, stats_text, detection_text + verdict
        
    except Exception as e:
        import traceback
        error_msg = f"Errore: {str(e)}\n\n{traceback.format_exc()}"
        return error_msg, "", ""


# CSS personalizzato
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.main-title {
    text-align: center !important;
    font-size: 2.8em !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #00d4ff, #00ff88, #ffd93d);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 25px 0 35px 0 !important;
    padding: 0 !important;
}

.center-content {
    text-align: center !important;
}

.generate-btn {
    background: linear-gradient(135deg, #00d4ff, #00ff88) !important;
    border: none !important;
    font-size: 1.1em !important;
    padding: 12px 35px !important;
    border-radius: 25px !important;
    color: #000 !important;
    font-weight: bold !important;
}

.footer-text {
    text-align: center !important;
    color: #666 !important;
    padding: 25px !important;
    font-size: 0.95em !important;
}
"""

def create_demo():
    """Crea e restituisce l'interfaccia Gradio."""
    
    with gr.Blocks(title="Selective Watermarking Demo", css=custom_css) as demo:
        
        # Titolo centrato
        gr.HTML("""
        <h1 class="main-title">🔐 Selective Watermarking System</h1>
        """)
        
        # Contenuto principale
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configurazione", elem_classes=["center-content"])
                
                task_select = gr.Dropdown(
                    choices=["QA", "Summary", "News"],
                    value="QA",
                    label="🎯 Task",
                    info="Seleziona il task per cui generare il testo watermarked"
                )
                
                prompt_input = gr.Textbox(
                    label="📝 Prompt",
                    placeholder="Inserisci il prompt di partenza...",
                    value="Artificial intelligence is",
                    lines=3
                )
                
                tokens_slider = gr.Slider(
                    minimum=50,
                    maximum=200,
                    value=100,
                    step=10,
                    label="📏 Max Token"
                )
                
                generate_btn = gr.Button(
                    "🚀 Genera e Rileva Watermark", 
                    variant="primary",
                    elem_classes=["generate-btn"]
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### 📄 Testo Generato", elem_classes=["center-content"])
                output_text = gr.Textbox(
                    label="",
                    lines=10,
                    placeholder="Il testo watermarked apparirà qui dopo la generazione..."
                )
        
        # Statistiche
        gr.Markdown("<br>")
        gr.Markdown("### 📊 Statistiche Generazione", elem_classes=["center-content"])
        stats_output = gr.HTML()
        
        # Detection
        gr.Markdown("---")
        gr.Markdown("### 🔍 Risultati Detection", elem_classes=["center-content"])
        detection_output = gr.HTML()
        
        # Eventi
        generate_btn.click(
            fn=generate_and_detect,
            inputs=[prompt_input, task_select, tokens_slider],
            outputs=[output_text, stats_output, detection_output]
        )
        
        # Separatore
        gr.Markdown("<br>")
        gr.Markdown("---")
        
        # System Information in fondo
        with gr.Accordion("ℹ️ System Information", open=False):
            gr.HTML("""
<div style="padding: 20px; max-width: 950px; margin: 0 auto;">
    
    <h3 style="color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px;">🎯 System Overview</h3>
    <p style="color: #ccc; line-height: 1.8;">
        This system implements a <strong>task-selective watermarking scheme</strong> that embeds different watermark signatures 
        based on the intended use case (QA, Summary, or News generation). This allows identifying not only 
        <em>if</em> a text was generated by the system, but also <em>for which purpose</em> it was generated.
    </p>
    
    <h3 style="color: #00ff88; border-bottom: 2px solid #00ff88; padding-bottom: 10px; margin-top: 30px;">🔧 Watermarking Techniques</h3>
    <table style="width: 100%; border-collapse: collapse; margin: 15px 0;">
        <thead>
            <tr style="background: #2d3748;">
                <th style="padding: 12px; text-align: left; color: #00d4ff;">Task</th>
                <th style="padding: 12px; text-align: left; color: #00d4ff;">Techniques</th>
                <th style="padding: 12px; text-align: center; color: #00d4ff;">Secret Key</th>
                <th style="padding: 12px; text-align: left; color: #00d4ff;">Description</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #2d3748;">
                <td style="padding: 12px; color: #fff; font-weight: bold;">QA</td>
                <td style="padding: 12px; color: #00ff88;">Crypto + Character</td>
                <td style="padding: 12px; text-align: center; color: #a78bfa; font-family: monospace;">π (314159265)</td>
                <td style="padding: 12px; color: #ccc;">Bit-level cryptographic watermark + Unicode lookalike substitution</td>
            </tr>
            <tr style="border-bottom: 1px solid #2d3748;">
                <td style="padding: 12px; color: #fff; font-weight: bold;">Summary</td>
                <td style="padding: 12px; color: #ffd93d;">Crypto + Synonym</td>
                <td style="padding: 12px; text-align: center; color: #a78bfa; font-family: monospace;">e (271828182)</td>
                <td style="padding: 12px; color: #ccc;">Bit-level cryptographic watermark + semantic synonym substitution</td>
            </tr>
            <tr>
                <td style="padding: 12px; color: #fff; font-weight: bold;">News</td>
                <td style="padding: 12px; color: #ff6b6b;">Crypto + Synonym + Character</td>
                <td style="padding: 12px; text-align: center; color: #a78bfa; font-family: monospace;">φ (161803398)</td>
                <td style="padding: 12px; color: #ccc;">Full multilayer approach with all three techniques</td>
            </tr>
        </tbody>
    </table>
    
    <h3 style="color: #ffd93d; border-bottom: 2px solid #ffd93d; padding-bottom: 10px; margin-top: 30px;">🏗️ System Architecture</h3>
    <div style="background: #1a1a2e; border-radius: 15px; padding: 25px; margin: 15px 0; border: 1px solid #2d3748;">
        
        <!-- Generation Flow -->
        <div style="text-align: center; margin-bottom: 20px; color: #888; font-size: 0.9em;">⬇️ GENERATION FLOW</div>
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                <div style="background: #2d3748; padding: 12px 20px; border-radius: 8px; color: #00d4ff; font-weight: bold;">📝 Input Prompt</div>
                <span style="color: #666; font-size: 1.5em;">→</span>
                <div style="background: #2d3748; padding: 12px 20px; border-radius: 8px; color: #00ff88; font-weight: bold;">🔑 Task Selection</div>
                <span style="color: #666; font-size: 1.5em;">→</span>
                <div style="background: #2d3748; padding: 12px 20px; border-radius: 8px; color: #ffd93d; font-weight: bold;">🤖 GPT-2 Medium</div>
            </div>
            <span style="color: #666; font-size: 1.5em;">↓</span>
            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                <div style="background: linear-gradient(135deg, #064e3b, #065f46); padding: 12px 20px; border-radius: 8px; color: #00ff88; font-weight: bold; border: 1px solid #00ff88;">
                    🔐 Crypto Watermark<br><small style="font-weight: normal;">(Bit-level PRF)</small>
                </div>
                <span style="color: #666; font-size: 1.5em;">→</span>
                <div style="background: linear-gradient(135deg, #064e3b, #065f46); padding: 12px 20px; border-radius: 8px; color: #ffd93d; font-weight: bold; border: 1px solid #ffd93d;">
                    ✏️ Post-Processing<br><small style="font-weight: normal;">(Synonym/Character)</small>
                </div>
            </div>
            <span style="color: #666; font-size: 1.5em;">↓</span>
            <div style="display: flex; align-items: center; gap: 30px; flex-wrap: wrap; justify-content: center;">
                <div style="background: #7f1d1d; padding: 12px 20px; border-radius: 8px; color: #ff6b6b; font-weight: bold; border: 1px solid #ff6b6b;">
                    📄 Watermarked Text
                </div>
                <div style="background: #2d3748; padding: 12px 20px; border-radius: 8px; color: #00d4ff; font-weight: bold;">
                    📍 Block Boundaries
                </div>
            </div>
        </div>
        
        <!-- Separator -->
        <div style="border-top: 1px dashed #4a5568; margin: 25px 0;"></div>
        
        <!-- Detection Flow -->
        <div style="text-align: center; margin-bottom: 20px; color: #888; font-size: 0.9em;">⬆️ DETECTION FLOW</div>
        <div style="display: flex; flex-direction: column; align-items: center; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap; justify-content: center;">
                <div style="background: #7f1d1d; padding: 12px 20px; border-radius: 8px; color: #ff6b6b; font-weight: bold; border: 1px solid #ff6b6b;">
                    📄 Input Text
                </div>
                <span style="color: #666; font-size: 1.5em;">→</span>
                <div style="background: #2d3748; padding: 12px 20px; border-radius: 8px; color: #a78bfa; font-weight: bold;">
                    🔍 Multi-Task Detection<br><small style="font-weight: normal;">(Test all 3 keys)</small>
                </div>
                <span style="color: #666; font-size: 1.5em;">→</span>
                <div style="background: linear-gradient(135deg, #064e3b, #065f46); padding: 12px 20px; border-radius: 8px; color: #00ff88; font-weight: bold; border: 1px solid #00ff88;">
                    🎯 Task Identified<br><small style="font-weight: normal;">(Highest confidence)</small>
                </div>
            </div>
        </div>
    </div>
    
    <h3 style="color: #a78bfa; border-bottom: 2px solid #a78bfa; padding-bottom: 10px; margin-top: 30px;">🔍 How Detection Works</h3>
    <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; margin: 15px 0; border: 1px solid #2d3748;">
        <ol style="color: #ccc; line-height: 2; margin: 0; padding-left: 20px;">
            <li><strong style="color: #00d4ff;">Input Analysis</strong> - The text is tokenized and converted to bit representation</li>
            <li><strong style="color: #00ff88;">Multi-Key Testing</strong> - The system tests detection with each task's secret key (π, e, φ)</li>
            <li><strong style="color: #ffd93d;">Statistical Scoring</strong> - For each key, compute T(x) = Σ(-log(vⱼ)) across all blocks</li>
            <li><strong style="color: #ff6b6b;">Threshold Comparison</strong> - Compare scores against threshold: T(x) > n + λ√n</li>
            <li><strong style="color: #a78bfa;">Confidence Calculation</strong> - Combine crypto, synonym, and character detection results</li>
            <li><strong style="color: #fff;">Task Selection</strong> - The task with highest combined confidence is identified</li>
        </ol>
    </div>
    
    <h3 style="color: #ff6b6b; border-bottom: 2px solid #ff6b6b; padding-bottom: 10px; margin-top: 30px;">📈 Overall Performance Evaluation</h3>
    <div style="display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; margin: 20px 0;">
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px 30px; border-radius: 12px; text-align: center; min-width: 140px; border: 1px solid #2d3748;">
            <div style="font-size: 2.2em; font-weight: bold; color: #00d4ff;">86.9%</div>
            <div style="color: #888; margin-top: 5px;">Detection Rate</div>
        </div>
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px 30px; border-radius: 12px; text-align: center; min-width: 140px; border: 1px solid #2d3748;">
            <div style="font-size: 2.2em; font-weight: bold; color: #00ff88;">85.1%</div>
            <div style="color: #888; margin-top: 5px;">Task Accuracy</div>
        </div>
        <div style="background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 20px 30px; border-radius: 12px; text-align: center; min-width: 140px; border: 1px solid #2d3748;">
            <div style="font-size: 2.2em; font-weight: bold; color: #ffd93d;">450</div>
            <div style="color: #888; margin-top: 5px;">Test Samples</div>
        </div>
    </div>
    
    <!-- Task Performance Table -->
    <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
        <thead>
            <tr style="background: #2d3748;">
                <th style="padding: 10px; text-align: left; color: #00d4ff;">Task</th>
                <th style="padding: 10px; text-align: center; color: #00d4ff;">Accuracy</th>
                <th style="padding: 10px; text-align: left; color: #00d4ff;">Performance</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #2d3748;">
                <td style="padding: 10px; color: #fff;">QA</td>
                <td style="padding: 10px; text-align: center; color: #00ff88; font-weight: bold;">96.7%</td>
                <td style="padding: 10px;">
                    <div style="background: #2d3748; border-radius: 5px; height: 20px; width: 100%;">
                        <div style="background: #00ff88; width: 96.7%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </td>
            </tr>
            <tr style="border-bottom: 1px solid #2d3748;">
                <td style="padding: 10px; color: #fff;">Summary</td>
                <td style="padding: 10px; text-align: center; color: #ffd93d; font-weight: bold;">73.3%</td>
                <td style="padding: 10px;">
                    <div style="background: #2d3748; border-radius: 5px; height: 20px; width: 100%;">
                        <div style="background: #ffd93d; width: 73.3%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </td>
            </tr>
            <tr>
                <td style="padding: 10px; color: #fff;">News</td>
                <td style="padding: 10px; text-align: center; color: #ff6b6b; font-weight: bold;">85.3%</td>
                <td style="padding: 10px;">
                    <div style="background: #2d3748; border-radius: 5px; height: 20px; width: 100%;">
                        <div style="background: #ff6b6b; width: 85.3%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </td>
            </tr>
        </tbody>
    </table>
    
    <!-- Length Impact -->
    <div style="margin-top: 25px;">
        <h4 style="color: #888; margin-bottom: 15px;">📏 Length Impact on Accuracy</h4>
        <div style="display: flex; gap: 15px; flex-wrap: wrap; justify-content: center;">
            <div style="background: #2d3748; padding: 15px 25px; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold; color: #ff6b6b;">77.3%</div>
                <div style="color: #888; font-size: 0.85em;">50 tokens</div>
            </div>
            <div style="font-size: 2em; color: #4a5568; display: flex; align-items: center;">→</div>
            <div style="background: #2d3748; padding: 15px 25px; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold; color: #ffd93d;">86.0%</div>
                <div style="color: #888; font-size: 0.85em;">100 tokens</div>
            </div>
            <div style="font-size: 2em; color: #4a5568; display: flex; align-items: center;">→</div>
            <div style="background: #2d3748; padding: 15px 25px; border-radius: 10px; text-align: center;">
                <div style="font-size: 1.5em; font-weight: bold; color: #00ff88;">92.0%</div>
                <div style="color: #888; font-size: 0.85em;">200 tokens</div>
            </div>
        </div>
        <p style="color: #666; text-align: center; margin-top: 10px; font-size: 0.9em;">Longer texts provide more watermark blocks, improving detection reliability</p>
    </div>
    
    <!-- Confusion Matrix -->
    <div style="margin-top: 25px;">
        <h4 style="color: #888; margin-bottom: 15px;">🔀 Confusion Matrix</h4>
        <div style="display: flex; justify-content: center;">
            <table style="border-collapse: collapse; text-align: center;">
                <thead>
                    <tr>
                        <th style="padding: 10px; color: #888;"></th>
                        <th style="padding: 10px; color: #00d4ff; font-size: 0.85em;">Pred: QA</th>
                        <th style="padding: 10px; color: #ffd93d; font-size: 0.85em;">Pred: SUM</th>
                        <th style="padding: 10px; color: #ff6b6b; font-size: 0.85em;">Pred: NEWS</th>
                        <th style="padding: 10px; color: #888; font-size: 0.85em;">None</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td style="padding: 10px; color: #00d4ff; font-weight: bold;">True: QA</td>
                        <td style="padding: 10px; background: #064e3b; color: #00ff88; font-weight: bold; border-radius: 5px;">145</td>
                        <td style="padding: 10px; color: #666;">0</td>
                        <td style="padding: 10px; color: #666;">0</td>
                        <td style="padding: 10px; color: #888;">5</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; color: #ffd93d; font-weight: bold;">True: SUM</td>
                        <td style="padding: 10px; color: #888;">7</td>
                        <td style="padding: 10px; background: #064e3b; color: #00ff88; font-weight: bold; border-radius: 5px;">110</td>
                        <td style="padding: 10px; color: #666;">0</td>
                        <td style="padding: 10px; color: #888;">33</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; color: #ff6b6b; font-weight: bold;">True: NEWS</td>
                        <td style="padding: 10px; color: #888;">1</td>
                        <td style="padding: 10px; color: #666;">0</td>
                        <td style="padding: 10px; background: #064e3b; color: #00ff88; font-weight: bold; border-radius: 5px;">128</td>
                        <td style="padding: 10px; color: #888;">21</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    
    <!-- ROBUSTNESS ANALYSIS SECTION -->
    <h3 style="color: #f472b6; border-bottom: 2px solid #f472b6; padding-bottom: 10px; margin-top: 40px;">🛡️ Robustness Analysis</h3>
    <p style="color: #ccc; line-height: 1.8; margin-bottom: 20px;">
        We evaluated the watermark's robustness against three types of attacks to understand its real-world resilience. 
        Tests were conducted on <strong>434 samples</strong> with valid watermark boundaries.
    </p>
    
    <!-- Attack Descriptions -->
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; margin: 20px 0;">
        <div style="background: #1a1a2e; border-radius: 12px; padding: 18px; border-left: 4px solid #00d4ff;">
            <h4 style="color: #00d4ff; margin: 0 0 10px 0;">📏 Truncation Attack</h4>
            <p style="color: #aaa; font-size: 0.9em; margin: 0;">
                Removes portions of the text from the end. Tests if watermark survives partial text extraction. 
                Simulates scenarios where only part of generated content is used.
            </p>
        </div>
        <div style="background: #1a1a2e; border-radius: 12px; padding: 18px; border-left: 4px solid #ff6b6b;">
            <h4 style="color: #ff6b6b; margin: 0 0 10px 0;">🔤 Character Perturbation</h4>
            <p style="color: #aaa; font-size: 0.9em; margin: 0;">
                Randomly inserts, deletes, or replaces characters. Tests robustness against typos, OCR errors, 
                or intentional character-level modifications.
            </p>
        </div>
        <div style="background: #1a1a2e; border-radius: 12px; padding: 18px; border-left: 4px solid #a78bfa;">
            <h4 style="color: #a78bfa; margin: 0 0 10px 0;">📝 Paraphrasing Attack</h4>
            <p style="color: #aaa; font-size: 0.9em; margin: 0;">
                Rewrites text using <strong>T5 model</strong> (<code style="background: #2d3748; padding: 2px 6px; border-radius: 4px; font-size: 0.85em;">humarin/chatgpt_paraphraser_on_T5_base</code>). 
                Tests if watermark survives semantic-preserving rewrites that change word choices and sentence structure.
            </p>
        </div>
    </div>
    
    <!-- Robustness Results -->
    <div style="margin-top: 25px;">
        <h4 style="color: #888; margin-bottom: 15px;">📊 Robustness Results</h4>
        
        <!-- Truncation Results -->
        <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; margin-bottom: 15px; border: 1px solid #2d3748;">
            <h5 style="color: #00d4ff; margin: 0 0 15px 0;">📏 Truncation</h5>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 100px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #00ff88;">87.6%</div>
                    <div style="font-size: 0.75em; color: #888;">100% text</div>
                </div>
                <div style="display: flex; align-items: center; color: #4a5568;">→</div>
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 100px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #00ff88;">82.5%</div>
                    <div style="font-size: 0.75em; color: #888;">75% text</div>
                </div>
                <div style="display: flex; align-items: center; color: #4a5568;">→</div>
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 100px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #ffd93d;">73.5%</div>
                    <div style="font-size: 0.75em; color: #888;">50% text</div>
                </div>
                <div style="display: flex; align-items: center; color: #4a5568;">→</div>
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 100px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #ff6b6b;">50.2%</div>
                    <div style="font-size: 0.75em; color: #888;">25% text</div>
                </div>
            </div>
            <p style="color: #666; text-align: center; margin-top: 12px; font-size: 0.85em;">
                ⭐⭐⭐ <strong>High robustness</strong> - Watermark remains effective even with 50% of text removed
            </p>
        </div>
        
        <!-- Perturbation Results -->
        <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; margin-bottom: 15px; border: 1px solid #2d3748;">
            <h5 style="color: #ff6b6b; margin: 0 0 15px 0;">🔤 Character Perturbation</h5>
            <div style="display: flex; gap: 10px; flex-wrap: wrap; justify-content: center;">
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 90px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #00ff88;">87.6%</div>
                    <div style="font-size: 0.75em; color: #888;">0% noise</div>
                </div>
                <div style="display: flex; align-items: center; color: #4a5568;">→</div>
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 90px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #ffd93d;">75.6%</div>
                    <div style="font-size: 0.75em; color: #888;">1% noise</div>
                </div>
                <div style="display: flex; align-items: center; color: #4a5568;">→</div>
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 90px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #ffd93d;">68.0%</div>
                    <div style="font-size: 0.75em; color: #888;">5% noise</div>
                </div>
                <div style="display: flex; align-items: center; color: #4a5568;">→</div>
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 90px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #ffd93d;">68.2%</div>
                    <div style="font-size: 0.75em; color: #888;">10% noise</div>
                </div>
                <div style="display: flex; align-items: center; color: #4a5568;">→</div>
                <div style="text-align: center; padding: 12px 20px; background: #2d3748; border-radius: 8px; min-width: 90px;">
                    <div style="font-size: 1.3em; font-weight: bold; color: #ffd93d;">68.0%</div>
                    <div style="font-size: 0.75em; color: #888;">20% noise</div>
                </div>
            </div>
            <p style="color: #666; text-align: center; margin-top: 12px; font-size: 0.85em;">
                ⭐⭐ <strong>Medium robustness</strong> - Crypto watermark compensates for character watermark degradation
            </p>
        </div>
        
        <!-- Paraphrase Results -->
        <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; border: 1px solid #2d3748;">
            <h5 style="color: #a78bfa; margin: 0 0 15px 0;">📝 Paraphrasing (T5 Model)</h5>
            <div style="display: flex; gap: 30px; flex-wrap: wrap; justify-content: center;">
                <div style="text-align: center; padding: 15px 25px; background: #2d3748; border-radius: 8px; min-width: 140px;">
                    <div style="font-size: 1.5em; font-weight: bold; color: #00ff88;">87.6%</div>
                    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">Original Text</div>
                </div>
                <div style="display: flex; align-items: center; font-size: 2em; color: #4a5568;">→</div>
                <div style="text-align: center; padding: 15px 25px; background: #2d3748; border-radius: 8px; min-width: 140px;">
                    <div style="font-size: 1.5em; font-weight: bold; color: #ff6b6b;">29.7%</div>
                    <div style="font-size: 0.85em; color: #888; margin-top: 5px;">Paraphrased</div>
                </div>
            </div>
            <p style="color: #666; text-align: center; margin-top: 15px; font-size: 0.85em;">
                ⭐ <strong>Low robustness</strong> - Paraphrasing destroys watermark (29.7% ≈ 33.3% random chance)
            </p>
            <div style="background: #2d3748; border-radius: 8px; padding: 12px; margin-top: 15px;">
                <p style="color: #aaa; font-size: 0.85em; margin: 0; text-align: center;">
                    <strong style="color: #f472b6;">Note:</strong> After paraphrasing, the Detection Rate remains high (~91%) but Task Accuracy drops to random levels. 
                    This occurs because the detector finds coincidental pattern matches in the rewritten text, but cannot identify the correct task 
                    since the original watermark structure has been destroyed.
                </p>
            </div>
        </div>
    </div>
    
    <!-- Robustness Summary Table -->
    <div style="margin-top: 25px; background: linear-gradient(135deg, #1a1a2e, #16213e); border-radius: 12px; padding: 20px; border: 1px solid #4a5568;">
        <h4 style="color: #fff; margin: 0 0 15px 0; text-align: center;">🎯 Robustness Summary</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="border-bottom: 1px solid #4a5568;">
                    <th style="padding: 10px; text-align: left; color: #888;">Attack Type</th>
                    <th style="padding: 10px; text-align: center; color: #888;">Robustness</th>
                    <th style="padding: 10px; text-align: left; color: #888;">Conclusion</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom: 1px solid #2d3748;">
                    <td style="padding: 10px; color: #00d4ff;">Truncation</td>
                    <td style="padding: 10px; text-align: center;">⭐⭐⭐</td>
                    <td style="padding: 10px; color: #ccc;">Effective up to 50% text removal</td>
                </tr>
                <tr style="border-bottom: 1px solid #2d3748;">
                    <td style="padding: 10px; color: #ff6b6b;">Character Perturbation</td>
                    <td style="padding: 10px; text-align: center;">⭐⭐</td>
                    <td style="padding: 10px; color: #ccc;">Crypto layer compensates (~68% accuracy)</td>
                </tr>
                <tr>
                    <td style="padding: 10px; color: #a78bfa;">Paraphrasing</td>
                    <td style="padding: 10px; text-align: center;">⭐</td>
                    <td style="padding: 10px; color: #ccc;">Destroyed (expected for text rewriting)</td>
                </tr>
            </tbody>
        </table>
    </div>
    
    <h3 style="color: #00ff88; border-bottom: 2px solid #00ff88; padding-bottom: 10px; margin-top: 40px;">📚 Reference</h3>
    <p style="color: #ccc; line-height: 1.8;">
        <strong>Christ, M., Gunn, S., & Zamir, O. (2024)</strong><br>
        <em>"Undetectable Watermarks for Language Models"</em><br>
        <span style="color: #888;">Conference on Learning Theory (COLT)</span>
    </p>
    
</div>
            """)
        
        # Footer
        gr.HTML("""
<div class="footer-text">
    <p style="font-size: 1.1em; color: #888; margin-bottom: 5px;"><strong>AI Systems Engineering Project Work - Selective Watermarking</strong></p>
    <p style="color: #666;">Built with GPT-2 Medium | Gradio UI</p>
</div>
        """)
    
    return demo


def main():
    print("=" * 70)
    print("GRADIO DEMO - Selective Watermarking")
    print("=" * 70)
    
    print("\nCreazione interfaccia...")
    demo = create_demo()
    
    print("\nAvvio server...")
    print("Apri http://localhost:7860 nel browser\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
