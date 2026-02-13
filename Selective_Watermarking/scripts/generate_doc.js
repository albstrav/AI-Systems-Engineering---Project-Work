const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
        PageBreak, TableOfContents, LevelFormat } = require('docx');
const fs = require('fs');

// Helper per creare paragrafi
const p = (text, options = {}) => new Paragraph({
    children: [new TextRun({ text, ...options.run })],
    ...options.para
});

const h1 = (text) => new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({ text, bold: true })]
});

const h2 = (text) => new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({ text, bold: true })]
});

const h3 = (text) => new Paragraph({
    heading: HeadingLevel.HEADING_3,
    children: [new TextRun({ text, bold: true })]
});

const code = (text) => new Paragraph({
    children: [new TextRun({ text, font: "Courier New", size: 20 })],
    shading: { fill: "F5F5F5", type: ShadingType.CLEAR }
});

const bullet = (text, ref = "bullets", level = 0) => new Paragraph({
    numbering: { reference: ref, level },
    children: [new TextRun(text)]
});

// Tabella
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

const cell = (text, width = 3120, header = false) => new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: "E8E8E8", type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({ 
        children: [new TextRun({ text, bold: header })] 
    })]
});

const tableRow = (cells) => new TableRow({ children: cells });

// Documento
const doc = new Document({
    styles: {
        default: { document: { run: { font: "Arial", size: 24 } } },
        paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 36, bold: true, font: "Arial", color: "2E74B5" },
              paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 28, bold: true, font: "Arial", color: "2E74B5" },
              paragraph: { spacing: { before: 280, after: 180 }, outlineLevel: 1 } },
            { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
              run: { size: 24, bold: true, font: "Arial" },
              paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
        ]
    },
    numbering: {
        config: [
            { reference: "bullets",
              levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
                style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
                { level: 1, format: LevelFormat.BULLET, text: "○", alignment: AlignmentType.LEFT,
                style: { paragraph: { indent: { left: 1080, hanging: 360 } } } }] },
            { reference: "numbers",
              levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
                style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        ]
    },
    sections: [{
        properties: {
            page: {
                size: { width: 12240, height: 15840 },
                margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
            }
        },
        children: [
            // ========== TITOLO ==========
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 400 },
                children: [new TextRun({ 
                    text: "SELECTIVE WATERMARKING FOR AI-GENERATED TEXT",
                    bold: true, size: 48, color: "2E74B5"
                })]
            }),
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 200 },
                children: [new TextRun({ 
                    text: "Specifica Tecnica e Documentazione del Codice",
                    size: 28, italics: true
                })]
            }),
            new Paragraph({
                alignment: AlignmentType.CENTER,
                spacing: { after: 600 },
                children: [new TextRun({ text: "Corso AISE - Gennaio 2026", size: 24 })]
            }),
            
            // TOC
            new TableOfContents("Indice", { hyperlink: true, headingStyleRange: "1-3" }),
            new Paragraph({ children: [new PageBreak()] }),

            // ========== 1. INTRODUZIONE ==========
            h1("1. Introduzione"),
            p("Questo documento descrive in dettaglio l'implementazione del progetto Selective Watermarking, un sistema che permette non solo di identificare se un testo è stato generato da AI, ma anche di determinare per quale task specifico (QA, Summary, News)."),
            p(""),
            h2("1.1 Problema Affrontato"),
            p("I sistemi attuali di watermarking per LLM possono rilevare se un testo è AI-generated, ma non possono distinguere il contesto d'uso. Il nostro sistema risolve questo problema assegnando chiavi crittografiche diverse a task diversi."),
            p(""),
            h2("1.2 Soluzione Proposta"),
            p("Il sistema combina tre tecniche complementari:"),
            bullet("Watermarking Crittografico (Christ et al., 2024): tecnica principale basata su PRF"),
            bullet("Watermarking basato su Sinonimi: tecnica post-hoc per testi brevi"),
            bullet("Watermarking basato su Caratteri Unicode: tecnica invisibile all'occhio umano"),
            p(""),
            h2("1.3 Paper di Riferimento"),
            p("L'implementazione del watermarking crittografico segue ESATTAMENTE il paper:"),
            new Paragraph({
                shading: { fill: "F0F8FF", type: ShadingType.CLEAR },
                spacing: { before: 120, after: 120 },
                children: [new TextRun({ 
                    text: "Christ, M., Gunn, S., & Zamir, O. (2024). Undetectable Watermarks for Language Models. Proceedings of Machine Learning Research, vol 196:1-15, COLT 2024.",
                    italics: true, size: 22
                })]
            }),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // ========== 2. ARCHITETTURA ==========
            h1("2. Architettura del Sistema"),
            p(""),
            h2("2.1 Struttura dei File"),
            code("selective_watermarking_v2/"),
            code("├── src/"),
            code("│   ├── watermarkers/"),
            code("│   │   ├── crypto_watermark_christ.py  # Algoritmi 1 e 2 del paper"),
            code("│   │   ├── synonym_watermark.py        # Sostituzione sinonimi"),
            code("│   │   ├── character_watermark.py      # Unicode lookalikes"),
            code("│   │   ├── multilayer.py               # Sistema combinato"),
            code("│   │   └── __init__.py"),
            code("│   └── __init__.py"),
            code("├── scripts/"),
            code("│   └── test_crypto_watermark.py        # Test con GPT-2"),
            code("└── requirements.txt"),
            p(""),
            h2("2.2 Flusso di Esecuzione"),
            p("1. L'utente fornisce un prompt e specifica il task (QA/Summary/News)"),
            p("2. Il sistema seleziona la configurazione appropriata (chiave + tecniche)"),
            p("3. GPT-2 genera testo con watermark crittografico integrato"),
            p("4. Le tecniche post-hoc (Synonym, Character) vengono applicate"),
            p("5. Il testo watermarked viene restituito all'utente"),
            p(""),
            p("Per la detection:"),
            p("1. Il testo viene testato contro TUTTE le chiavi dei task"),
            p("2. Ogni tecnica calcola uno score di confidence"),
            p("3. Il sistema usa voting pesato per determinare il task"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // ========== 3. CRYPTO WATERMARK ==========
            h1("3. Watermarking Crittografico (crypto_watermark_christ.py)"),
            p("Questo file implementa ESATTAMENTE gli Algoritmi 1 e 2 del paper di Christ et al."),
            p(""),
            
            h2("3.1 Concetti Fondamentali dal Paper"),
            p(""),
            h3("3.1.1 Riduzione a Alfabeto Binario (Sezione 4.1)"),
            p("Il paper opera su alfabeto binario T = {0, 1}. Per vocabolari grandi come GPT-2 (50,257 token), ogni token viene codificato in ceil(log2(50257)) = 16 bit."),
            p(""),
            p("La distribuzione condizionale per ogni bit è calcolata come:"),
            new Paragraph({
                shading: { fill: "FFFACD", type: ShadingType.CLEAR },
                spacing: { before: 80, after: 80 },
                children: [new TextRun({ 
                    text: "p'_{i,j}(b) = Σ_{t ∈ T} p_i(t) · 1[E(t)_k = b_{i,k} per k < j e E(t)_j = b]",
                    font: "Courier New", size: 22
                })]
            }),
            p("Dove p_i è la distribuzione sui token e E(t) è la codifica binaria del token t."),
            p(""),
            
            h3("3.1.2 PRF - Pseudorandom Function"),
            p("Una PRF F_sk: {0,1}^n → [0,1] produce output computazionalmente indistinguibili da random per chi non conosce la chiave sk."),
            p(""),
            p("Implementazione (linee 100-170):"),
            code("class PRF:"),
            code("    def __init__(self, secret_key: int):"),
            code("        self._key_bytes = secret_key.to_bytes(32, byteorder='big')"),
            code(""),
            code("    def evaluate(self, *inputs) -> float:"),
            code("        input_str = repr(inputs).encode('utf-8')"),
            code("        h = hmac.new(self._key_bytes, input_str, hashlib.sha256)"),
            code("        z = int.from_bytes(h.digest()[:8], byteorder='big')"),
            code("        return z / (2**64)  # Normalizza in [0, 1]"),
            p(""),
            p("Usiamo HMAC-SHA256 che è uno standard industriale per PRF."),
            p(""),
            
            h3("3.1.3 Entropia Empirica (Definizione 3-4)"),
            p("L'entropia empirica di un output x dato il modello è:"),
            new Paragraph({
                shading: { fill: "FFFACD", type: ShadingType.CLEAR },
                spacing: { before: 80, after: 80 },
                children: [new TextRun({ 
                    text: "H_e(Model, PROMPT, x) = -log P[Model(PROMPT) = x]",
                    font: "Courier New", size: 22
                })]
            }),
            p("Per ogni bit: H_i = -log2(p_i(x_i)). Il watermark è rilevabile solo con abbastanza entropia."),
            p(""),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            h2("3.2 Algoritmo 1: Generazione Watermarked (Wat_sk)"),
            p("Pagina 12 del paper. Pseudocodice:"),
            p(""),
            code("r ← ⊥, H ← 0, i ← 1, ℓ ← 1"),
            code("while done ∉ (x_1, ..., x_{i-1}) do"),
            code("    p_i ← Model(PROMPT, x_1, ..., x_{i-1})"),
            code("    "),
            code("    if r = ⊥ then"),
            code("        Sample x_i ← p_i  // Random VERO"),
            code("    else"),
            code("        x_i ← 1[F_sk(r, ℓ) ≤ p_i(1)]  // Watermark"),
            code("    "),
            code("    H ← H - log p_i(x_i)"),
            code("    "),
            code("    if H ≥ (2/ln 2) · λ · √ℓ then"),
            code("        r ← (x_{i-ℓ}, ..., x_i)  // Nuovo blocco"),
            code("        H ← 0, ℓ ← 0"),
            code("    "),
            code("    i ← i + 1, ℓ ← ℓ + 1"),
            p(""),
            h3("Spiegazione passo per passo:"),
            bullet("r = ⊥ inizialmente: non abbiamo ancora un nonce"),
            bullet("I primi bit sono generati con random VERO (non PRF)"),
            bullet("Quando l'entropia accumulata supera la soglia (2/ln2)·λ·√ℓ:"),
            bullet("  Il blocco corrente diventa il nuovo nonce r", "bullets", 1),
            bullet("  Da questo punto, i bit sono generati con x_i = 1[F_sk(r, ℓ) ≤ p_i(1)]", "bullets", 1),
            bullet("Questo crea correlazione rilevabile SOLO con la chiave corretta"),
            p(""),
            
            h3("Implementazione (linee 400-550):"),
            code("def generate(self, model, tokenizer, prompt, max_new_tokens):"),
            code("    r = None  # Nonce (⊥)"),
            code("    H = 0.0   # Entropia accumulata"),
            code("    ell = 1   # Lunghezza blocco"),
            code("    "),
            code("    for token_idx in range(max_new_tokens):"),
            code("        # Ottieni distribuzione dal modello"),
            code("        token_probs = softmax(model(input_ids).logits)"),
            code("        "),
            code("        # Per ogni bit del token (16 bit per GPT-2)"),
            code("        for bit_idx in range(16):"),
            code("            prob_bit_1 = compute_bit_probability(...)"),
            code("            "),
            code("            if r is None:"),
            code("                bit = 1 if random() < prob_bit_1 else 0"),
            code("            else:"),
            code("                u = self.prf.evaluate(r, ell)"),
            code("                bit = 1 if u <= prob_bit_1 else 0"),
            code("            "),
            code("            H += -log2(prob del bit scelto)"),
            code("            "),
            code("            if H >= threshold:"),
            code("                r = serialize_block(current_bits)"),
            code("                H = 0"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            h2("3.3 Algoritmo 2: Detection (Detect_sk)"),
            p("Pagina 12 del paper. Pseudocodice:"),
            p(""),
            code("for i, ℓ ∈ [L], ℓ < i do"),
            code("    r^{(i,ℓ)} ← (x_{i-ℓ}, ..., x_i)"),
            code("    "),
            code("    for each j ∈ [L]:"),
            code("        v^{(i,ℓ)}_j ← x_j · F_sk(r, j-i-1) + (1-x_j) · (1-F_sk(r, j-i-1))"),
            code("    "),
            code("    for k ∈ [i+1, L] do"),
            code("        if Σ ln(1/v_j) > (k-i) + λ√(k-i) then"),
            code("            return true"),
            code(""),
            code("return false"),
            p(""),
            h3("Spiegazione:"),
            bullet("Per ogni possibile blocco candidato r:"),
            bullet("  Calcola v_j per ogni bit successivo", "bullets", 1),
            bullet("  v_j misura l'allineamento tra il bit e la PRF", "bullets", 1),
            bullet("  Se il testo è watermarked con la chiave giusta, v_j tende a essere alto", "bullets", 1),
            bullet("Lo score Σ ln(1/v_j) è alto per testo watermarked"),
            bullet("La soglia (k-i) + λ√(k-i) distingue watermarked da non-watermarked"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // ========== 4. SYNONYM ==========
            h1("4. Watermarking Sinonimi (synonym_watermark.py)"),
            p("Tecnica POST-HOC complementare al watermarking crittografico."),
            p(""),
            h2("4.1 Motivazione"),
            p("Il watermarking crittografico richiede molti bit per essere affidabile. Per testi brevi (< 50 token), la detection è rumorosa. I sinonimi aggiungono ridondanza."),
            p(""),
            h2("4.2 Algoritmo di Embedding"),
            code("Per ogni parola w nel testo:"),
            code("    if w appartiene a un gruppo di sinonimi G:"),
            code("        context = parole circostanti"),
            code("        h = hash(secret_key, context, base_word)"),
            code("        selected = G[h mod |G|]  # Selezione deterministica"),
            code("        sostituisci w con selected"),
            p(""),
            h2("4.3 Algoritmo di Detection"),
            code("matches = 0"),
            code("Per ogni parola w nel testo:"),
            code("    if w appartiene a un gruppo di sinonimi G:"),
            code("        expected = G[hash(key, context, base) mod |G|]"),
            code("        if w == expected:"),
            code("            matches++"),
            code(""),
            code("return matches / total > threshold"),
            p(""),
            h2("4.4 Gruppi di Sinonimi Curati"),
            p("50+ gruppi curati manualmente per garantire intercambiabilità semantica:"),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [3120, 6240],
                rows: [
                    tableRow([cell("Parola Base", 3120, true), cell("Sinonimi", 6240, true)]),
                    tableRow([cell("big"), cell("big, large, huge, enormous, vast, massive")]),
                    tableRow([cell("important"), cell("important, significant, crucial, vital, essential")]),
                    tableRow([cell("say"), cell("say, state, declare, mention, express")]),
                    tableRow([cell("problem"), cell("problem, issue, challenge, difficulty, obstacle")]),
                ]
            }),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // ========== 5. CHARACTER ==========
            h1("5. Watermarking Caratteri (character_watermark.py)"),
            p("Tecnica POST-HOC completamente INVISIBILE all'occhio umano."),
            p(""),
            h2("5.1 Principio"),
            p("Molti caratteri Unicode sembrano identici ai caratteri ASCII ma hanno codici diversi:"),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [2340, 2340, 2340, 2340],
                rows: [
                    tableRow([cell("ASCII", 2340, true), cell("Unicode", 2340, true), 
                              cell("Code ASCII", 2340, true), cell("Code Unicode", 2340, true)]),
                    tableRow([cell("a"), cell("а (Cyrillic)"), cell("U+0061"), cell("U+0430")]),
                    tableRow([cell("e"), cell("е (Cyrillic)"), cell("U+0065"), cell("U+0435")]),
                    tableRow([cell("o"), cell("о (Cyrillic)"), cell("U+006F"), cell("U+043E")]),
                    tableRow([cell("c"), cell("с (Cyrillic)"), cell("U+0063"), cell("U+0441")]),
                ]
            }),
            p(""),
            h2("5.2 Algoritmo"),
            code("rng = Random(secret_key)"),
            code("Per ogni carattere c nel testo:"),
            code("    if c ha un lookalike:"),
            code("        if rng.random() < substitution_rate:"),
            code("            sostituisci c con lookalike"),
            p(""),
            p("La detection ri-esegue lo stesso processo e verifica la corrispondenza dei pattern."),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // ========== 6. MULTILAYER ==========
            h1("6. Sistema Multi-Layer (multilayer.py)"),
            p("Combina le tre tecniche con configurazioni specifiche per task."),
            p(""),
            h2("6.1 Configurazioni per Task"),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [1560, 2340, 3120, 2340],
                rows: [
                    tableRow([cell("Task", 1560, true), cell("Chiave", 2340, true), 
                              cell("Tecniche", 3120, true), cell("Rationale", 2340, true)]),
                    tableRow([cell("QA"), cell("314159265"), cell("Crypto + Synonym"), cell("Risposte lunghe")]),
                    tableRow([cell("Summary"), cell("271828182"), cell("Crypto + Synonym"), cell("Testi medi")]),
                    tableRow([cell("News"), cell("161803398"), cell("Crypto + Synonym + Char"), cell("Anche brevi")]),
                ]
            }),
            p(""),
            p("Ogni task ha una chiave DIVERSA per permettere l'identificazione del task."),
            p(""),
            h2("6.2 Weighted Voting"),
            p("La detection finale usa voting pesato:"),
            new Paragraph({
                shading: { fill: "FFFACD", type: ShadingType.CLEAR },
                spacing: { before: 80, after: 80 },
                children: [new TextRun({ 
                    text: "final_score = Σ(weight_i × confidence_i) / Σ(weight_i)",
                    font: "Courier New", size: 22
                })]
            }),
            p("Pesi default: Crypto=0.5, Synonym=0.3, Character=0.2"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // ========== 7. GUIDA REALIZZAZIONE ==========
            h1("7. Guida Passo-Passo alla Realizzazione"),
            p(""),
            h2("7.1 Prerequisiti"),
            bullet("Python 3.9+"),
            bullet("CUDA (opzionale, per accelerazione GPU)"),
            bullet("8GB+ RAM (per GPT-2 Medium)"),
            p(""),
            h2("7.2 Installazione"),
            code("# 1. Clona o estrai il progetto"),
            code("cd selective_watermarking_v2"),
            code(""),
            code("# 2. Crea virtual environment"),
            code("python -m venv venv"),
            code("source venv/bin/activate  # Linux/Mac"),
            code("# oppure: venv\\Scripts\\activate  # Windows"),
            code(""),
            code("# 3. Installa dipendenze"),
            code("pip install -r requirements.txt"),
            code(""),
            code("# 4. Scarica modello GPT-2 (automatico al primo uso)"),
            code("python -c \"from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2-medium')\""),
            p(""),
            h2("7.3 Test di Verifica"),
            code("# Test base (senza GPU)"),
            code("python src/watermarkers/crypto_watermark_christ.py"),
            code("python src/watermarkers/synonym_watermark.py"),
            code("python src/watermarkers/character_watermark.py"),
            code(""),
            code("# Test completo con GPT-2"),
            code("python scripts/test_crypto_watermark.py --max-tokens 30"),
            p(""),
            h2("7.4 Esempio di Utilizzo"),
            code("from transformers import GPT2LMHeadModel, GPT2Tokenizer"),
            code("from watermarkers import MultiLayerWatermarker"),
            code(""),
            code("# Carica modello"),
            code("model = GPT2LMHeadModel.from_pretrained('gpt2-medium')"),
            code("tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')"),
            code(""),
            code("# Crea sistema"),
            code("system = MultiLayerWatermarker()"),
            code(""),
            code("# Genera con watermark"),
            code("result = system.generate(model, tokenizer, 'What is AI?', 'qa', max_new_tokens=50)"),
            code("print(result.text)"),
            code(""),
            code("# Detection"),
            code("detection = system.detect_all_tasks(result.text, tokenizer)"),
            code("print(f'Task: {detection.best_match}, Confidence: {detection.best_confidence}')"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // ========== 8. METRICHE ==========
            h1("8. Metriche di Valutazione"),
            p(""),
            h2("8.1 Metriche Target"),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [4680, 2340, 2340],
                rows: [
                    tableRow([cell("Metrica", 4680, true), cell("Target", 2340, true), cell("Priorità", 2340, true)]),
                    tableRow([cell("Task Accuracy (identificazione task corretto)"), cell(">90%"), cell("Alta")]),
                    tableRow([cell("False Positive Rate (testo umano rilevato)"), cell("<15%"), cell("Alta")]),
                    tableRow([cell("Cross-task Detection (task X con chiave Y)"), cell("<15%"), cell("Media")]),
                    tableRow([cell("Length Impact (accuratezza su testi brevi)"), cell("+25pp"), cell("Alta")]),
                ]
            }),
            p(""),
            h2("8.2 Esperimenti Pianificati"),
            p("1. Generare 600 testi (150 per task + 150 controlli)"),
            p("2. Testare detection con chiavi corrette e sbagliate"),
            p("3. Analizzare impatto della lunghezza"),
            p("4. Creare confusion matrix 3x3"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // ========== 9. APPENDICE ==========
            h1("9. Appendice: Mapping Completo Codice-Paper"),
            p(""),
            new Table({
                width: { size: 100, type: WidthType.PERCENTAGE },
                columnWidths: [3120, 3120, 3120],
                rows: [
                    tableRow([cell("Concetto Paper", 3120, true), cell("File", 3120, true), cell("Linee", 3120, true)]),
                    tableRow([cell("PRF F_sk"), cell("crypto_watermark_christ.py"), cell("100-170")]),
                    tableRow([cell("Riduzione binaria (Sez 4.1)"), cell("crypto_watermark_christ.py"), cell("180-300")]),
                    tableRow([cell("Algoritmo 1 (Wat_sk)"), cell("crypto_watermark_christ.py"), cell("400-550")]),
                    tableRow([cell("Algoritmo 2 (Detect_sk)"), cell("crypto_watermark_christ.py"), cell("560-680")]),
                    tableRow([cell("Soglia entropia"), cell("crypto_watermark_christ.py"), cell("360-380")]),
                    tableRow([cell("Multi-task config"), cell("multilayer.py"), cell("50-100")]),
                ]
            }),
            p(""),
            p("Questo documento è stato generato automaticamente. Per la versione più aggiornata del codice, consultare i file sorgente."),
        ]
    }]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("/mnt/user-data/outputs/Selective_Watermarking_Documentazione_Completa.docx", buffer);
    console.log("Documento creato: Selective_Watermarking_Documentazione_Completa.docx");
});
