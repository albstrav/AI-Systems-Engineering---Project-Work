const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, 
        HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
        PageBreak, TableOfContents, LevelFormat, Header, Footer, PageNumber } = require('docx');
const fs = require('fs');

const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, bold: true })], spacing: { before: 360, after: 240 } });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, bold: true })], spacing: { before: 280, after: 180 } });
const h3 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text, bold: true })], spacing: { before: 200, after: 120 } });
const p = (text) => new Paragraph({ children: [new TextRun(text)], spacing: { after: 120 } });
const code = (text) => new Paragraph({ children: [new TextRun({ text, font: "Courier New", size: 18 })], shading: { fill: "F5F5F5", type: ShadingType.CLEAR }, spacing: { after: 60 } });
const bullet = (text) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: [new TextRun(text)] });

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cell = (text, width, header = false) => new TableCell({
    borders, width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: "E8E8E8", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold: header, size: 20 })] })]
});

const doc = new Document({
    styles: {
        default: { document: { run: { font: "Arial", size: 22 } } },
        paragraphStyles: [
            { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 36, bold: true, font: "Arial", color: "1F4E79" }, paragraph: { spacing: { before: 360, after: 240 }, outlineLevel: 0 } },
            { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 28, bold: true, font: "Arial", color: "1F4E79" }, paragraph: { spacing: { before: 280, after: 180 }, outlineLevel: 1 } },
            { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 24, bold: true, font: "Arial" }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
        ]
    },
    numbering: { config: [
        { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
        { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]},
    sections: [{
        properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 } } },
        headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Selective Watermarking - Documentazione", size: 18, italics: true })] })] }) },
        footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Pagina ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
        children: [
            // TITLE
            new Paragraph({ spacing: { after: 2000 } }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "SELECTIVE WATERMARKING", bold: true, size: 56, color: "1F4E79" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "FOR AI-GENERATED TEXT", bold: true, size: 56, color: "1F4E79" })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 800 }, children: [new TextRun({ text: "Documentazione Tecnica Completa", size: 32, italics: true })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: "Corso AISE - Gennaio 2026", size: 24 })] }),
            new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Basato su: Christ, Gunn, Zamir "Undetectable Watermarks for Language Models" COLT 2024', size: 20, italics: true })] }),
            new Paragraph({ children: [new PageBreak()] }),
            
            // TOC
            new Paragraph({ children: [new TextRun({ text: "INDICE", bold: true, size: 32 })] }),
            new TableOfContents("Indice", { hyperlink: true, headingStyleRange: "1-3" }),
            new Paragraph({ children: [new PageBreak()] }),

            // SECTION 1
            h1("1. Panoramica del Progetto"),
            h2("1.1 Obiettivo"),
            p("Creare un sistema di watermarking che inserisce un marchio invisibile nel testo AI-generated, permettendo di:"),
            bullet("Rilevare se un testo e stato generato da AI"),
            bullet("Identificare per quale TASK specifico e stato generato (QA, Summary, News)"),
            p(""),
            
            h2("1.2 I Tre Task con Tecniche Diversificate"),
            new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [1800, 2400, 3600, 1560], rows: [
                new TableRow({ children: [cell("Task", 1800, true), cell("Chiave", 2400, true), cell("Tecniche", 3600, true), cell("Uso", 1560, true)] }),
                new TableRow({ children: [cell("QA", 1800), cell("314159265 (pi)", 2400), cell("Crypto + Character", 3600), cell("Q&A", 1560)] }),
                new TableRow({ children: [cell("Summary", 1800), cell("271828182 (e)", 2400), cell("Crypto + Synonym", 3600), cell("Riassunti", 1560)] }),
                new TableRow({ children: [cell("News", 1800), cell("161803398 (phi)", 2400), cell("Crypto + Synonym + Char", 3600), cell("Notizie", 1560)] }),
            ]}),
            p(""),
            p("NOTA IMPORTANTE: Ogni task usa tecniche DIVERSE per massimizzare la distinguibilita durante la detection."),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            h2("1.3 Le Tre Tecniche"),
            h3("Crypto (Crittografico)"),
            p("Basato su Christ et al. (COLT 2024). Watermark inserito DURANTE la generazione."),
            bullet("Usa PRF (Pseudorandom Function) con chiave segreta"),
            bullet("Statisticamente INDISTINGUIBILE dal testo normale"),
            bullet("Implementato con HMAC-SHA256"),
            p(""),
            h3("Synonym (Sinonimi)"),
            p("Sostituisce parole con sinonimi selezionati deterministicamente."),
            bullet("Applicata DOPO la generazione"),
            bullet("50+ gruppi di sinonimi curati manualmente"),
            bullet("Scelta dipende da: chiave + contesto + parola base"),
            p(""),
            h3("Character (Unicode)"),
            p("Sostituisce caratteri ASCII con lookalikes Unicode invisibili."),
            bullet("Esempio: 'a' ASCII -> 'a' Cirillico (visivamente identici)"),
            bullet("Pattern deterministico basato sulla chiave"),
            bullet("Completamente invisibile all occhio umano"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // SECTION 2
            h1("2. Guida al Lavoro Diviso (3 Persone)"),
            
            h2("2.1 Assegnazione"),
            new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [2340, 1560, 2808, 2652], rows: [
                new TableRow({ children: [cell("Persona", 2340, true), cell("Task", 1560, true), cell("Ambiente", 2808, true), cell("Tecniche", 2652, true)] }),
                new TableRow({ children: [cell("Collega 1", 2340), cell("QA", 1560), cell("Google Colab", 2808), cell("Crypto + Character", 2652)] }),
                new TableRow({ children: [cell("Collega 2", 2340), cell("Summary", 1560), cell("Google Colab", 2808), cell("Crypto + Synonym", 2652)] }),
                new TableRow({ children: [cell("Tu", 2340), cell("News", 1560), cell("PC Locale", 2808), cell("Crypto + Syn + Char", 2652)] }),
            ]}),
            p(""),
            p("SICUREZZA PC: Il codice NON danneggia il PC. Non modifica file di sistema, non richiede permessi admin, usa solo CPU/RAM per calcoli."),
            p(""),
            
            h2("2.2 Comandi per Google Colab (Colleghi)"),
            code("# Cella 1: Setup"),
            code("!unzip -q selective_watermarking_v2_FINALE.zip"),
            code("%cd selective_watermarking_v2"),
            code("!pip install -q torch transformers tqdm"),
            code(""),
            code("# Cella 2: Verifica installazione"),
            code("!python src/watermarkers/crypto_watermark_christ.py"),
            code(""),
            code("# Cella 3: Genera dataset (TASK = qa oppure summary)"),
            code("!python scripts/generate_single_task.py --task TASK --samples 50"),
            code(""),
            code("# Cella 4: Scarica risultato"),
            code("from google.colab import files"),
            code("files.download('data/generated/TASK_dataset.csv')"),
            p(""),
            
            h2("2.3 Comandi per PC Locale (Tu)"),
            code("# Setup (una sola volta)"),
            code("cd selective_watermarking_v2"),
            code("python -m venv venv"),
            code("source venv/bin/activate  # Linux/Mac"),
            code("# oppure: venv\\Scripts\\activate  # Windows"),
            code("pip install -r requirements.txt"),
            code(""),
            code("# Genera task News"),
            code("python scripts/generate_single_task.py --task news --samples 50"),
            code(""),
            code("# Dopo aver ricevuto i CSV dai colleghi, mettili in data/generated/"),
            code("# Poi unisci e valuta:"),
            code("python scripts/merge_datasets.py"),
            code("python scripts/run_evaluation.py"),
            code("python scripts/create_plots.py"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // SECTION 3
            h1("3. Algoritmi dal Paper"),
            
            h2("3.1 Algoritmo 1: Generazione (Wat_sk)"),
            code("r = NULL, H = 0, l = 1"),
            code("while not done:"),
            code("    p_i = Model(prompt, x_1..x_{i-1})"),
            code("    if r = NULL:"),
            code("        x_i ~ p_i  // Random vero"),
            code("    else:"),
            code("        x_i = 1[F_sk(r, l) <= p_i(1)]  // Watermark"),
            code("    H = H - log(p_i(x_i))"),
            code("    if H >= (2/ln2) * lambda * sqrt(l):"),
            code("        r = blocco_corrente"),
            code("        H = 0, l = 0"),
            p(""),
            
            h2("3.2 Algoritmo 2: Detection (Detect_sk)"),
            code("Per ogni blocco candidato (i, l):"),
            code("    r = (x_{i-l}, ..., x_i)"),
            code("    score = 0"),
            code("    for j = i to L:"),
            code("        v_j = x_j * F_sk(r,j-i) + (1-x_j)*(1-F_sk(r,j-i))"),
            code("        score += -log(v_j)"),
            code("        if score > (j-i) + lambda*sqrt(j-i):"),
            code("            return TRUE"),
            code("return FALSE"),
            p(""),
            
            h2("3.3 PRF - Implementazione"),
            p("Usiamo HMAC-SHA256, standard industriale per PRF:"),
            code("def evaluate(self, *inputs) -> float:"),
            code("    h = hmac.new(key, repr(inputs).encode(), sha256)"),
            code("    z = int.from_bytes(h.digest()[:8], 'big')"),
            code("    return z / (2**64)  # Valore in [0, 1]"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // SECTION 4
            h1("4. Metriche di Valutazione"),
            
            h2("4.1 Metriche Target"),
            new Table({ width: { size: 100, type: WidthType.PERCENTAGE }, columnWidths: [3120, 1560, 4680], rows: [
                new TableRow({ children: [cell("Metrica", 3120, true), cell("Target", 1560, true), cell("Descrizione", 4680, true)] }),
                new TableRow({ children: [cell("Task Accuracy", 3120), cell(">90%", 1560), cell("Testi classificati nel task corretto", 4680)] }),
                new TableRow({ children: [cell("False Positive Rate", 3120), cell("<15%", 1560), cell("Testi non-watermarked rilevati erroneamente", 4680)] }),
                new TableRow({ children: [cell("Cross-task", 3120), cell("<15%", 1560), cell("Task X rilevato come task Y", 4680)] }),
            ]}),
            p(""),
            
            h2("4.2 File Generati"),
            bullet("data/results/evaluation_results.json - Metriche complete"),
            bullet("data/results/confusion_matrix.csv - Matrice 3x3"),
            bullet("figures/summary_dashboard.png - Dashboard grafico"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // SECTION 5
            h1("5. Struttura File del Progetto"),
            code("selective_watermarking_v2/"),
            code("  src/watermarkers/"),
            code("    crypto_watermark_christ.py  # ~1000 righe - Algoritmi 1,2"),
            code("    synonym_watermark.py        # ~300 righe - Sinonimi"),
            code("    character_watermark.py      # ~200 righe - Unicode"),
            code("    multilayer.py               # ~550 righe - Sistema combinato"),
            code("  src/generation/"),
            code("    prompts.py                  # 150 prompt (50 per task)"),
            code("    generator.py                # Wrapper GPT-2"),
            code("  scripts/"),
            code("    generate_single_task.py     # Genera UN task"),
            code("    merge_datasets.py           # Unisce dataset"),
            code("    run_evaluation.py           # Calcola metriche"),
            code("    create_plots.py             # Genera grafici"),
            code("  tests/"),
            code("    test_watermarkers.py        # Unit tests"),
            p(""),
            p("TOTALE: ~5300 righe di codice Python"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // SECTION 6
            h1("6. Troubleshooting"),
            
            h3("Errore: ModuleNotFoundError torch"),
            code("pip install torch transformers"),
            p(""),
            h3("Errore: CUDA out of memory"),
            p("Il codice funziona anche senza GPU. Usa automaticamente CPU se CUDA non disponibile."),
            p(""),
            h3("Generazione lenta"),
            p("Tempi stimati per 200 campioni: ~20 min con GPU, ~60 min senza GPU."),
            p("Per test veloci usa: --samples 5"),
            p(""),
            h3("File non trovato"),
            code("cd selective_watermarking_v2"),
            code("ls  # Verifica di essere nella cartella giusta"),
            
            new Paragraph({ children: [new PageBreak()] }),
            
            // SECTION 7
            h1("7. Checklist Finale"),
            bullet("Dataset QA generato (Collega 1)"),
            bullet("Dataset Summary generato (Collega 2)"),
            bullet("Dataset News generato (Tu)"),
            bullet("Merge completato con merge_datasets.py"),
            bullet("Valutazione eseguita con run_evaluation.py"),
            bullet("Grafici generati con create_plots.py"),
            bullet("Risultati inseriti in questo documento"),
            p(""),
            h2("Riferimento"),
            p("Christ, M., Gunn, S., & Zamir, O. (2024). Undetectable Watermarks for Language Models. COLT 2024."),
        ]
    }]
});

Packer.toBuffer(doc).then(buffer => {
    fs.writeFileSync("/mnt/user-data/outputs/Selective_Watermarking_Guida_Completa.docx", buffer);
    console.log("Documento creato: Selective_Watermarking_Guida_Completa.docx");
});
