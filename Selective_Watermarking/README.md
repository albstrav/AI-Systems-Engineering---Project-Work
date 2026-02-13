# Selective Watermarking 

src/watermarkers/

crypto_watermark_christ.py → Implementazione del watermark crittografico 
synonym_watermark.py → Watermark basato su sostituzione di sinonimi
character_watermark.py → Watermark basato su caratteri Unicode lookalike
multilayer.py → Combina le tre tecniche e gestisce la configurazione per task (QA, Summary, News)

tests/

test_watermarkers.py → Test automatici per verificare che i watermarker funzionino

src/generation/

generator.py → Generatore di testo watermarked usando GPT-2 Medium
prompts.py → Lista di prompt predefiniti per ogni task


scripts/

generate_dataset.py → Genera il dataset di 450 campioni watermarked
generate_single_task.py → Genera campioni per un singolo task
merge_datasets.py → Unisce più dataset
evaluate.py → Valuta detection rate e task accuracy sul dataset
create_plots.py → Crea i grafici dei risultati
robustness_analysis.py → Analisi di robustezza (truncation, perturbation, paraphrasing)
test_character.py → Test del character watermark
test_crypto_watermark.py → Test del crypto watermark
test_multilayer.py → Test del sistema multilayer
test_paraphrase.py → Test del paraphrasing con T5
test_synonym.py → Test del synonym watermark
demo.py → Demo interattiva Gradio

data/generated/

dataset.csv → Dataset con testi watermarked
boundaries.json → Block boundaries per ogni campione
dataset_metadata.json → Contiene informazioni sul dataset generato

data/results/

evaluation_results.json → Risultati della valutazione
robustness_results.json → Risultati analisi di robustezza
confusion_matrix.csv → Matrice di confusione (QA vs Summary vs News)
detailed_predictions.csv → Predizioni dettagliate per ogni campione del dataset
length_impact.csv → Accuracy per lunghezza testo (50, 100, 200 token)

figures/

confusion_matrix.png → Matrice di confusione
accuracy_by_length.png → Accuracy per lunghezza testo
length_impact.png → Impatto lunghezza
summary_dashboard.png → summary risultati
robustness_truncation.png → Grafico robustezza truncation
robustness_perturbation.png → Grafico robustezza perturbation
robustness_paraphrase.png → Grafico robustezza paraphrasing
robustness_summary.png → Riepilogo robustezza



python tests/test_watermarkers.py
python scripts/test_synonym.py
python scripts/test_character.py
python scripts/test_crypto_watermark.py
python scripts/test_multilayer.py
python scripts/test_paraphrase.py
python scripts/generate_dataset.py
python scripts/generate_single_task.py
python scripts/merge_datasets.py
python scripts/create_plots.py
python scripts/run_evaluation.py
python scripts/robustness_analysis.py
python scripts/robustness_analysis.py --only-truncation
python scripts/robustness_analysis.py --only-perturbation
python scripts/robustness_analysis.py --only-paraphrase
python scripts/demo.py

