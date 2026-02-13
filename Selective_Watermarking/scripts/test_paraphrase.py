#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def test_paraphraser():
    print("=" * 60)
    print("TEST PARAPHRASING MODEL")
    print("=" * 60)
    
    # Modello
    model_name = "humarin/chatgpt_paraphraser_on_T5_base"
    
    print(f"\n1. Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"   ✓ Loaded on {device.type.upper()}")
    
    # Testi di test (alcuni con caratteri Unicode lookalike)
    test_texts = [
        # Testo normale
        "Artificial intelligence is transforming the way we live and work.",
        
        # Testo con caratteri Unicode lookalike (simula watermark)
        "Thе quіck brоwn fоx jumps оver thе lаzy dоg.",
        
        # Testo tecnico
        "Machine learning algorithms can identify patterns in large datasets.",
        
        # Testo più lungo
        "The development of quantum computing represents a significant advancement in computational technology. These systems leverage quantum mechanical phenomena to process information in fundamentally new ways."
    ]
    
    print("\n2. Testing paraphrasing...\n")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n[Test {i}]")
        print(f"ORIGINAL:    {text}")
        
        # Paraphrase
        input_text = f"paraphrase: {text}"
        
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
                num_return_sequences=1,
                early_stopping=True
            )
        
        paraphrased = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"PARAPHRASED: {paraphrased}")
        
        # Verifica se è diverso
        if text.lower().strip() == paraphrased.lower().strip():
            print("⚠️  WARNING: Output identical to input!")
        elif len(set(text.split()) & set(paraphrased.split())) / len(set(text.split())) > 0.8:
            print("⚠️  WARNING: Very similar to input")
        else:
            print("✓  Good: Text was paraphrased")
    
    print("\n" + "-" * 60)
    print("\n3. Test complete!")
    print("   If paraphrases look different from originals, the model works.")
    print()

if __name__ == "__main__":
    test_paraphraser()
