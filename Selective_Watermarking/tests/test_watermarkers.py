#!/usr/bin/env python3

import sys
import os
import pytest

# Aggiungi src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# TEST PRF
class TestPRF:
    
    def test_determinism(self):
        """La PRF deve produrre lo stesso output per lo stesso input."""
        from watermarkers.crypto_watermark_christ import PRF
        
        prf1 = PRF(secret_key=12345)
        prf2 = PRF(secret_key=12345)
        
        for i in range(100):
            v1 = prf1.evaluate("test", i)
            v2 = prf2.evaluate("test", i)
            assert v1 == v2, f"PRF non deterministica per input {i}"
    
    def test_range(self):
        """La PRF deve produrre valori in [0, 1]."""
        from watermarkers.crypto_watermark_christ import PRF
        
        prf = PRF(secret_key=42)
        
        for i in range(1000):
            v = prf.evaluate("test", i, "extra")
            assert 0 <= v <= 1, f"Valore {v} fuori range [0, 1]"
    
    def test_different_keys(self):
        """Chiavi diverse devono produrre output diversi."""
        from watermarkers.crypto_watermark_christ import PRF
        
        prf1 = PRF(secret_key=11111)
        prf2 = PRF(secret_key=22222)
        
        same_count = 0
        for i in range(100):
            v1 = prf1.evaluate("test", i)
            v2 = prf2.evaluate("test", i)
            if abs(v1 - v2) < 0.001:
                same_count += 1
        
        # Non più del 5% dovrebbe essere uguale per caso
        assert same_count < 10, f"Troppe collisioni: {same_count}/100"


# TEST BINARY ENCODER
class TestBinaryEncoder:
    
    def test_encode_decode_roundtrip(self):
        """Encode → Decode deve restituire l'originale."""
        from watermarkers.crypto_watermark_christ import BinaryEncoder
        
        encoder = BinaryEncoder(vocab_size=50257)
        
        test_tokens = [0, 1, 100, 1000, 10000, 50000, 50256]
        for token in test_tokens:
            bits = encoder.encode_token(token)
            decoded = encoder.decode_bits(bits)
            assert token == decoded, f"Token {token} decodificato come {decoded}"
    
    def test_bits_per_token(self):
        """Deve usare ceil(log2(vocab_size)) bit."""
        from watermarkers.crypto_watermark_christ import BinaryEncoder
        import math
        
        for vocab_size in [100, 1000, 10000, 50257]:
            encoder = BinaryEncoder(vocab_size=vocab_size)
            expected_bits = math.ceil(math.log2(vocab_size))
            assert encoder.bits_per_token == expected_bits
    
    def test_encode_length(self):
        """Ogni token deve essere codificato in bits_per_token bit."""
        from watermarkers.crypto_watermark_christ import BinaryEncoder
        
        encoder = BinaryEncoder(vocab_size=50257)
        
        for token in [0, 1, 100, 50000]:
            bits = encoder.encode_token(token)
            assert len(bits) == encoder.bits_per_token


# TEST SYNONYM WATERMARKER
class TestSynonymWatermarker:
    
    def test_apply_changes_text(self):
        """Apply deve modificare il testo (almeno alcune parole)."""
        from watermarkers.synonym_watermark import SynonymWatermarker
        
        wm = SynonymWatermarker(secret_key=12345)
        
        text = "The big problem is very important. We need to find a good solution quickly."
        result = wm.apply(text)
        
        # Deve avere fatto almeno qualche sostituzione
        assert result.substitutions_made > 0
        # Il testo deve essere diverso (con alta probabilità)
        # Nota: potrebbe essere uguale se per caso tutti i sinonimi selezionati sono gli originali
    
    def test_detection_correct_key(self):
        """Detection con chiave corretta deve rilevare il watermark."""
        from watermarkers.synonym_watermark import SynonymWatermarker
        
        wm = SynonymWatermarker(secret_key=12345)
        
        text = "The big problem is very important. We need to find a good solution quickly."
        watermarked = wm.apply(text)
        detection = wm.detect(watermarked.text)
        
        # Con la chiave corretta, match_ratio deve essere 100%
        assert detection.match_ratio == 1.0, f"Match ratio: {detection.match_ratio}"
    
    def test_detection_wrong_key(self):
        """Detection con chiave sbagliata deve avere match_ratio basso."""
        from watermarkers.synonym_watermark import SynonymWatermarker
        
        wm_apply = SynonymWatermarker(secret_key=12345)
        wm_detect = SynonymWatermarker(secret_key=99999)
        
        text = "The big problem is very important. We need to find a good solution quickly."
        watermarked = wm_apply.apply(text)
        detection = wm_detect.detect(watermarked.text)
        
        # Con chiave sbagliata, match_ratio dovrebbe essere basso (circa 1/|sinonimi| ≈ 0.2)
        assert detection.match_ratio < 0.5, f"Match ratio troppo alto: {detection.match_ratio}"
    
    def test_preserves_meaning(self):
        """Il testo watermarked deve essere comprensibile (non verificabile automaticamente, ma test base)."""
        from watermarkers.synonym_watermark import SynonymWatermarker
        
        wm = SynonymWatermarker(secret_key=12345)
        
        text = "The important problem needs a good solution."
        result = wm.apply(text)
        
        # Verifica che il testo non sia vuoto o corrotto
        assert len(result.text) > 0
        assert result.text.count(' ') > 0  # Ha più parole


# TEST CHARACTER WATERMARKER
class TestCharacterWatermarker:
    
    def test_apply_substitutes_chars(self):
        """Apply deve sostituire alcuni caratteri."""
        from watermarkers.character_watermark import CharacterWatermarker
        
        wm = CharacterWatermarker(secret_key=12345, substitution_rate=0.5)
        
        text = "The quick brown fox jumps over the lazy dog."
        result = wm.apply(text)
        
        assert result.substituted_chars > 0, "Nessun carattere sostituito"
    
    def test_detection_correct_key(self):
        """Detection con chiave corretta deve funzionare."""
        from watermarkers.character_watermark import CharacterWatermarker
        
        wm = CharacterWatermarker(secret_key=12345, substitution_rate=0.3)
        
        text = "The quick brown fox jumps over the lazy dog."
        watermarked = wm.apply(text)
        detection = wm.detect(watermarked.text)
        
        # Con chiave corretta, match_ratio deve essere 100%
        assert detection.match_ratio == 1.0, f"Match ratio: {detection.match_ratio}"
    
    def test_detection_wrong_key(self):
        """Detection con chiave sbagliata deve avere match_ratio ~50%."""
        from watermarkers.character_watermark import CharacterWatermarker
        
        wm_apply = CharacterWatermarker(secret_key=12345, substitution_rate=0.3)
        wm_detect = CharacterWatermarker(secret_key=99999, substitution_rate=0.3)
        
        text = "The quick brown fox jumps over the lazy dog."
        watermarked = wm_apply.apply(text)
        detection = wm_detect.detect(watermarked.text)
        
        # Con chiave sbagliata, match_ratio dovrebbe essere circa 0.5 (random)
        assert 0.3 < detection.match_ratio < 0.7, f"Match ratio anomalo: {detection.match_ratio}"
    
    def test_visual_identity(self):
        """Il testo watermarked deve sembrare identico visivamente."""
        from watermarkers.character_watermark import CharacterWatermarker, normalize_text
        
        wm = CharacterWatermarker(secret_key=12345, substitution_rate=0.3)
        
        text = "Hello world"
        watermarked = wm.apply(text)
        
        # Dopo normalizzazione, deve tornare all'originale
        normalized = normalize_text(watermarked.text)
        assert normalized == text, f"Normalizzazione fallita: '{normalized}' != '{text}'"


# TEST MULTILAYER
class TestMultiLayer:
    
    def test_available_tasks(self):
        """Deve avere i task configurati."""
        from watermarkers.multilayer import MultiLayerWatermarker
        
        system = MultiLayerWatermarker()
        tasks = system.get_available_tasks()
        
        assert 'qa' in tasks
        assert 'summary' in tasks
        assert 'news' in tasks
    
    def test_different_keys_per_task(self):
        """Ogni task deve avere una chiave diversa."""
        from watermarkers.multilayer import DEFAULT_TASK_CONFIGS
        
        keys = [config.key for config in DEFAULT_TASK_CONFIGS.values()]
        assert len(keys) == len(set(keys)), "Chiavi duplicate tra task!"
    
    def test_apply_posthoc(self):
        """Apply_posthoc deve modificare il testo."""
        from watermarkers.multilayer import MultiLayerWatermarker
        
        system = MultiLayerWatermarker()
        
        text = "The big problem is very important and difficult to solve."
        result = system.apply_posthoc(text, 'news')
        
        # News usa sia synonym che character
        assert 'synonym' in result.techniques_applied
        assert 'character' in result.techniques_applied
    
    def test_detection_identifies_task(self):
        """Detection deve identificare il task corretto."""
        from watermarkers.multilayer import MultiLayerWatermarker
        
        system = MultiLayerWatermarker()
        
        # Applica watermark per 'news'
        text = "The big problem is very important and difficult to solve."
        watermarked = system.apply_posthoc(text, 'news')
        
        # Rileva
        detection = system.detect_all_tasks(watermarked.text)
        
        # Deve identificare 'news' come best match
        assert detection.best_match == 'news', f"Best match: {detection.best_match}"
        
        # Confidence per 'news' deve essere la più alta
        news_conf = detection.task_results['news'].confidence
        qa_conf = detection.task_results['qa'].confidence
        summary_conf = detection.task_results['summary'].confidence
        
        assert news_conf > qa_conf, f"news={news_conf}, qa={qa_conf}"
        assert news_conf > summary_conf, f"news={news_conf}, summary={summary_conf}"


# TEST PROMPTS
class TestPrompts:
    
    def test_prompt_counts(self):
        """Deve avere 50 prompt per task."""
        from generation.prompts import get_prompt_stats
        
        stats = get_prompt_stats()
        
        assert stats['qa'] >= 50, f"QA ha solo {stats['qa']} prompt"
        assert stats['summary'] >= 50, f"Summary ha solo {stats['summary']} prompt"
        assert stats['news'] >= 50, f"News ha solo {stats['news']} prompt"
    
    def test_prompts_not_empty(self):
        """Nessun prompt deve essere vuoto."""
        from generation.prompts import get_all_prompts
        
        all_prompts = get_all_prompts()
        
        for task, prompts in all_prompts.items():
            for i, prompt in enumerate(prompts):
                assert len(prompt.strip()) > 0, f"Prompt vuoto: {task}[{i}]"


# TEST INTEGRAZIONE (RICHIEDE GPT-2)
@pytest.mark.slow
class TestIntegration:
    """Test di integrazione che richiedono il modello GPT-2."""
    
    def test_full_pipeline(self):
        """Test completo: generazione + detection."""
        try:
            from generation.generator import SelectiveWatermarkGenerator
            
            generator = SelectiveWatermarkGenerator(model_name='gpt2')
            
            # Genera
            result = generator.generate(
                prompt="What is AI?",
                task="qa",
                max_tokens=30,
                use_watermark=True
            )
            
            assert len(result.text) > len("What is AI?")
            assert result.tokens_generated > 0
            
            # Rileva
            detection = generator.watermarker.detect_all_tasks(
                result.text, 
                generator.tokenizer
            )
            
            # Dovrebbe rilevare watermark
            assert detection.is_watermarked or detection.best_confidence > 0.2
            
        except ImportError:
            pytest.skip("transformers non installato")


# MAIN
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
