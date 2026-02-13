#!/usr/bin/env python3
"""Test Synonym Watermark"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.watermarkers.synonym_watermark import SynonymWatermarker

print("=" * 60)
print("TEST: Synonym Watermark")
print("=" * 60)

wm = SynonymWatermarker(secret_key=12345)

text = 'The quick brown fox jumps over the lazy dog. This is a simple test.'
result = wm.apply(text)
print(f'\nOriginale:   {text}')
print(f'Watermarked: {result.text}')
print(f'Sostituzioni: {result.substitutions_made}')

det = wm.detect(result.text)
print(f'\nDetect chiave CORRETTA: detected={det.detected}, match={det.match_ratio:.2%}')

wm_wrong = SynonymWatermarker(secret_key=99999)
det_wrong = wm_wrong.detect(result.text)
print(f'Detect chiave SBAGLIATA: detected={det_wrong.detected}, match={det_wrong.match_ratio:.2%}')

print("\n" + "=" * 60)
if det.match_ratio > 0.9 and det_wrong.match_ratio < 0.6:
    print("TEST PASSED")
else:
    print("TEST FAILED")
print("=" * 60)
