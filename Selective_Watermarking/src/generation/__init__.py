"""
Generation Package

Moduli per la generazione di testo con watermark selettivo.

Modules:
    prompts: Collezione di 150 prompt (50 per task)
    generator: Generatore di testo watermarked
"""

from .prompts import (
    QA_PROMPTS,
    SUMMARY_PROMPTS,
    NEWS_PROMPTS,
    get_prompts_for_task,
    get_all_prompts,
    get_prompt_stats
)

from .generator import (
    SelectiveWatermarkGenerator,
    GenerationResult,
    create_generator
)

__all__ = [
    # Prompts
    'QA_PROMPTS',
    'SUMMARY_PROMPTS', 
    'NEWS_PROMPTS',
    'get_prompts_for_task',
    'get_all_prompts',
    'get_prompt_stats',
    # Generator
    'SelectiveWatermarkGenerator',
    'GenerationResult',
    'create_generator'
]
