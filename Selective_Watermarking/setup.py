#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="selective_watermarking",
    version="1.0.0",
    description="Selective Watermarking for AI-Generated Text",
    author="AISE Project",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "nltk>=3.8.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "demo": ["gradio>=4.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "watermark-test=scripts.test_crypto_watermark:main",
            "watermark-generate=scripts.generate_dataset:main",
            "watermark-evaluate=scripts.run_evaluation:main",
        ],
    },
)
