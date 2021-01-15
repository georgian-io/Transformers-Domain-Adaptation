from pathlib import Path
from setuptools import setup, find_packages


HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="transformers-domain-adaptation",
    version="0.3.0a1",
    description="Adapt Transformer-based language models to new text domains",
    url="https://github.com/georgianpartners/NLP-Domain-Adaptation",
    author="Christopher Tee",
    author_email="chris@georgian.io",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    package_dir={"": "src"},
    packages=find_packages(where='src', exclude=("etl*", "utils*", "experimental*", "tests")),
    install_requires=[
        "transformers>=4,<5",
        "tokenizers>=0.9,<0.10",
        "datasets>=1.2,<1.3",
        "pandas",
        "torch>=1.7,<1.8",
        "scipy==1.5.4",
        "scikit-learn",
        "tqdm",
    ]
)
