from pathlib import Path
from setuptools import setup, find_packages


HERE = Path(__file__).parent
README = (HERE / "README.md").read_text()
with open("README.md", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name="transformers-domain-adaptation",
    version="0.3.0",
    description="Adapt Transformer-based language models to new text domains",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/georgianpartners/Transformers-Domain-Adaptation",
    author="Christopher Tee",
    author_email="chris@georgian.io",
    license="MIT",
    python_requires=">=3.6",
    keywords=[
        "transformers",
        "tokenizers",
        "huggingface",
        "pytorch",
        "domain-adaptation",
        "transfer-learning",
        "natural-language-processing",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: General",
    ],
    package_dir={"": "src"},
    packages=find_packages(
        where="src", exclude=("etl*", "utils*", "experimental*", "tests")
    ),
    install_requires=[
        "transformers>=4,<5",
        "tokenizers>=0.9,<0.10",
        "datasets>=1.2,<1.3",
        "pandas",
        "torch>=1.7,<1.8",
        "scipy==1.5.4",
        "scikit-learn",
        "tqdm",
    ],
)
