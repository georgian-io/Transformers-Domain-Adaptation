# NLP Domain Adaptation
Experimental work investigating the effects of domain-adapting a Transformers language model (e.g. BERT) to improve downstream NLP task performance.

## Domain Adaptation
This overall process of domain adaptation can be broken down into three phases:
1. Data Selection - Select a subset of relevant documents from the in-domain corpus
2. Vocabulary augmentation — Replace unused BERT tokens with new terms from the in-domain corpus
3. Domain pre-training — Continue language model training of BERT on corpus to learn linguistic nuances of domain
4. Fine-tuning — Attach classification head onto domain pre-trained BERT and train model on NLP dataset

## Domains
In this research, domain adaptation will be applied to three disparate English domains:
1. Biology
2. Law
3. News

The data can be found on `s3://nlp-domain-adaptation/domains`


## Installation
Install dependencies (on a virtual environment) using pip:
```
pip install -r requirements.txt
```

## Usage
Refer to [USAGE.md](USAGE.md).
