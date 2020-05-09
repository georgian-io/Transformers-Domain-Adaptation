# Usage
This document details the usage of each of the following four steps in Domain Adaptation.

1. Data Selection (DS)
2. Vocabulary Augmentation (VA)
3. Domain Pre-Training (DPT)
4. Fine-Tuning (FT)

---

## 1. Data Selection (DS)
Not all documents in a (large) domain pre-training (DPT) corpus are necessarily beneficial to improving the performance of BERT on a downstream fine-tuning task. Hence, selecting a subset of the DPT corpus brings about two benefits:
1. Domain pre-training on the selected subset of data instead of the full DPT corpus leads to better downstream task performance
2. It may lead to shorter training times

Data selection is chosen based on a linear combination of similarity and diversity metrics found from [*Learning to select data for transfer learning with Bayesian Optimization*](https://arxiv.org/abs/1707.05246).


| Input | Output |
| ----- | ------ |
| DPT Corpus (txt file) | Subset of corpus containing selected documents (txt file) |
| BERT vocabulary (txt file) | |
| FT Corpus (txt file) — Required for Similar Data Select only | |

### Usage
Select data randomly
```
python -m scripts.domain_adaptation.select_data \
    --corpus <path-to-corpus-txt-file> \
    --dst <output-directory> \
    random \
    --pct 0.02 \
    --seed 42
```

Select data based on a single similarity metric. Valid similarity functions are `['jensen-shannon, 'renyi', 'cosine', 'euclidean', 'variational', 'bhattacharyya']`.
```
python -m scripts.domain_adaptation.select_data \
    --corpus <path-to-corpus-txt-file> \
    --dst <output-directory> \
    similar \
    --sim-func <similarity-function> \
    --fine-tune-text <path-to-fine-tuning-corpus-txt-file> \
    --vocab-file bert-base-uncased-vocab.txt
```

Select data based on a single diversity metric. Valid diversity functions are `['num_word_types', 'type_token_ratio', 'entropy, 'simpsons_index', 'renyi_entropy']`.
```
python -m scripts.domain_adaptation.select_data \
    --corpus <path-to-corpus-txt-file> \
    --dst <output-directory> \
    diverse \
    --div-func <diversity-function>
```

Select data based on multiple metrics
```
##### To-Be-Implemented #####
```

---

## 2. Vocabulary Augmentation (VA)
The original BERT vocabulary comes with `[unused*]` placeholder tokens. These placeholder tokens can be replaced with in-domain terms so that meaningful representations of these terms can be learnt during domain pre-training.

| Input | Output |
| ----- | ------ |
| Original BERT Vocabulary (txt file) | Augmented BERT Vocabulary (txt file) |
| DPT Corpus (txt file) | |

### Usage
If the corpus is large, there will be a performance speedup by first sharding it so shards can be read in parallel during Vocabulary Augmentation
```
python -m src.etl.shard \
    --src <path-to-corpus-file> \
    --dst <output-directory> \
    --shard-size 100000
```
where `--shard-size` is the number of lines in each shard.

Augment the BERT vocabulary
```
python -m scripts.domain_adaptation.augment_vocab \
    --bert-vocab bert-base-uncased-vocab.txt \
    --corpus <path-to-corpus-file-or-folder-of-corpus-shards> \
    --dst <output-directory> \
    --vocab-size 30522
```

---

## 3. Domain Pre-Training (DPT)
BERT is pre-trained on general English corpora. Domain pre-training involves continuing the pre-training phase of BERT, on in-domain data, in a similar unsupervised fashion. This allows BERT to pick up the semantics of in-domain terminology.

| Input | Output |
| ----- | ------ |
| DPT Training Corpus (txt file) | Domain Pre-Trained BERT model folder |
| DPT Eval Corpus (txt file) [Optional] | |


### Usage
There are a large number of parameters in this script. Here is a sample usage that should fit most purposes.
```
python -m scripts.domain_adaptation.domain_pre_train \
    --output_dir <output-directory> \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --tokenizer_name <path-to-(augmented)-bert-vocab> \
    --block_size 512 \
    --num_train_epochs <num-epochs> \
    --do_train \
    --train_data_file <path-to-training-corpus> \
    --per_gpu_train_batch_size <per-gpu-train-batch-size> \
    --mlm \
    --do_eval \
    --eval_data_file <path-to-eval-corpus> \
    --per_gpu_eval_batch_size <per-gpu-eval-batch-size> \
    --evaluate_during_training \
    --eval_all_checkpoints \
    --save_steps <checkpoint-frequency> \
    --should_continue
```

To train distributedly with multiple GPUs (using `torch.nn.parallel.DistributedDataParallel`)
```
python -m torch.distributed.launch --nproc_per_node <num-gpus> \
    ./scripts/domain_adaptation/domain_pre_train.py \
    ...  # Include same args from usage above
```
For an optional (read: unverified) performance boost, you could set the `OMP_NUM_THREADS` environmental variable to enable multi-processed batch creation for each GPU.
```
export OMP_NUM_THREADS=<num-cpus-per-gpu>
```

---

## 4. Fine-Tuning (FT)
BERT has to be fine-tuned before it can be used for downstream tasks such as sentiment analysis, sentence classification, etc. Currently, only name entity recognition and multi-label text classification is supported.


### Name Entity Recognition
| Input | Output |
| ----- | ------ |
| Domain pre-trained BERT model folder | Fine-tuned BERT model folder |
| Fine-tune training corpus (txt file) | |
| Fine-tune eval corpus (txt file) | |
| Fine-tune test corpus (txt file ) | |
| Fine-tune labels (txt file) | |

#### Usage
To fine-tune BERT for POS tagging
```
python -m scripts.domain_adaptation.fine_tune_ner \
    --data_dir <data-dir-containing-train-eval-and-test-FT-corpora> \
    --labels <FT-corpus-labels> \
    --output_dir <output-directory> \
    --model_type bert \
    --model_name_or_path <path-to-domain-pre-trained-bert-model-folder> \
    --do_lower_case \
    --max_seq_length <max-ner-length> \
    --do_train \
    --num_train_epochs <num-epochs> \
    --do_eval \
    --eval_all_checkpoints \
    --evaluate_during_training \
    --do_predict \
    --save_steps <num-save-steps>
```

### Multilabel Text Classification
| Input | Output |
| ----- | ------ |
| Domain pre-trained BERT model folder | Fine-tuned BERT model folder |
| Fine-tune training corpus (txt file) | |
| Fine-tune eval corpus (txt file) | |
| Fine-tune test corpus (txt file ) | |
| Fine-tune training labels (txt file ) | |
| Fine-tune eval labels (txt file ) | |
| Fine-tune test labels (txt file ) | |
| Universe of fine-tune labels (txt file) | |

#### Usage
To fine-tune BERT for POS tagging
```
python -m scripts.domain_adaptation.fine_tun_mltc \
    --data_dir <data-dir-containing-train-eval-and-test-FT-corpora> \
    --labels <FT-corpus-labels> \
    --output_dir <output-directory> \
    --model_name_or_path <path-to-domain-pre-trained-bert-model-folder> \
    --tokenizer_vocab <path-to-(augmented)-bert-vocab> \
    --truncation <truncation-strategy> \
    --do_lower_case \
    --do_train \
    --num_train_epochs <num-epochs> \
    --do_eval \
    --eval_all_checkpoints \
    --evaluate_during_training \
    --do_predict \
    --save_steps <num-save-steps>
```
Valid values for `<truncation-strategy>` include `'first'`, `'last'` and two-comma separate values adding up to 510 (e.g. `200,31` — use first 200 and last 310 tokens).


---

## Integrated Domain Adaptation
The Vocab Augmentation, Domain Pre-Training and Fine-Tuning steps have been integrated together in `domain_adaptation_pipeline.sh`.

`run_pipeline.sh` serves as a helper script on how to call `domain_adaptation_pipeline.sh`. Feel free to use and modify arguments in this script to suit your use case! 🚀
