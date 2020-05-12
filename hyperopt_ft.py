import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from hyperopt import hp, fmin, tpe, space_eval

from src.utils.shell import run_shell


def define_search_space():
    space = [
        hp.quniform('max_seq_length', 16, 512, 1),
        hp.quniform('num_train_epochs', 3, 20, 1),
        hp.loguniform('learning_rate', -13.815510558, -6.907755279),  # Log of [1e-6, 1e-3]
        hp.loguniform('weight_decay', -1e10, -6.907755279),  # Log of [0, 1e-3]
        hp.quniform('warmup_steps', 0, 5000, 1)
    ]
    return space


def objective(params):
    max_seq_length, num_train_epochs, learning_rate, weight_decay, warmup_steps = params
    max_seq_length = int(max_seq_length)
    num_train_epochs = int(num_train_epochs)
    warmup_steps = int(warmup_steps)
    now = datetime.now().strftime(r'%Y-%m-%d_%H:%M:%S')

    command = f"python -m scripts.domain_adaptation.fine_tune_ner \
        --data_dir data/law/tasks/eurlex57k \
        --labels data/law/tasks/eurlex57k/all_labels.txt \
        --output_dir results/hyperopt/2pct_similar/{now} \
        --model_type bert \
        --model_name_or_path bert-base-uncased \
        --do_lower_case \
        --max_seq_length {max_seq_length} \
        --do_train \
        --num_train_epochs {num_train_epochs} \
        --do_eval \
        --eval_all_checkpoints \
        --do_predict \
        --save_steps 1000 \
        --learning_rate {learning_rate} \
        --weight_decay {weight_decay} \
        --warmup_steps {warmup_steps} \
        "

    run_shell(command)
    metrics = pd.read_csv(f'results/hyperopt/2pct_similar/{now}/eval_results.txt',
                          sep=' = ', names=['metrics', 'values'], index_col=0)
    return -metrics.at['end_f1', 'values']


def optimize():
    # Curry function while maintaining function metadata
    # global objective
    # objective = wraps(objective)(partial(objective, args=args))

    best = fmin(objective, define_search_space(),
                algo=tpe.suggest, max_evals=10)
    print(best)


if __name__ == '__main__':
    # args = parse_args()
    # create_csv_header(args.store)
    optimize()
