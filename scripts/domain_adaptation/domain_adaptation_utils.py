"""Module containing utility functions for domain adaptation."""
import shutil
from pathlib import Path


def copy_files(src: Path, dst: Path) -> None:
    src, dst = Path(src), Path(dst)
    dst.mkdir(exist_ok=True, parents=True)
    files = ('pytorch_model.bin', 'config.json',
             'tokenizer_config.json', 'special_tokens_map.json',
             'vocab.txt')
    for file in files:
        shutil.copyfile(str(src / file), str(dst / file))
