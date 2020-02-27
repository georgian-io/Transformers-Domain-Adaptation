from pathlib import Path


def cat(src: Path, dst: Path, suffix='.txt'):
    files = src.glob(f'*{suffix}')
    with open(dst, 'w') as out_file:
        for file in files:
            with open(file) as in_file:
                for line in in_file:
                    out_file.write(line)
