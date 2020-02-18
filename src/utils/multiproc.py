from functools import partial
from typing import Optional, Dict, Any
from multiprocessing import cpu_count, Pool

from tqdm import tqdm


def parallelize(func,
                iterable,
                length: Optional[int] = None,
                n_workers: Optional[int] = None,
                desc: Optional[str] = None,
                chunksize: Optional[int] = None,
                **func_kwargs,
               ):
    workers = (cpu_count() - 1) if n_workers is None else n_workers
    chunksize = 1 if chunksize is None else chunksize
    total = len(iterable) if length is None else length
    func = partial(func, **func_kwargs)

    with Pool(workers) as p:
        return list(tqdm(p.imap(func, iterable, chunksize=chunksize),
                         desc=desc, dynamic_ncols=True,
                         total=total))
