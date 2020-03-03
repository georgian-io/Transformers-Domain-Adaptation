import itertools as it
from typing import Any, Iterable, TypeVar


A = TypeVar('Any')


def batch(iterable: Iterable[A], size: int) -> Iterable[A]:
    iters = iter(iterable)
    def take():
        while True:
            yield tuple(it.islice(iters, size))
    return iter(take().__next__, tuple())
