from typing import List, Optional, Union


def _clean_args(tokenized, max_length, keep, cls_token, sep_token):
    # Ensure `tokenized` is a nested list
    if not isinstance(tokenized[0], list):
        tokenized = [tokenized]

    # Check `keep`
    STRATEGIES = ("first", "last")
    if isinstance(keep, str) and keep not in STRATEGIES:
        raise ValueError(f"Invalid `keep` strategy {keep} provided.")
    elif isinstance(keep, list):
        if len(keep) != 2:
            raise ValueError(
                "Expecting a `keep` list to be of length 2 but "
                f"got {len(keep)} instead."
            )
        if any([(x <= 0 or not isinstance(x, int)) for x in keep]):
            raise ValueError("Elements in `keep` should be positive integers.")
        if sum(keep) != max_length:
            raise ValueError(
                "`keep` list should sum to 510 but got " f"{sum(keep)} instead."
            )

    # Assign `cls_token` and `sep_token` if not provided
    DEFAULTS = {"cls": {str: "[CLS]", int: 101}, "sep": {str: "[SEP]", int: 102}}
    if cls_token is None:
        cls_token = DEFAULTS["cls"][type(tokenized[0][0])]
    if sep_token is None:
        sep_token = DEFAULTS["sep"][type(tokenized[0][0])]

    # Check `cls_token` and `sep_token`
    if not (
        type(cls_token) == type(sep_token) and type(tokenized[0][0]) == type(sep_token)
    ):
        raise ValueError(
            "`tokenized`, `cls_token` and `sep_token` have to be " "of the same type."
        )

    # Strip possible special tokens in `tokenized`
    tokenized_cleaned = []
    for _tokenized in tokenized:
        if _tokenized[0] == cls_token:
            _tokenized = _tokenized[1:]
        if _tokenized[-1] == sep_token:
            _tokenized = _tokenized[:-1]
        tokenized_cleaned.append(_tokenized)

    return tokenized_cleaned, max_length, keep, cls_token, sep_token


def truncate(
    tokenized: Union[List[str], List[List[str]], List[int], List[List[int]]],
    max_length: int = 510,
    keep: str = "first",
    cls_token: Optional[Union[str, int]] = None,
    sep_token: Optional[Union[str, int]] = None,
) -> Union[List[str], List[List[str]], List[int], List[List[int]]]:
    # Perform appropriate cleaning and checking of args
    # `tokenized` has been stripped of possible special chars
    tokenized, max_length, keep, cls_token, sep_token = _clean_args(
        tokenized, max_length, keep, cls_token, sep_token
    )

    ret = []
    for _tokenized in tokenized:
        if len(_tokenized) > max_length:
            if keep == "first":
                _tokenized = _tokenized[:max_length]
            elif keep == "last":
                _tokenized = _tokenized[-max_length:]
            elif isinstance(keep, list):
                _tokenized = _tokenized[: keep[0]] + _tokenized[-keep[1] :]
        ret.append([cls_token] + _tokenized + [sep_token])

    return ret if len(ret) > 1 else ret[0]
