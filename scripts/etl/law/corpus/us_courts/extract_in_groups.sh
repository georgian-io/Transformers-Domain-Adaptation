#!/bin/zsh

# This allows for glob exclusions
setopt extendedglob

declare -A PATTERNS
PATTERNS["a"]="a"
PATTERNS["b"]="b"
PATTERNS["ca"]="ca"
PATTERNS["c^a"]="c_not_a"
PATTERNS["[de]"]="d_and_e"
PATTERNS["f"]="f"
PATTERNS["[gh]"]="g_and_e"
PATTERNS["i"]="i"
PATTERNS["[jk]"]="j_and_k"
PATTERNS["l"]="l"
PATTERNS["m"]="m"
PATTERNS["n"]="n"
PATTERNS["o"]="o"
PATTERNS["p"]="p"
PATTERNS["[rs]"]="r_and_s"
PATTERNS["t"]="t"
PATTERNS["[uvw]"]="u_v_and_w"

for key val in ${(kv)PATTERNS[@]}; do
    key="${key%\"}"
    key="${key#\"}"

    val="${val%\"}"
    val="${val#\"}"

    python -m scripts.etl.law.corpus.us_courts.2_extract_text \
        --src "data/law/corpus/us_courts/unzipped/$key*" \
        --dst "data/law/corpus/us_courts/corpus/$val"
done
