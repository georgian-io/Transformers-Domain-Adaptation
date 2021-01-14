from typing import NewType, Sequence


Token = NewType("Token", str)
Document = NewType("Document", str)
Corpus = NewType("Corpus", Sequence[Document])
