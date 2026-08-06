"""Shared paths, constants and loaders for every Slot A stage.

Imported by 01_index.py, verify_stage1.py, and every later stage. Nothing in here
retrieves, reranks, scores or reports — it is only the things that must be
IDENTICAL across stages, because the moment two stages disagree about (say) how a
document's text is assembled, every number downstream is quietly wrong.

Three invariants live here and nowhere else:
  1. `doc_text()`      — how title+abstract become one string (both arms use it)
  2. `sorted_doc_ids()`— the canonical doc order, which is the FAISS ordinal space
  3. `embed_*()`       — dtype, normalization and instruction handling
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths. Override the root with SLOTA_ROOT so the same code runs from a Colab
# Drive mount (/content/drive/MyDrive/slotA-rag-harness) or a local checkout.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(os.environ.get("SLOTA_ROOT", Path(__file__).resolve().parent.parent))

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACT_DIR / "data"        # SciFact download cache (gitignored)
INDEX_DIR = ARTIFACT_DIR / "index"
RUNS_DIR = ARTIFACT_DIR / "runs"
NOTES_DIR = PROJECT_ROOT / "notes"

FAISS_PATH = INDEX_DIR / "faiss.index"
BM25_PATH = INDEX_DIR / "bm25.pkl"
DOCMAP_PATH = INDEX_DIR / "docmap.json"
META_PATH = INDEX_DIR / "index_meta.json"

STAGE1_ARTIFACTS = (FAISS_PATH, BM25_PATH, DOCMAP_PATH, META_PATH)

# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SPLIT = "test"

# Published BEIR SciFact stats — used as a sanity check, never as a source of truth.
EXPECTED_N_DOCS = 5183
EXPECTED_N_TEST_QUERIES = 300

# ---------------------------------------------------------------------------
# Dense embedder
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_DIM = 768                 # asserted at build time, never assumed
MAX_SEQ_LENGTH = 512            # bge-base is BERT-based; includes [CLS]/[SEP]
EMBED_DTYPE = "float16"         # T4/sm_75: fp16 only, never bf16
ATTN_IMPLEMENTATION = "sdpa"    # never flash_attention_2 (needs sm_80+)
DEFAULT_BATCH_SIZE = 64

# bge-en-v1.5's retrieval instruction. Applied to QUERIES ONLY, at search time.
# Passages are embedded raw — this is bge's own convention and is NOT the e5
# "query:"/"passage:" prefix scheme that reference repo A uses for e5-large-v2.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Fraction of docs allowed to exceed MAX_SEQ_LENGTH before we abandon
# one-vector-per-doc and chunk instead. See notes/stage1-index.md for the
# reasoning; the point of a threshold is that chunking imposes doc-level dedup
# on every later stage, which is a real complexity cost to pay for a rounding error.
TRUNCATION_TOLERANCE = 0.01

# BM25
BM25_TOKENIZER_NOTE = "str.lower().split() — no stemming, no stopword removal, punctuation retained"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def scifact_dir() -> Path:
    return DATA_DIR / "scifact"


def load_scifact(split: str = SCIFACT_SPLIT, force_download: bool = False):
    """Load SciFact via BEIR. Returns (corpus, queries, qrels).

    corpus : {docid: {"title": str, "text": str}}
    queries: {qid: str}                       (already filtered to `split`)
    qrels  : {qid: {docid: int}}              ground truth, the reason SciFact was chosen

    Uses the cache under artifacts/data/ when present, so re-runs and later
    stages need no network.
    """
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = scifact_dir()

    if force_download or not (target / "corpus.jsonl").exists():
        data_path = Path(util.download_and_unzip(SCIFACT_URL, str(DATA_DIR)))
    else:
        data_path = target

    corpus, queries, qrels = GenericDataLoader(data_folder=str(data_path)).load(split=split)
    return corpus, queries, qrels


def doc_text(doc: dict) -> str:
    """The single canonical text for a document — used by BOTH arms.

    Concatenation is `title + " " + text`, whitespace-stripped. This is the BEIR
    baseline convention. Both the dense embedder and BM25 see exactly this string,
    so the two arms are compared on identical input rather than on a preprocessing
    difference we forgot about.
    """
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    return f"{title} {text}".strip()


def sorted_doc_ids(corpus: dict) -> list[str]:
    """Canonical, reproducible doc order — this IS the FAISS ordinal space.

    Numeric-aware so SciFact's integer-like docids sort 2, 10, 11 rather than
    10, 11, 2. Sorting rather than trusting dict insertion order means the
    ordinal <-> docid mapping is reproducible across runs and machines.
    """
    ids = list(corpus.keys())
    if all(i.isdigit() for i in ids):
        return sorted(ids, key=int)
    return sorted(ids)


DOCID_ORDER_NOTE = "sorted numerically when all docids are digit strings, else lexicographically"
TEXT_FIELD_NOTE = 'doc_text() = (title + " " + text).strip()'


# ---------------------------------------------------------------------------
# Dense embedder
# ---------------------------------------------------------------------------
def load_embedder(device: str | None = None):
    """bge-base-en-v1.5 in fp16 with SDPA attention, per methodology §3."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs: dict[str, Any] = {"attn_implementation": ATTN_IMPLEMENTATION}
    if device == "cuda":
        # fp16 on CPU is pathologically slow and partly unimplemented — only cast on GPU.
        model_kwargs["torch_dtype"] = torch.float16

    model = SentenceTransformer(EMBED_MODEL, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


def _finalize(vectors) -> "Any":
    """fp16 encode output -> float32, C-contiguous, L2-normalized in place.

    Order matters. We cast to float32 BEFORE normalizing so the norm is computed
    at full precision, and because FAISS refuses anything but contiguous float32.
    After this, inner product == cosine similarity, which is the whole reason
    IndexFlatIP is the right index.
    """
    import faiss
    import numpy as np

    out = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    faiss.normalize_L2(out)
    return out


NORMALIZATION_NOTE = "cast fp16->float32, then faiss.normalize_L2 in place (IP == cosine)"


def embed_passages(model, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE, progress: bool = True):
    """Embed passages RAW — no instruction, no prefix (bge convention)."""
    raw = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,   # we normalize in float32 in _finalize()
        show_progress_bar=progress,
    )
    return _finalize(raw)


def embed_queries(model, queries: list[str], use_instruction: bool = True, batch_size: int = DEFAULT_BATCH_SIZE):
    """Embed queries, prepending bge's retrieval instruction by default.

    Asymmetric by design: instruction on queries, nothing on passages. Whether the
    instruction actually helps on SciFact is a Stage 2 question — this function
    just makes sure Stage 1's smoke check and Stage 2 use one code path, so a
    convention mismatch cannot appear between them.
    """
    prepared = [BGE_QUERY_INSTRUCTION + q for q in queries] if use_instruction else list(queries)
    raw = model.encode(
        prepared,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )
    return _finalize(raw)


# ---------------------------------------------------------------------------
# Persisted-index access — what every later stage calls instead of rebuilding
# ---------------------------------------------------------------------------
@dataclass
class DocMap:
    """FAISS row ordinal <-> docid <-> raw text.

    ordinal_to_docid[i] is the docid of FAISS row i. docid_to_ordinals is a list
    per docid so the chunked policy needs no schema change (one_vector_per_doc
    simply has exactly one ordinal each).
    """

    chunk_policy: str
    dedup_needed: bool
    n_units: int
    ordinal_to_docid: list[str]
    docid_to_ordinals: dict[str, list[int]]
    docid_to_text: dict[str, str]
    # The exact string embedded at each ordinal. Under one_vector_per_doc this is
    # the doc text; under `chunked` it is the decoded chunk. Stored because the
    # alignment proof in verify_stage1.py has to re-embed precisely what was
    # embedded — without it that check cannot run at all under chunking.
    ordinal_to_text: list[str] | None = None

    def docid(self, ordinal: int) -> str:
        return self.ordinal_to_docid[ordinal]

    def unit_text(self, ordinal: int) -> str | None:
        """The exact text embedded at `ordinal`, or None for a pre-patch docmap."""
        if self.ordinal_to_text is None:
            return None
        return self.ordinal_to_text[ordinal]

    def ordinals(self, docid: str) -> list[int]:
        return self.docid_to_ordinals[docid]

    def text(self, docid: str) -> str:
        return self.docid_to_text[docid]

    @property
    def doc_ids(self) -> list[str]:
        return list(self.docid_to_ordinals.keys())


def load_docmap(path: Path = DOCMAP_PATH) -> DocMap:
    with open(path) as f:
        raw = json.load(f)
    return DocMap(
        chunk_policy=raw["chunk_policy"],
        dedup_needed=raw["dedup_needed"],
        n_units=raw["n_units"],
        ordinal_to_docid=raw["ordinal_to_docid"],
        docid_to_ordinals=raw["docid_to_ordinals"],
        docid_to_text=raw["docid_to_text"],
        ordinal_to_text=raw.get("ordinal_to_text"),  # absent in pre-patch docmaps
    )


def load_faiss(path: Path = FAISS_PATH):
    import faiss

    return faiss.read_index(str(path))


def load_bm25(path: Path = BM25_PATH) -> dict:
    """Returns the pickled bundle: {"bm25", "doc_ids", "tokenizer_note", ...}.

    BM25 is ALWAYS doc-level and never chunked, so `doc_ids` is the score-array
    ordering: bm25.get_scores(tokens)[j] is the score of doc_ids[j].
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def load_meta(path: Path = META_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def bm25_tokenize(text: str) -> list[str]:
    """Repo A's convention, kept deliberately (see notes/repo-read.md §5)."""
    return text.lower().split()


def stage1_artifacts_present() -> bool:
    return all(p.exists() for p in STAGE1_ARTIFACTS)


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for dist in (
        "torch",
        "transformers",
        "sentence-transformers",
        "faiss-cpu",
        "rank-bm25",
        "beir",
        "numpy",
    ):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = "NOT INSTALLED"
    return out
