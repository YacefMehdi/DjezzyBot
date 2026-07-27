"""
indexer.py — Chunking + e5 embeddings + FAISS IndexFlatIP.

Turns scraped pages into a searchable vector store. Two rules from the spec are
enforced here and nowhere else, so they can't drift:

  1. multilingual-e5 prefix convention:
        documents are embedded with "passage: " (build time)
        queries   are embedded with "query: "   (retrieval time, via embed_query)
  2. vectors are L2-normalized, so a FAISS IndexFlatIP (inner product) computes
     cosine similarity.

A single shared SentenceTransformer is loaded once (lazily) and used for both
documents and queries, guaranteeing the two paths share the exact same model.

Public API
----------
    get_embedder()                  -> SentenceTransformer (cached)
    chunk_documents(pages)          -> list[Document]
    build_index(pages)              -> FAISS            (also persists to FAISS_DIR)
    load_index()                    -> FAISS | None
    embed_query(text)               -> np.ndarray       ("query: " + normalize)

`Document` is langchain_core.documents.Document with metadata
{source_url, page_title}.
"""

import os
import logging

import config

logger = logging.getLogger("djezzybot.indexer")

# LangChain's FAISS decides how to embed a QUERY by checking `isinstance(emb,
# Embeddings)`; if our adapter isn't a real subclass it falls back to CALLING the
# object (→ "'_E5Embeddings' object is not callable"). So we subclass the base.
# Imported with a fallback so the pure-logic modules still import where LangChain
# isn't installed (local tests); on Colab the real base is used.
try:
    from langchain_core.embeddings import Embeddings as _LCEmbeddings
except Exception:
    _LCEmbeddings = object

_embedder = None  # cached SentenceTransformer


# ===========================================================================
# Embedder (shared by documents and queries)
# ===========================================================================
def get_embedder():
    """Load (once) and return the multilingual-e5-base SentenceTransformer."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("loading embedder %s", config.EMBED_MODEL_ID)
        _embedder = SentenceTransformer(config.EMBED_MODEL_ID)
    return _embedder


class _E5Embeddings(_LCEmbeddings):
    """LangChain-compatible embeddings adapter that applies the e5 prefixes.

    LangChain's FAISS wrapper calls embed_documents()/embed_query(); we make those
    prepend the correct prefix and normalize, so the rest of the codebase never
    has to remember the convention.
    """

    def __init__(self):
        self._model = get_embedder()

    def _encode(self, texts):
        return self._model.encode(
            texts,
            normalize_embeddings=config.EMBED_NORMALIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_documents(self, texts):
        prefixed = [config.E5_PASSAGE_PREFIX + t for t in texts]
        return self._encode(prefixed).tolist()

    def embed_query(self, text):
        prefixed = config.E5_QUERY_PREFIX + text
        return self._encode([prefixed])[0].tolist()


# ===========================================================================
# Chunking
# ===========================================================================
def chunk_documents(pages: list) -> list:
    """Split each page into ~1200-char chunks (120 overlap), never crossing pages.

    Splitting is done per page (each page handled independently), so a chunk can
    never mix content from two different URLs. Metadata carried on every chunk:
    source_url, page_title.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs = []
    for page in pages:
        content = page.get("content", "")
        if not content.strip():
            continue
        meta = {
            "source_url": page.get("url", ""),
            "page_title": page.get("title", ""),
        }
        for chunk in splitter.split_text(content):  # per-page → no cross-page chunks
            docs.append(Document(page_content=chunk, metadata=dict(meta)))

    logger.info("chunk_documents: %d pages -> %d chunks", len(pages), len(docs))
    return docs


# ===========================================================================
# Build / load FAISS
# ===========================================================================
def build_index(pages: list):
    """Chunk `pages`, embed with the passage prefix, build + persist a FAISS index.

    Uses langchain_community.vectorstores.FAISS, whose default index for cosine
    on normalized vectors is inner-product (IndexFlatIP). Persisted to FAISS_DIR.
    Returns the in-memory FAISS store.
    """
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.vectorstores.utils import DistanceStrategy
    except ImportError:
        logger.warning("build_index: langchain_community not installed")
        return None

    docs = chunk_documents(pages)
    if not docs:
        raise ValueError("build_index: no chunks produced (empty/short pages?)")

    embeddings = _E5Embeddings()
    store = FAISS.from_documents(
        docs,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # IndexFlatIP
    )
    os.makedirs(config.FAISS_DIR, exist_ok=True)
    store.save_local(config.FAISS_DIR)
    logger.info("build_index: %d vectors saved to %s", len(docs), config.FAISS_DIR)
    return store


def load_index():
    """Load the persisted FAISS store, or None if it hasn't been built yet."""
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        logger.warning("load_index: langchain_community not installed")
        return None

    if not os.path.exists(os.path.join(config.FAISS_DIR, "index.faiss")):
        logger.warning("load_index: no index at %s", config.FAISS_DIR)
        return None
    embeddings = _E5Embeddings()
    return FAISS.load_local(
        config.FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True,  # our own local pickle, trusted
    )


# ===========================================================================
# Query embedding (retrieval side)
# ===========================================================================
def embed_query(text: str):
    """Embed a search string with the "query: " prefix and normalization.

    Returns a 1-D numpy array. The retriever uses the FAISS store's own search
    methods (which call _E5Embeddings.embed_query internally), but this is exposed
    for any direct vector math or debugging.
    """
    import numpy as np
    model = get_embedder()
    vec = model.encode(
        config.E5_QUERY_PREFIX + text,
        normalize_embeddings=config.EMBED_NORMALIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vec)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from scraper import load_pages
    pages = load_pages()
    if not pages:
        print("No scraped pages found — run scraper.py first.")
    else:
        store = build_index(pages)
        # Pass the RAW query — the store's embed_query adds the "query: " prefix.
        hits = store.similarity_search("forfait internet pas cher", k=3)
        for h in hits:
            print("-", h.metadata.get("source_url"), "|", h.page_content[:80])
