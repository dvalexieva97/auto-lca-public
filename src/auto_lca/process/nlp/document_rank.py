import concurrent.futures
import os
import threading
from typing import List, Optional

import nltk
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from auto_lca.process.nlp.config import NLPConfig
from auto_lca.process.nlp.nltk_utils import stopwords

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# TODO Optimize tensor convert for cosine:
# /Users/dvalexieva/Documents/scripts/uni/auto-lca/auto-lca-env/lib/python3.13/site-packages/sentence_transformers/util.py:44: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/torch/csrc/utils/tensor_new.cpp:256.)


class DocumentRanker:
    # Class-level lock to ensure thread-safe model initialization
    _init_lock = threading.Lock()

    def __init__(self, config=None, device: Optional[str] = None) -> None:
        """
        Initialize the DocumentRanker.
        CrossEncoder is now lazy-loaded to avoid meta-device issues when not needed.
        """
        self.config = config or NLPConfig()
        self.device = device or os.environ.get("NLP_DEVICE", "cpu")

        # Defer model initialization; create on first use to avoid meta-device issues
        self._bi_encoder: Optional[SentenceTransformer] = None
        self._cross_encoder: Optional[CrossEncoder] = None
        # Instance-level lock for thread-safe lazy loading
        self._bi_encoder_lock = threading.Lock()
        self._cross_encoder_lock = threading.Lock()

    def _get_bi_encoder(self) -> SentenceTransformer:
        # Double-checked locking pattern for thread-safe lazy initialization
        if self._bi_encoder is None:
            with self._bi_encoder_lock:
                # Check again after acquiring lock (another thread might have initialized it)
                if self._bi_encoder is None:
                    self._bi_encoder = SentenceTransformer(
                        self.config.BI_ENCODER_MODEL,
                        device=self.device,
                        trust_remote_code=False,
                    )
        return self._bi_encoder

    def _get_cross_encoder(self) -> CrossEncoder:
        # Double-checked locking pattern for thread-safe lazy initialization
        if self._cross_encoder is None:
            with self._cross_encoder_lock:
                # Check again after acquiring lock (another thread might have initialized it)
                if self._cross_encoder is None:
                    self._cross_encoder = CrossEncoder(
                        self.config.CROSS_ENCODER_MODEL,
                        device=self.device,
                        trust_remote_code=False,
                    )
        return self._cross_encoder

    @classmethod
    def tokenize_into_sentences(
        cls, documents: list[str]
    ) -> tuple[list[str], list[tuple[int, str]]]:
        """
        Tokenize a list of documents into sentences using NLTK, in parallel.
        Returns:
            sentences: List of all sentences.
            mapping: List of (doc_index, sentence) tuples mapping each sentence to its document index.
        """

        def tokenize_doc(idx_doc):
            idx, doc = idx_doc
            sents = nltk.sent_tokenize(doc)
            return [(idx, sent) for sent in sents]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(tokenize_doc, enumerate(documents)))

        sentences = []
        mapping = []
        for sent_pairs in results:
            for idx, sent in sent_pairs:
                sentences.append(sent)
                mapping.append((idx, sent))
        return sentences, mapping

    def compute_cosine_similarity(self, doc1: str, doc2: str):
        """
        Computes cosine similarity between two documents
        Returns: cosine similarity
        """
        embeddings = self.encode_documents([doc1, doc2])
        # Convert to numpy array to avoid slow tensor creation from list of arrays
        emb_array = np.array(embeddings)
        # Use numpy for cosine similarity (faster than converting to tensor)
        emb1 = emb_array[0:1]  # Keep as 2D for broadcasting
        emb2 = emb_array[1:2]
        cosine_sim = np.dot(emb1, emb2.T) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2)
        )
        return float(cosine_sim[0][0])

    def compute_fuzzy_cosine_similarity(self, doc1: str, doc2: str):
        """Computes fuzzy cosine similarity by:
        - Splitting the input into words
        - Removing stopwords
        - Computing the average cosine between the full-word match
        and the maximum matched word
        Returns: fuzzy cosine or None in case both inputs are single-word
        """

        a_words = doc1.split()
        b_words = doc2.split()

        if len(a_words) == 1 and len(b_words) == 1:
            # TODO Log warning
            return None

        variants_a = [doc1] + list(set(a_words) - stopwords)
        variants_b = [doc2] + list(set(b_words) - stopwords)

        encoder = self._get_bi_encoder()
        embs_a = encoder.encode(variants_a, normalize_embeddings=True)
        embs_b = encoder.encode(variants_b, normalize_embeddings=True)

        # Convert to numpy arrays to avoid slow tensor creation
        embs_a = np.array(embs_a) if not isinstance(embs_a, np.ndarray) else embs_a
        embs_b = np.array(embs_b) if not isinstance(embs_b, np.ndarray) else embs_b

        # Use numpy for cosine similarity (already normalized, so just dot product)
        sims = np.dot(embs_a, embs_b.T)
        full_word = sims[0][0]
        best = sims.max().item()
        # Get the average of the full word
        # and the best performing match:
        return float(np.average([best, full_word]))

    def build_query_from_topic(
        self,
        keywords: list[str],
        separator: str = ", ",
    ) -> str:
        """
        Build a query string from a list of topic words.
        """
        return separator.join(keywords)

    def encode_documents(self, documents, normalize_embeddings=False):
        """
        Encode a list of documents using the bi-encoder.
        Args:
            documents (list): List of documents to encode.
        Returns:
            list: List of document embeddings.
        """
        encoder = self._get_bi_encoder()
        return encoder.encode(
            documents,
            normalize_embeddings=normalize_embeddings,
            convert_to_tensor=False,
        )

    def prerank_documents(
        self,
        query: str,
        documents: list,
        doc_embeddings: list = None,
        prefilter_k: int = 200,
    ) -> list:
        """
        Rank documents using bi-encoder prefiltering.
        Args:
            query (str): Query string.
            documents (list): List of documents to rank.
            prefilter_k (int): Number of top documents to return after bi-encoder filtering.
        Returns:
            list: List of top documents based on cosine similarity scores.
        """
        # print("Preranking documents with bi-encoder...")  # TODO Logs
        if len(documents) != len(doc_embeddings):
            raise ValueError(
                "The number of documents and document embeddings must match."
            )
        if len(documents) < prefilter_k:
            prefilter_k = len(documents)

        if doc_embeddings is None:
            doc_embeddings = self.encode_documents(documents)
        query_embedding = self.encode_documents([query])

        # Convert to numpy arrays to avoid slow tensor creation
        query_emb = np.array(query_embedding)
        doc_embs = np.array(doc_embeddings)

        # Compute cosine similarity using numpy (faster)
        # Since embeddings are normalized, cosine similarity is just dot product
        cosine_scores = np.dot(query_emb, doc_embs.T)[0]
        top_k_indices = np.argsort(cosine_scores)[-prefilter_k:][::-1].tolist()
        top_docs = [documents[i] for i in top_k_indices]
        return top_docs

    def rank_documents(
        self,
        query: str,
        documents: list,
        count_top_ranks: int = None,
        return_documents: bool = True,
    ) -> list:
        """
        Rank documents based on the query using the CrossEncoder model.
        Args:
            query (str): Query string.
            documents (list): List of documents to rank.
            split_into_sentences (bool): Whether to split documents into sentences.
            count_top_ranks (int): Number of top ranks to return.
            If none, return all documents ranked.
            return_documents (bool): Whether to return the original documents.
        Returns:
            list: List of ranked document dicts with their scores and rank.
        """

        ranks = self._get_cross_encoder().rank(
            query, documents, return_documents=return_documents
        )

        top_ranks = ranks[:count_top_ranks] if count_top_ranks else ranks
        # TODO Add Ranking data model
        top_ranks = [
            {
                "query": query,
                "rank": i + 1,
                "score": float(rank["score"]),
                "text": rank["text"],
            }
            for i, rank in enumerate(top_ranks)
        ]
        return top_ranks

    def _rank_with_prerank(
        self, query, documents, count_top_ranks, doc_embeddings=None
    ):
        # 1. Prerank documents using the bi-encoder for speed:
        top_docs = self.prerank_documents(query, documents, doc_embeddings)
        # 2. Rank the top documents using a cross-encoder:
        return self.rank_documents(
            query,
            top_docs,
            count_top_ranks=count_top_ranks,
        )

    def rank_documents_bulk_queries(
        self,
        queries: List[str],
        documents: list,
        count_top_ranks: int = None,
    ):
        embeddings = self.encode_documents(documents)
        results = {}
        for query in queries:
            ranks = self._rank_with_prerank(
                query,
                documents,
                count_top_ranks=count_top_ranks,
                doc_embeddings=embeddings,
            )
            results[query] = ranks
        return results
