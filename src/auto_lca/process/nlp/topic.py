import os
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP

from auto_lca.models.enum import TopicExtractorBackend
from auto_lca.process.nlp.config import NLPConfig
from auto_lca.shared.util import save_list_to_jsonl


class TopicExtractor:
    def __init__(self, config=None):
        """
        Initialize the TopicExtractor with a configuration.
        Args:
            config (NLPConfig): Configuration object containing parameters for topic extraction.
        """
        if not config:
            self.config = NLPConfig()
        else:
            self.config = config

        umap_model = UMAP(
            n_neighbors=self.config.N_NEIGHBORS,
            n_components=self.config.N_COMPONENTS,
            min_dist=self.config.MIN_DIST,
            metric=self.config.SIMILARITY_METRIC,
        )
        embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        self.topic_model = BERTopic(
            embedding_model=embedding_model, umap_model=umap_model
        )

    def extract_topics(self, documents, backend=None) -> list:
        """
        Extract topics from the given documents using the specified backend.
        Args:
            documents (List[str]): List of documents
            backend (str): Backend to use for topic extraction.
        """
        if not backend:
            backend = self.config.BACKEND
        if backend == TopicExtractorBackend.BERTopic.value:
            _, topic_info_df = self.extract_topics_with_topic_model(documents)
            topic_info_df = topic_info_df.rename(
                columns={
                    "Topic": "topic_id",
                    "Count": "count",
                    "Representation": "representation",
                }
            ).drop(columns=["Name", "Representative_Docs"])
            topics = topic_info_df.to_dict(orient="records")
        elif backend == TopicExtractorBackend.TFIDF.value:
            topics = self.extract_topics_with_tfidf(documents)
        else:
            raise ValueError(
                f"Invalid Topic Modelling backend specified: {backend}. "
                f"Use 'bertopic' or 'tfidf'."  # TODO List as Enum
            )
        return topics

    def extract_topics_with_tfidf(self, documents) -> list:
        """
        Extract topics from the given documents using TF-IDF and KMeans clustering.
        Args:
            documents (List[str]): List of abstracts or titles+abstracts.
        Returns:
            topics (List[int]): Topic assignment for each document.
            topic_info (pd.DataFrame): DataFrame with topic details.
        # TODO: Add automatic language detection and pass on config
        """

        vectorizer = TfidfVectorizer(
            max_df=self.config.MAX_DF,
            min_df=self.config.MIN_DF,
            stop_words=self.config.DEFAULT_STOP_WORDS_LANGUAGE,
            ngram_range=self.config.N_GRAM_RANGE,
        )
        X = vectorizer.fit_transform(documents)

        # Step 2: Cluster the TF-IDF vectors using KMeans
        num_clusters = 10
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        kmeans.fit(X)

        # Step 3: Extract top keywords per cluster
        terms = vectorizer.get_feature_names_out()
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

        topic_keywords = []
        for i in range(num_clusters):
            top_terms = [terms[ind] for ind in order_centroids[i, :10]]
            topic_keywords.append(
                {
                    "topic_id": i,
                    "representation": top_terms,
                    "count": np.sum(kmeans.labels_ == i),
                }
            )

        topic_df = pd.DataFrame(topic_keywords).sort_values(by="count", ascending=False)
        return topic_df.to_dict(orient="records")

    def extract_topics_with_topic_model(self, documents) -> Tuple[list, pd.DataFrame]:
        """
        Extract topics from the given documents using BERTopic.
        Args:
            documents (List[str]): List of abstracts or titles+abstracts.
        Returns:
            topics (List[int]): Topic assignment for each document.
            topic_info (pd.DataFrame): DataFrame with topic details.
        """
        document_topics, _ = self.topic_model.fit_transform(documents)
        topic_info = self.topic_model.get_topic_info()

        return document_topics, topic_info

    def extract_topics_and_save_list_to_jsonl(
        self,
        documents,
        backend: TopicExtractorBackend = None,
        filepath: str = "topics.jsonl",
        default_folder: str = "output/tests/",
    ):
        """
        Extract topics and save the topic information to a JSONL file.
        Args:
            documents (List[str]): List of abstracts or titles+abstracts.
            filepath (str): Path to save the topic information.
        """
        if not backend:
            backend = self.config.BACKEND
        if not filepath:
            timestamp = str(datetime.now())[:19].replace(" ", "_").replace(":", "-")
            filepath = os.path.join(default_folder, f"topics_{timestamp}.jsonl")
        topics = self.extract_topics(documents, backend=backend)
        save_list_to_jsonl(topics, filepath=filepath)
        return topics
