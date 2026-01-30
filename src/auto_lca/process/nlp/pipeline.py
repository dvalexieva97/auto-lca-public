from auto_lca.process.nlp.config import NLPConfig
from auto_lca.process.nlp.document_rank import DocumentRanker
from auto_lca.process.nlp.topic import TopicExtractor
from auto_lca.shared.util import save_list_to_jsonl


class NLPipeline:
    def __init__(self) -> None:
        self.config = NLPConfig()  # TODO pass it on
        self.extractor = TopicExtractor(self.config)
        self.ranker = DocumentRanker(self.config)

    def process_documents(
        self,
        documents,
        folder=None,
        split_into_sentences=True,
        save_locally: bool = False,
    ):
        """
        Process documents to extract topics and rank them.
        Args:
            documents (list): List of documents to process.
            folder (str): Folder to save the output files.
        """
        topics_list = self.extractor.extract_topics(documents)

        if split_into_sentences:
            docs_to_rank, _ = self.ranker.tokenize_into_sentences(documents)
        else:
            docs_to_rank = documents

        # Encode documents once using the bi-encoder:
        embeddings = self.ranker.encode_documents(docs_to_rank)

        for topic in topics_list:
            representation = topic["representation"]  # TODO Enum
            topic["ranks"] = self.rank_topic(
                representation, docs_to_rank, doc_embeddings=embeddings
            )

        if save_locally:
            if not folder:
                folder = self.config.DEFAULT_OUTPUT_FOLDER

            save_list_to_jsonl(
                topics_list,
                filepath=f"{folder}/topics.jsonl",
            )

        return topics_list

    def rank_topic(self, topic_keywords, documents, doc_embeddings=None):

        query = self.ranker.build_query_from_topic(topic_keywords)
        print(f"Ranking documents for topic: {topic_keywords}")  # TODO Add logger

        # 1. Prerank documents using the bi-encoder for speed:
        top_docs = self.ranker.prerank_documents(query, documents, doc_embeddings)
        # 2. Rank the top documents using a cross-encoder:
        return self.ranker.rank_documents(
            query,
            top_docs,
            count_top_ranks=self.config.COUNT_TOP_RANKS,
        )
