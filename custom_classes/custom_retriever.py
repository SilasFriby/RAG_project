from llama_index.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.indices.vector_store.retrievers import VectorIndexAutoRetriever
from llama_index import QueryBundle
from llama_index.schema import NodeWithScore
from typing import List

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        meta_filter_retreiver: VectorIndexAutoRetriever,
        mode: str = "AND",
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        self._meta_filter_retreiver = meta_filter_retreiver
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)
        meta_filter_nodes = self._meta_filter_retreiver.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes}
        keyword_ids = {n.node.node_id for n in keyword_nodes}
        meta_filter_ids = {n.node.node_id for n in meta_filter_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})
        combined_dict.update({n.node.node_id: n for n in meta_filter_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids).intersection(meta_filter_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids).union(meta_filter_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes