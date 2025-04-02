# -*- coding:utf-8 -*-
# Created by liwenw at 9/11/23

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Type
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
# from langchain.vectorstores import Chroma
from omegaconf import OmegaConf
from chromadb.config import Settings
import chromadb


import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ChromaRetriever:
    def __init__(self, config):
        self.config = config

    def get_vector_store(self):
        embeddings = OpenAIEmbeddings(openai_api_key="API_KEY")
        persistent_client = chromadb.PersistentClient(path=self.config.chromadb.persist_directory)
        collection_name = self.config.chromadb.collection_name
        # persist_directory = self.config.chromadb.persist_directory
        # chroma_db_impl = self.config.chromadb.chroma_db_impl
        vector_store = Chroma(collection_name=collection_name,
                              embedding_function=embeddings,
                                client=persistent_client,
                              )
        
        return vector_store

    def max_marginal_relevance_search(self, query, k=10, lambda_mult=0.9):
        vector_store = self.get_vector_store()
        return vector_store.max_marginal_relevance_search(query, k=k, lambda_mult=lambda_mult)

    def similarity_search(self, query):
        vector_store = self.get_vector_store()
        return vector_store.similarity_search(query)
    
    

    def query_collection(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Query the chroma collection."""
        vector_store = self.get_vector_store()
        return vector_store.__query_collection(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            **kwargs,
        )


if __name__ == "__main__":
   



