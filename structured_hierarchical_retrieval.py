import os
import json
from typing import List
from upsert import load_documents
from llama_index.core import StorageContext, load_index_from_storage, TreeIndex
from llama_index.core.node_parser.relational.hierarchical import HierarchicalNodeParser
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding
from sklearn.metrics.pairwise import cosine_similarity

class TreeIndexManager:

    def __init__(self, guidelines: TreeIndex, gene: TreeIndex, embed_model):
        self.guidelines_tree_index = guidelines
        self.gene_tree_index = gene
        self.embed_model = embed_model

    def compute_similarity(self, query_text, node_texts):
        all_texts = [query_text] + node_texts
        embeddings = self.embed_model.get_text_embedding_batch(all_texts)

        query_embedding = embeddings[0] 
        node_embeddings = embeddings[1:]

        similarity_scores = cosine_similarity([query_embedding], node_embeddings)[0]
        return similarity_scores

    def query_tree_index(self, query_text: str, top_k: int):

        engine_guidelines = self.guidelines_tree_index.as_query_engine(similarity_top_k=top_k)
        response_guidelines = engine_guidelines.query(query_text)

        engine_gene = self.gene_tree_index.as_query_engine(similarity_top_k=top_k)
        response_gene = engine_gene.query(query_text)

        results_guidelines = response_guidelines.source_nodes
        results_gene = response_gene.source_nodes

        combined_results = results_guidelines + results_gene

        if not combined_results:
            return []

        node_texts = [node.node.get_text() for node in combined_results]

        similarity_scores = self.compute_similarity(query_text, node_texts)

        scored_results = [(node, score) for node, score in zip(combined_results, similarity_scores)]
        sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

        return sorted_results[:top_k]


def create_hierarchical_nodes(documents: List[Document]) -> List[Document]:
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512])
    return parser.get_nodes_from_documents(documents, show_progress=True)

def create_tree_index(nodes: List[Document], num_children: int = 10) -> TreeIndex:
    embed_model = OpenAIEmbedding(
      model="text-embedding-ada-002",
      # openai_api_key="API_KEY" 
    )

    return TreeIndex(
        nodes=nodes,
        num_children=num_children,
        build_tree=True,
        show_progress=True,
        similarity_top_k=3,
        embed_model=embed_model
    )

def tree_search(query_text, top_k):
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        # openai_api_key="API_KEY" 
    )

    cpic_guidelines_context = StorageContext.from_defaults(persist_dir="./index/cpic_guidelines_tree_index_tables")
    cpic_gene_context = StorageContext.from_defaults(persist_dir="./index/cpic_gene_tree_index_tables")

    cpic_guidelines_tree_index = load_index_from_storage(cpic_guidelines_context, embed_model=embed_model)
    cpic_gene_tree_index = load_index_from_storage(cpic_gene_context, embed_model=embed_model)

    cpic_tree_manager = TreeIndexManager(cpic_guidelines_tree_index, cpic_gene_tree_index, embed_model=embed_model)

    dpwg_storage_context = StorageContext.from_defaults(persist_dir="./index/dpwg_tree_index_tables")
    dpwg_tree_index = load_index_from_storage(dpwg_storage_context, embed_model=embed_model)

    cpic_best_results = cpic_tree_manager.query_tree_index(query_text, top_k=top_k)
    dpwg_qengine = dpwg_tree_index.as_query_engine(similarity_top_k=top_k)
    dpwg_results = dpwg_qengine.query(query_text).source_nodes

    cpic_result_texts = [(node.get_text(), score) for node, score in cpic_best_results]
    dpwg_result_texts = [(doc.node.get_text(), doc.score) for doc in dpwg_results]

    combined_results = {
        "cpic": cpic_result_texts,
        "dpwg": dpwg_result_texts
    }

    return combined_results

def query_tree_index(tree_index, query_text, top_k=3):
    retriever = tree_index.as_retriever(similarity_top_k=top_k)
    results = retriever.retrieve(query_text)
    return results

def get_max_depth_from_graph(node_id, children_map, current_depth=1):
    children = children_map.get(node_id, [])
    if not children:
        return current_depth
    return max(get_max_depth_from_graph(child_id, children_map, current_depth + 1) for child_id in children)

def get_structural_info(tree_index, label):
    index_struct = tree_index.index_struct
    root_nodes = index_struct.root_nodes
    children_map = index_struct.node_id_to_children_ids

    print(label)
    print(f"Number of trees (root nodes): {len(root_nodes)}")
    print(f"Root node IDs: {root_nodes}")

    for idx, root_id in root_nodes.items():
        children = children_map.get(root_id, [])
        max_depth = get_max_depth_from_graph(root_id, children_map)
        print(f"Root node {idx} ({root_id}):")
        print(f"  - Immediate children: {len(children)}")
        print(f"  - Max depth of tree: {max_depth}")

def construct_trees():

    # guildelines
    source_dir = "./data/guidelines"
    documents = load_documents(source_dir)

    cpic_documents = [doc for doc in documents if "cpic" in doc.metadata["source"].lower()]
    dpwg_documents = [doc for doc in documents if "dpwg" in doc.metadata["source"].lower()]

    cpic_nodes = create_hierarchical_nodes(cpic_documents)
    dpwg_nodes = create_hierarchical_nodes(dpwg_documents)

    cpic_guidelines_tree_index = create_tree_index(cpic_nodes)
    dpwg_guidelines_tree_index = create_tree_index(dpwg_nodes)

    # cpic gene(function, frequency)
    source_dir = "./data/allele_func_freq"
    documents = load_documents(source_dir)

    cpic_gene_nodes = create_hierarchical_nodes(documents)
    cpic_gene_tree_index = create_tree_index(cpic_gene_nodes)

    # store tree index
    cpic_guidelines_tree_index.storage_context.persist(persist_dir="./index/cpic_guidelines_tree_index_tables")
    dpwg_guidelines_tree_index.storage_context.persist(persist_dir="./index/dpwg_tree_index_tables")
    cpic_gene_tree_index.storage_context.persist(persist_dir="./index/cpic_gene_tree_index_tables")

    cpic_tree_manager = TreeIndexManager(cpic_guidelines_tree_index, cpic_gene_tree_index)

    return cpic_tree_manager, dpwg_guidelines_tree_index

if __name__ == "__main__":
    construct_trees()
