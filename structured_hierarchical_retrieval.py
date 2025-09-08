from typing import List
from upsert import load_documents
from llama_index.core import StorageContext, load_index_from_storage, TreeIndex
from llama_index.core.node_parser.relational.hierarchical import HierarchicalNodeParser
from llama_index.core import Document
from llama_index.embeddings.openai import OpenAIEmbedding


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

def Construct_trees():

    # guildeline
    source_dir = "./data"
    documents = load_documents(source_dir)

    cpic_documents = [doc for doc in documents if "cpic" in doc.metadata["source"].lower()]
    cpic_nodes = create_hierarchical_nodes(cpic_documents)
    cpic_guidelines_tree_index = create_tree_index(cpic_nodes)
    cpic_guidelines_tree_index.storage_context.persist(persist_dir="./index/cpic_tree_index_sample")

    return cpic_guidelines_tree_index

def tree_search(query_text, top_k):
    embed_model = OpenAIEmbedding(
        model="text-embedding-ada-002",
        # openai_api_key="API_KEY" 
    )

    cpic_storage_context = StorageContext.from_defaults(persist_dir="./index/cpic_tree_index_sample")
    cpic_tree_index = load_index_from_storage(dpwg_storage_context, embed_model=embed_model)

    cpic_qengine = dpwg_tree_index.as_query_engine(similarity_top_k=top_k)
    cpic_results = dpwg_qengine.query(query_text).source_nodes

    cpic_result_texts = [(doc.node.get_text(), doc.score) for doc in cpic_results]

    dpwg_storage_context = StorageContext.from_defaults(persist_dir="./index/dpwg_tree_index_sample")
    dpwg_tree_index = load_index_from_storage(dpwg_storage_context, embed_model=embed_model)

    dpwg_qengine = dpwg_tree_index.as_query_engine(similarity_top_k=top_k)
    dpwg_results = dpwg_qengine.query(query_text).source_nodes

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


if __name__ == "__main__":
    Construct_trees()
