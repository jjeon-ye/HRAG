import os
import openai
import argparse
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Document
from omegaconf import OmegaConf
from chardet.universaldetector import UniversalDetector

class CustomDocument(Document):
    def __init__(self, text: str, id_: str, metadata: dict = None):
        super().__init__(text=text, metadata=metadata)
        self.id_ = id_

def create_parser():
    parser = argparse.ArgumentParser(description='demo how to use ai embeddings to chat.')
    parser.add_argument("-y", "--yaml", dest="yamlfile",
                        help="Yaml file for project", metavar="YAML")
    return parser

def detect_encoding(file_path):
    """
    Detect the encoding of a text file using chardet.
    """
    detector = UniversalDetector()
    with open(file_path, 'rb') as f:
        for line in f:
            detector.feed(line)
            if detector.done:
                break
    detector.close()
    return detector.result['encoding']

def extract_drug_name(file_name):
    """
    Extracts the drug name from the file name using a predefined pattern.
    Assumes that drug names are words starting with an uppercase letter.
    """
    # Regex to extract words starting with an uppercase letter (assumed to be drug names)
    matches = file_name.split('_')

    if matches:
        return matches[0]
    return "Unknown"


def load_documents(pdf_directory):
    all_documents = []
    document_id = 1

    for file_name in os.listdir(pdf_directory):
        file_path = os.path.join(pdf_directory, file_name)

        if file_name.endswith(".pdf"):
            print(f"file name: {file_name}")

            loader = PyPDFLoader(file_path)
            drug_name = extract_drug_name(file_name)
            documents = loader.load()

            for doc in documents:
                doc.metadata["drug"] = drug_name
                custom_doc = CustomDocument(
                    text=doc.page_content,
                    id_=str(document_id),
                    metadata={
                        "source": file_path,
                        "page": doc.metadata.get("page", 0),
                        "drug": drug_name,
                        "gene": "",
                        "document_type": "drug label" if "FDA" in file_name else "guideline",
                        "summary": f"Prescription information for {drug_name}" if "FDA" in file_name else f"Guidelines considering genetic factors for the use of {drug_name}",
                    }
                )
                document_id += 1

                all_documents.append(custom_doc)

        elif file_name.endswith(".tsv"):


            df = pd.read_csv(file_path, delimiter="\t")
            drug_name = extract_drug_name(file_name)

            for _, row in df.iterrows():
                doc_text = " ".join(map(str, row.values))

                custom_doc = CustomDocument(
                    text=doc_text,
                    id_=str(document_id),
                    metadata={
                        "source": file_path,
                        "drug": drug_name,
                        "gene": "",
                        "document_type": "CDS alert" if "Alert" in file_name else "Genotype-phenotype",
                        "summary": "Drug label",
                    }
                )
                document_id += 1

                all_documents.append(custom_doc)

        elif file_name.endswith(".txt"):
            print(f"file name: {file_name}")

            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load_and_split()

            gene_name = extract_drug_name(file_name)
            for doc in documents:
                custom_doc = CustomDocument(
                        text=doc.page_content,
                        id_=str(document_id),
                        metadata={
                            "source": file_path,
                            "page": doc.metadata.get("page", 0),
                            "drug":"",
                            "gene": gene_name,
                            "document_type": "TBD",
                            "summary": "allele",
                        }
                )
                document_id += 1

                all_documents.append(custom_doc)


    return all_documents
