import os
import openai
import pandas as pd
from chroma_retriever import ChromaRetriever
from omegaconf import OmegaConf
from templates import system_provider_template, test_template

def get_response_with_retrieval(df):

        retrieved_contexts = []
        responses = []

        for question in df["question"]:
                print(f"Questions: {question}")
                docs = chroma_retriever.max_marginal_relevance_search(question)
                page_contents = [doc.page_content for doc in docs]
                
                print(page_contents)
                print("="*20,"docs","="*20)
                retrieved_contexts.append(page_contents)

                system_template = system_provider_template
                system_template_content = system_template.format(context=page_contents, input=question)
                print(system_template_content)
                messages = [ {"role":"user", "content":system_template_content} ]

                #response = llm.invoke(system_template_content)
                response = client.chat.completions.create(model="gpt-4o", 
                                                          messages = messages )
                result = response.choices[0].message.content
                print(response)
                responses.append(result)

        df["retrieved_context"] = retrieved_contexts
        df["final_response"] = responses
        print(df.head())
        return df



if __name__ == "__main__":
    df = pd.read_csv(test_file_path, sep ='\t')
    client = openai.OpenAI()

    config = OmegaConf.load(yamlfile)
    chroma_retriever = ChromaRetriever(config)

    result = get_response_with_retrieval(df)
    result.to_csv(output_path,sep='\t',index=False)
