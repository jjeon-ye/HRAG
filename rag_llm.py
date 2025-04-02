import os
import openai
import pandas as pd
from templates import system_provider_template, test_template
from structured_hierarchical_retrieval import tree_search

def get_response_with_retrieval(df):

        retrieved_contexts = []
        responses = []

        for question in df["question"]:
                print(f"Questions: {question}")
                docs = tree_search(question, top_k=2)
                
                print(docs)
                print("="*20,"docs","="*20)
                retrieved_contexts.append(docs)

                system_template = system_provider_template
                system_template_content = system_template.format(context=docs, input=question)
                print(type(system_template_content))
                messages = [ {"role":"user", "content":system_template_content} ]
                              
                       
                
                #response = llm.invoke(system_template_content)
                response = client.chat.completions.create(model="gpt-4o", 
                                                          messages = messages , temperature=0)
                result = response.choices[0].message.content
                print(response)
                print(result)
                
                responses.append(result)

        df["retrieved_context"] = retrieved_contexts
        df["final_response"] = responses
        print(df.head())
        return df

if __name__ == "__main__":
    df = pd.read_csv(test_file_path, sep ='\t')
    client = openai.OpenAI()

    result = get_response_with_retrieval(df)
    result.to_csv(output_path,sep='\t',index=False)
