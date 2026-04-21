import os
import json
import openai
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import time
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from collections import defaultdict, Counter
import re
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from pdb import set_trace
from tqdm import tqdm


# Label student code error
def error_label(problem, language, code, model='gpt-4o', temperature=0):
    openai.api_key = ''

    try:
        # Using code in the original pipeline
        user_prompt = f"Problem\n{problem}\n\nStudent code:\n{code}"

        system_content = """You are an expert Java programming instructor and automated code reviewer. Given a Java programming problem and a student's buggy code solution, your task is to identify all errors present in the code. There is at least one error in the code. Use concise and standardized label/taxonomy for each error. Make sure the error label is generalizable without problem specific description. Take the following error label examples as reference:
Syntax Error (Examples): Confusing assignment with equality, Unbalanced parentheses, Semicolon errors
Runtime Error (Examples): Uninitialized Variables, Parameter confusion, NullPointerExceptions
Logical Error (Examples): Off-by-one errors, Integer Division, Infinite Loops

Return a JSON object with this template:
{
  "errors": [
    {
      "Reasoning": "<one sentence explanation of the error in the code>",
      "Category": "Syntax | Runtime | Logical",
      "Label": "<error label>"
    }
  ]
}
"""

        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
        ],
            n=1
        )

        reply = response.choices[0].message.content.strip()
        reply_ls = json.loads(reply)

        return reply_ls

    except Exception as e:
        print(e)
        return None

def error_label_open_source(problem, language, code, model, device):
    try:
        # Using code in the original pipeline
        user_prompt = f"Problem\n{problem}\n\nStudent code:\n{code}"

        system_content = """You are an expert Java programming instructor and automated code reviewer. Given a Java programming problem and a student's buggy code solution, your task is to identify all errors present in the code. Use concise and standardized label/taxonomy for each error. Make sure the error label is generalizable without problem specific description. Take the following error label examples as reference:
Syntax Error (Examples): Confusing assignment with equality, Unbalanced parentheses, Semicolon errors
Runtime Error (Examples): Uninitialized Variables, Parameter confusion, NullPointerExceptions
Logical Error (Examples): Off-by-one errors, Integer Division, Infinite Loops

Return a JSON object with this template:
{
  "errors": [
    {
      "Reasoning": "<one sentence explanation of the error in the code>",
      "Category": "Syntax | Runtime | Logical",
      "Label": "<error label>"
    }
  ]
}
"""   
        messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
        ]

        pipe = pipeline("text-generation", model=model, temperature=0.1, device=device)
        output = pipe(messages)
        generated_output = output[0]['generated_text'][-1]['content'].strip()
        output_res = json.loads(generated_output)
      
        return output_res

    except Exception as e:
        print(e)
        return None


def error_cluster_linkage(error_list, encoder_name, n_cluster):
    encoder = SentenceTransformer(encoder_name)
    error_emb = encoder.encode(error_list, convert_to_numpy=True)

    distance_matrix = pdist(error_emb, metric='cosine')
    Z = linkage(distance_matrix, method='average')

    labels = fcluster(Z, t=n_cluster, criterion='maxclust')

    cluster_to_kcs = defaultdict(list)
    for kc_text, cluster_id in zip(error_list, labels):
        cluster_to_kcs[cluster_id].append(kc_text)


    return cluster_to_kcs


def error_summarize(error_dict, model='gpt-4o', temperature=0, n=1):
    kc_summarized, map_dict = {}, {}
    cnt = 0

    track_same_dict = defaultdict(list)

    for key, val in error_dict.items():
        if len(val) > 1:
            kc_name = error_cluster_summarize(val, model, temperature, n)
        else:
            kc_name = val[0].lower()

        if kc_name:
            kc_summarized[key] = kc_name

            for kc_i in val:
                map_dict[kc_i] = kc_name

            track_same_dict[kc_name].append(val)
            
            print(f'Processed: {cnt}')

        else:
            print(f'Invalid KC summarization: {cnt}')
        
        cnt += 1

    kc_cnt = sum([len(val[0]) for val in track_same_dict.values()])
    print('Error Count:', kc_cnt)
    print('Unique Error Count:', len(track_same_dict))

    # set_trace()
    # for idx, (key, val) in enumerate(track_same_dict.items()):
    #     print(f"{idx}: {key}")
    #     for item in val:
    #         print(item)
    #         print('-------------------------')

    print('Done')


    return kc_summarized, map_dict

def error_cluster_summarize(error_list, model='gpt-4o-mini', temperature=0, n=1):
    openai.api_key = ''

    try:
        user_prompt = f"The error list is:\n{error_list}" + "\n\nNow follow the instructions in system message. First, examine the list carefully to understand their shared meaning. Second, explicitly reason about the error. Third, based on the reasoning, either select one error that best represents the group if they share the meaning or summarize your analysis into one clear and concise phrase that accurately captures the essence of this cluster."

        system_content = """You are an experienced computer science teacher. You will be provided with a list of errors from student code in Java that may vary in wording but sometimes refer to the same or related underlying errors.

Your task is to:
1. Carefully examine all the errors in the list to ensure none are overlooked.
2. Reason explicitly whether the errors collectively refer to the same underlying concept, or if they are related but represent distinct or complementary aspects of a broader theme.
3. Based on your reasoning:
    - If the errors refer to the same concept, select one error from the list that best represents the group — choose the one that is most clearly worded, generalizable, and inclusive of the others.
    - If the errors are related but too distinct to be represented by a single error, create a brief and meaningful summary name that captures the broader theme shared by the errors. The summary name must not contain the word “and”.

Return your output strictly in the following JSON format:
{
  "reasoning": "...",        // Exactly one sentence explaining your reasoning
  "representative_error": "..." , // Selected error if applicable, otherwise null
  "summary_name": "..." ,       // Summary name if representative error not chosen, otherwise null
}
"""
    
        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
            n=n
        )

        reply = response.choices[0].message.content.strip()
        kc = json.loads(reply)

        representative = kc['representative_error'] if kc['representative_error'] is not None else kc['summary_name']

        return representative.lower()

    except Exception as e:
        print(e)
        return ''


def error_cluster_alg(error_list, encoder_name):
  encoder = SentenceTransformer(encoder_name)

  error_emb = encoder.encode(error_list, convert_to_numpy=True)

  cosine_dis_matrix = cosine_distances(error_emb)

  
  clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')
  labels = clusterer.fit_predict(cosine_dis_matrix.astype('float64'))

  df = pd.DataFrame({'Error': error_list, 'label': labels, 'embedding': error_emb.tolist()})
  errors = df.groupby('label')

  centroids = {}
  error_dict = {}
  set_trace()

  for name, group in errors:
      cluster_no = group['label'].unique()[0]
      cluster = list(group['Error'])
      print(cluster)


      # if cluster_no != -1:
      #     emb_sub = list(group['embedding'])
      #     emb_sub = np.array(emb_sub)
      #     centroid = emb_sub.mean(axis=0)
      #     centroids[cluster_no] = centroid
      #     kc_dict[cluster_no] = cluster


def LLM_as_Judge(problem, code, error_ls, model='o4-mini'):
    openai.api_key = ''

    try:
        user_prompt = f"Problem\n{problem}\n\nStudent code:\n{code}\n\nThe error list is:\n{error_ls}" + "\n\nNow follow the instructions in system message, select all errors from the list that are present in the student code."

        system_content = """You are an experienced code reviewer. You will be provided with a programming problem along with a student code and a list of errors. 

Your task is to:
1. Carefully examine the student code and all the errors in the list to ensure none are overlooked.
2. Reason explicitly which errors from the list are included in the student code and return all that apply based on your reasoning.
3. Return an empty list if none of the errors are present in the code or the code is correct.

Return your output strictly in the following JSON format:
{
  "reasoning": "...",        // Exactly one sentence explaining your reasoning
  "included errors": [<error 1>, <error 2>, ...] , // Errors appear in the student code from the error list
}
"""
    
        
        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {'role': 'user', 'content': user_prompt},
            ],
            n=1
        )

        reply = response.choices[0].message.content.strip()
        errors = json.loads(reply)

        return errors

    except Exception as e:
        print(e)
        return {}



def main():
  # check error label dataset
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  total_df = pd.read_pickle('data/data_inc_error_label_full.pkl')
  
  length = total_df['error_labels'].apply(lambda x: len(x))
  print(np.mean(length))

  filt = length == 0
  filt_sub = total_df[filt]
  print(len(filt_sub))

  # set_trace()
  # for idx, row in filt_sub.iterrows():
  #     problem = row['prompt']
  #     code = row['Code']
  #     error_label_open_source(problem, 'Java', code, "deepseek-ai/deepseek-coder-33b-instruct", device)

  valid_rows = total_df[~filt]
  print(len(valid_rows))

  # error_category_counts = valid_rows['error_labels'].explode().str[0].value_counts() 

  nunique = valid_rows['error_labels'].explode().unique() #[(type, error)]

  errors = [item[1] for item in nunique]
  clean_errors = [re.sub(r'(?<!^)(?=[A-Z])', ' ', s) for s in errors]
  clean_errors = [re.sub(r"\s+", " ", s) for s in clean_errors]
  clean_errors = [s.lower().strip() for s in clean_errors]


  print(len(set(clean_errors)))
  error_ls = list(set(clean_errors))

  check = [item for item in error_ls if 'typo' in item]


  error_cluster_linkage(error_ls, "all-mpnet-base-v2", 50)

  
  # set_trace()
  # print('Done')
  


if __name__ == "__main__":
  main()
