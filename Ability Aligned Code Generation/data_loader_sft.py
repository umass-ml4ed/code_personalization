import pandas as pd
import json
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from pdb import set_trace
from model import *
import random
import math
from torch.utils.data import Dataset
from error_labeling import *

def read_data(file, configs):
    df = pd.read_pickle(file)
    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)
    df["Code"] = df["Code"].str.replace(r"/\*.*?\*/", "", regex=True).str.replace(r"/\*[\s\S]*?\*/", "", regex=True).str.replace(r"//.*", "", regex=True).str.strip()

    # # split correct and incorrect submissions for error labeling
    # incorrect_sub = df[df['Score'] < 1]

    # subsets = np.array_split(incorrect_sub, 6)

    # for i in range(0, 6):
    #     error_labels = []
    #     subdf = subsets[i].copy()
    #     start = time.perf_counter()

    #     for idx, row in tqdm(subdf.iterrows()):
    #         set_trace()
    #         print(row['prompt'])
    #         print(row['Code'])
        

    #         output = error_label(row['prompt'], 'java', row['Code'])
            
    #         if output:
    #             error_ls = [(item_i['Category'], item_i['Label']) for item_i in output['errors']]
    #             error_labels.append(error_ls)

    #         else:
    #             error_labels.append([])
        
       
    #     elapsed = time.perf_counter() - start
    #     # just in case
    #     with open(f'data/error_labels_{i}.pkl', 'wb') as f:
    #         pickle.dump(error_labels, f)

    #     subdf['error_labels'] = error_labels
    #     subdf.to_pickle(f'data/subset_incorrect_with_error_labels_{i}.pkl')
    
    #     print(f"Time taken for subset {i}: {elapsed} seconds")
    #     print('Finished labeling subset {}'.format(i))
    # set_trace()

    # Decide final score format
    if configs.label_type == 'binary':
        if "Score" in df.columns:
            df['Score'] = np.where(df["Score"] == 1, 1, 0)
        else:
            df['Score'] = np.where(df["Score_x"] == 1, 1, 0)
    else:
        df['Score'] = df['Score']

    df = df.drop_duplicates(subset=['SubjectID', 'ProblemID'],keep='first').reset_index(drop=True)


    if 'SubjectID_appendix' not in df.columns:

        # Merge compiler results dataset for error distribution
        error_data = pd.read_csv('data/data_compile_res.csv')

        df = df.merge(error_data[['SubjectID', 'ProblemID', 'compilable', 'error_types']], on=['SubjectID', 'ProblemID'], how='left')

        df['compilable'].fillna(1, inplace=True)

        prev_subject_id = 0
        subjectid_appendix = []
        timesteps = []
        max_len = 50
        for i in tqdm(range(len(df)), desc="splitting students' records ..."):
            if prev_subject_id != df.iloc[i].SubjectID:
                # when encountering a new student ID
                prev_subject_id = df.iloc[i].SubjectID
                accumulated = 0
                id_appendix = 1
            else:
                accumulated += 1
                if accumulated >= max_len:
                    id_appendix += 1
                    accumulated = 0
            timesteps.append(accumulated)
            subjectid_appendix.append(id_appendix)
        df['timestep'] = timesteps
        df['SubjectID_appendix'] = subjectid_appendix
        df['SubjectID'] = [df.iloc[i].SubjectID + '_{}'.format(df.iloc[i].SubjectID_appendix) for i in range(len(df))]

    # Split by problems
    if configs.split_by == 'problem':
        uniq_ls = df['ProblemID'].unique()
        train_set, test_set = train_test_split(uniq_ls, test_size=configs.test_size, random_state=configs.seed)
        valid_set, test_set = train_test_split(test_set, test_size=0.5, random_state=configs.seed)

    else:
        # Split by students
        uniq_ls = df['SubjectID'].unique()
        train_set, test_set = train_test_split(uniq_ls, test_size=configs.test_size, random_state=configs.seed)
        valid_set, test_set = train_test_split(test_set, test_size=0.5, random_state=configs.seed)

    return train_set, valid_set, test_set, df, uniq_ls

def get_inputs(df):
    problem_ls = df['prompt'].tolist()
    code_ls = df['Code'].tolist()

    return problem_ls, code_ls

def get_inputs_ability(df):
    problem_ls = df['prompt'].tolist()
    code_ls = df['Code'].tolist()
    ability_ls = df['kc_level'].tolist()
    kc_ls = df['knowledge_component'].tolist()

    return problem_ls, code_ls, ability_ls, kc_ls


def build_prompt_with_special_tokens(prompt, tokenizer):
    system_content = "You are an LLM that simulates a human student writing Java code. For the given problem, respond the way such a student realistically would: sometimes producing correct code, sometimes making mistakes that students are likely to make. Output code only without any explanations or comments. Do not wrap the code in markdown fences (no ```)."
    user_prompt = f"Problem: {prompt} Student written code:"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )
   
    return prompt

def build_input_with_special_tokens(prompt, code, tokenizer):
    system_content = "You are an LLM that simulates a human student writing Java code. For the given problem, respond the way such a student realistically would: sometimes producing correct code, sometimes making mistakes that students are likely to make. Output code only without any explanations or comments. Do not wrap the code in markdown fences (no ```)."
    user_prompt = f"Problem: {prompt} Student written code:"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": code.strip()},
    ]

    training_str = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=False, 
        tokenize=False
    )

    return training_str


def tokenize(tokenizer, completions, train=True):
    if train:
        completions_tokenized = tokenizer(completions, truncation=True, padding=True, max_length=900, return_tensors='pt')

        inputs_ids, attention_mask = completions_tokenized['input_ids'], completions_tokenized['attention_mask']
        delimiter_token_id = tokenizer.convert_tokens_to_ids("assistant")

        inputs_ids[:, -1] = tokenizer.eos_token_id
        delimiter_indices = torch.where(inputs_ids == delimiter_token_id, 1, 0)

        prompt_id_lens = torch.argmax(delimiter_indices, dim=-1)
        prompt_id_lens = torch.add(prompt_id_lens, 2) # adding 2 since delimiter token is assistand and there is \n following it

        labels = inputs_ids.clone()
        labels = labels.masked_fill((attention_mask == 0), -100)

        range_tensor = torch.arange(inputs_ids.size(1)).unsqueeze(0)
        range_tensor = range_tensor.repeat(prompt_id_lens.size(0), 1)
        mask_tensor = (range_tensor < prompt_id_lens.unsqueeze(-1)) 
        labels[mask_tensor] = -100

        return inputs_ids, attention_mask, labels
    
    else:
        prompts_tokenized = tokenizer(completions, padding=True, return_tensors='pt')
        return prompts_tokenized["input_ids"], prompts_tokenized["attention_mask"]
    


def make_dataloader(inputs, tokenizer, batch_size=8, train=True):
    if train:
        llama_input_ids, llama_attn_masks, llama_labels = tokenize(tokenizer, inputs, train=train)
        dataset = sft_Dataset(llama_input_ids, llama_attn_masks, llama_labels)
    else:
        llama_input_ids, llama_attn_masks = tokenize(tokenizer, inputs, train=train)
        dataset = sft_test_Dataset(llama_input_ids, llama_attn_masks)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)

    return dataloader


def build_prompt_with_special_tokens_ability(prompt, tokenizer, ability_ls, kc_ls):
    system_content = "You are a student code simulator. Given a programming problem and the student's mastery levels for specific knowledge components (KCs), generate Java code that reflects that understanding, including plausible student errors. Output only the code, with no explanations or comments."

    prompt = "Question: " + prompt + "\n\nStudent information:"
    for idx, (kc_i, kc_ability) in enumerate(zip(kc_ls, ability_ls)):
        kc_intro = f" KC {idx+1}: {kc_i}."
        kc_level = f" The student's mastery level on {kc_i} is {kc_ability}."
        prompt += kc_intro + kc_level
    
    prompt += " Simulate the student written code:"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]

    final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return final_prompt


def build_input_with_special_tokens_ability(prompt, code, tokenizer, ability_ls, kc_ls):
    system_content = "You are a student code simulator. Given a programming problem and the student's mastery levels for specific knowledge components (KCs), generate Java code that reflects that understanding, including plausible student errors. Output only the code, with no explanations or comments."

    prompt = "Question: " + prompt + "\n\nStudent information:"
    for idx, (kc_i, kc_ability) in enumerate(zip(kc_ls, ability_ls)):
        kc_intro = f" KC {idx+1}: {kc_i}."
        kc_level = f" The student's mastery level on {kc_i} is {kc_ability}."
        prompt += kc_intro + kc_level
    
    prompt += " Simulate the student written code:"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": code.strip()},
    ]

    training_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    check = tokenizer.apply_chat_template(messages, tokenize=False, add_special_tokens=False, add_generation_prompt=False)


    return training_str



class sft_Dataset(Dataset):
    def __init__(self, input_ids, attn_masks, labels):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels

    def __getitem__(self, index):
        a = self.input_ids[index]
        b = self.attn_masks[index]
        c = self.labels[index]
        return {'input_ids': a, 'attention_mask': b, 'labels':c}

    def __len__(self):
        return len(self.input_ids)


class sft_test_Dataset(Dataset):
    def __init__(self, input_ids, attn_masks):
        self.input_ids = input_ids
        self.attn_masks = attn_masks

    def __getitem__(self, index):
        a = self.input_ids[index]
        b = self.attn_masks[index]
        return {'input_ids': a, "attention_mask": b}

    def __len__(self):
        return len(self.input_ids)