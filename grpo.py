import argparse
import json
import os
import re
from datetime import datetime

import pandas as pd
from trl import GRPOTrainer, GRPOConfig
import torch
from transformers import AutoModelForCausalLM
from sklearn.model_selection import train_test_split
import wandb
from datasets import Dataset

from data_loader_sft import (
    build_prompt_with_special_tokens,
    build_prompt_with_special_tokens_ability,
    get_inputs,
    get_inputs_ability,
    make_dataloader,
)
from eval_sft import compute_code_bleu_modified, evaluate, inference
from model import create_tokenizer, load_model_eval, lora_model_load
from utils import set_random_seed


def read_data(args, test_size=0.2, random_state=0):
    df = pd.read_pickle(args.data_path)

    if 'timestep' in df.columns:
        df.sort_values(by=['SubjectID', 'timestep'], inplace=True)
    else:
        df.sort_values(by=['SubjectID', 'attempt'], inplace=True)

    # The released experiments use one observation per student/problem: the
    # first AST-convertible submission in the prepared dataset.
    df = df.drop_duplicates(subset=['SubjectID', 'ProblemID'], keep='first').reset_index(drop=True)

    # First-submission annotations are keyed by student and timestamp. A left
    # join preserves correct submissions whose error list is empty.
    error_df = pd.read_pickle(args.error_labels_path)
    df = df.merge(
        error_df[['SubjectID', 'ServerTimestamp', 'error_labels']],
        on=['SubjectID', 'ServerTimestamp'],
        how='left',
    )

    problem_error_dict = pd.read_pickle(args.problem_errors_path)
    map_dict = pd.read_pickle(args.error_mapping_path)

    df['problem_error_labels'] = df['prompt'].map(problem_error_dict)

    df.rename(columns={'error_labels': 'orig_error_labels'}, inplace=True)
    df["Code"] = df["Code"].str.replace(r"/\*.*?\*/", "", regex=True).str.replace(r"/\*[\s\S]*?\*/", "", regex=True).str.replace(r"//.*", "", regex=True).str.strip()
    def clean_error_labels(row):
        problem = row['prompt']
        error_list = row['orig_error_labels']
        if not isinstance(error_list, (list, tuple)) or len(error_list) == 0:
            return []
        
        error_ls_i = list(set([item[-1] for item in error_list]))
        clean_errors = [re.sub(r'(?<!^)(?<!-)(?=[A-Z])', ' ', s) for s in error_ls_i]
        clean_errors = [re.sub(r"\s+", " ", s) for s in clean_errors]
        clean_errors = [s.lower().strip() for s in clean_errors]
        clean_errors = [s[:-1] if s.endswith('s') and not s.endswith('ss') else s for s in clean_errors]

        mapping = map_dict[problem]
        mapped_errors = sorted({mapping[s] for s in clean_errors if s in mapping})

        return mapped_errors

    df['error_labels'] = df.apply(clean_error_labels, axis=1)

    if not args.with_knowledge:
        problem_to_codes = df.groupby("prompt")["Code"].apply(list).to_dict()
        df["code_candidates"] = df["prompt"].map(problem_to_codes)

    if args.split_by == 'student':
        students = df['SubjectID'].unique()
        train_stu, test_stu = train_test_split(students, test_size=test_size, random_state=random_state)
        valid_stu, test_stu = train_test_split(test_stu, test_size=0.5, random_state=random_state)

        train_set = df[df['SubjectID'].isin(train_stu)].reset_index(drop=True)
        valid_set = df[df['SubjectID'].isin(valid_stu)].reset_index(drop=True)
        test_set = df[df['SubjectID'].isin(test_stu)].reset_index(drop=True)
    
    else:
        problems = df['ProblemID'].unique()
        train_problem, test_problem = train_test_split(problems, test_size=test_size, random_state=random_state)
        valid_problem, test_problem = train_test_split(test_problem, test_size=0.5, random_state=random_state)

        train_set = df[df['ProblemID'].isin(train_problem)].reset_index(drop=True)
        valid_set = df[df['ProblemID'].isin(valid_problem)].reset_index(drop=True)
        test_set = df[df['ProblemID'].isin(test_problem)].reset_index(drop=True)

    return train_set, valid_set, test_set


def load_student_dataset(df, tokenizer, language, prompt_column = "prompt"):
    ds = Dataset.from_pandas(df, preserve_index=False)

    def to_prompt(ex):
        system_content = f"You are an LLM that simulates a student writing {language} code. For the given problem, respond the way such a student realistically would: sometimes producing correct code, sometimes making mistakes that students are likely to make. Output code only without any explanations or comments. Do not wrap the code in markdown fences (no ```)."
        user_prompt = f"Problem: {ex[prompt_column]} Student written code:"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        out = {"prompt": prompt,
               "code_candidates": ex["code_candidates"]
        }
        
        return out

    ds = ds.map(to_prompt, remove_columns=ds.column_names)

    return ds


def load_student_ability_dataset(df, tokenizer, language, prompt_column = "prompt"):
    ds = Dataset.from_pandas(df, preserve_index=False)

    def to_prompt(ex):
        system_content = f"You are a student code simulator. Given a programming problem and the student's mastery levels for specific knowledge components (KCs), generate {language} code that reflects that understanding, including plausible student errors. Output only the code, with no explanations or comments. Do not wrap the code in markdown fences (no ```)."

        kc_ability_list = ex['kc_level']
        problem = ex['prompt']
        kc_ls = ex['knowledge_component']

        errors = ex['error_labels']
        problem_errors = ex['problem_error_labels']
        
        prompt = "Problem: " + problem + "\n\nStudent information:"
        for idx, (kc_i, kc_ability) in enumerate(zip(kc_ls, kc_ability_list)):
            kc_intro = f" KC {idx+1}: {kc_i}."
            kc_level = f" The student's mastery level on {kc_i} is {kc_ability}."
            prompt += kc_intro + kc_level
        
        prompt += " Simulate the student written code:"

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        out = {"prompt": prompt,
               "problem": problem,
               "code": ex["Code"],
               "errors": errors,
               "problem_errors": problem_errors
        }
        
        return out

    ds = ds.map(to_prompt, remove_columns=ds.column_names)

    return ds



def match_reward(completions, code_candidates, language='java', **kwargs):
    rewards = []
    n = len(completions)

    for i, group in enumerate(completions):
        generation_i = completions[i].strip()
        code_s = code_candidates[i]

        max_similarity = []
        for gt_code_i in code_s:
            sim = compute_code_bleu_modified([gt_code_i.strip()], [generation_i], language)[0]
            max_similarity.append(sim)

        rewards.append(max(max_similarity))

    return rewards


def diversity_reward(completions, language='java', **kwargs):
    rewards = []
    n = len(completions)
    codebleu_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            codebleu_matrix[i][j] = compute_code_bleu_modified([completions[i]], [completions[j]], language)[0]

    for i in range(n):
        max_sim = max(codebleu_matrix[i])
        div_reward = 1 - max_sim
        rewards.append(div_reward)
    
    return rewards


def make_error_reward_fn(model, tokenizer):
    @torch.inference_mode()
    def contains_error(outputs, problems, error_ls, problem_errors_ls):
        system_content = """You are an experienced code reviewer. You will be provided with a programming problem along with a student code and a list of errors. Your task is to reason on the code and select the errors that best apply to the student code or an empty list if none of the errors are present in the code or the code is correct in JSON format. Choose the smallest set of labels that best explains the code's issues.
Output rules (CRITICAL):
- Output MUST be valid JSON.
- Use DOUBLE QUOTES (") for all strings and keys.
- Do NOT use single quotes (').
- Do NOT include explanations, reasoning, comments, or extra text.
- Do NOT wrap the JSON in markdown or code blocks.        

The output MUST match this exact schema:
{"errors": ["error 1", "error 2", ...]}
"""

        prompt_ls = []
        for output, problem, error_i in zip(outputs, problems, problem_errors_ls):
            user_prompt = f"Problem\n{problem}\n\nCode:\n{output}\n\nThe error list is:\n{error_i}" + "\n\nNow follow the instructions in system message, select all errors from the list that are present in the code in one line json format."

            message = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ]

            prompt = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )

            prompt_ls.append(prompt)

        inputs = tokenizer(prompt_ls, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**inputs, do_sample=False, max_new_tokens=120)
        
        generated = out[:, inputs["input_ids"].shape[1]:]

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        cleaned_out = [s.replace("```json", "").replace("```", "").strip() for s in decoded]

        def iou(generated_code_ls, gt_code_ls):
            set1, set2 = set(generated_code_ls), set(gt_code_ls)
            intersection = set1 & set2
            union = set1 | set2
            return len(intersection) / len(union) if union else 0.0


        try:
            ans = [json.loads(s) for s in cleaned_out]
            error_res = [s['errors'] for s in ans]
            error_reward = [iou(error_res_i, error_ls_i) for error_res_i, error_ls_i in zip(error_res, error_ls)]
        except Exception as e:
            print(f"Error in parsing JSON: {e}")
            for gen_i in cleaned_out:
                print(f"Generated output: {gen_i}")

            error_reward = [0.0 for _ in range(len(outputs))]
        
        return error_reward
    
    @torch.inference_mode()
    def evaluate_code_correctness(outputs, problems, **kwargs):
        system_content = """You are an experienced code reviewer. Given a programming problem along with a code, your task is to determine whether the code correctly solves the problem or not. Output "correct" if the code is correct, otherwise output "incorrect"."""

        prompt_ls = []
        for output, problem in zip(outputs, problems):
            user_prompt = f"Problem\n{problem}\n\nCode:\n{output}"

            message = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ]

            prompt = tokenizer.apply_chat_template(
                message,
                add_generation_prompt=True,
                tokenize=False,
            )

            prompt_ls.append(prompt)

        inputs = tokenizer(prompt_ls, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(**inputs, do_sample=False)
        
        generated = out[:, inputs["input_ids"].shape[1]:]

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        ans = [s.strip() for s in decoded]
        corr_reward = [1 if s.lower() == 'correct' else 0 for s in ans]

        return corr_reward

    def reward_fn(completions, problem, errors, problem_errors, **kwargs):
        # Two cases: if ground truth code is correct, use reward to check if the generated code is also correct, 1 for correct, 0 for incorrect
        if len(errors[0]) == 0:
            if len(errors[len(errors)//2]) == 0:
                corr_reward = evaluate_code_correctness(completions, problem)
                return corr_reward
            else:
                # if ground truth code is incorrect, use reward to check if the generated code contains the same errors
                first_sample_completions = completions[:len(completions)//2]
                first_sample_prompts = problem[:len(problem)//2]
                corr_reward = evaluate_code_correctness(first_sample_completions, first_sample_prompts)
                
                second_sample_completions = completions[len(completions)//2:]
                second_sample_prompts = problem[len(problem)//2:]
                second_sample_errors = errors[len(errors)//2:]
                second_sample_problem_errors = problem_errors[len(problem_errors)//2:]
                
                error_reward = contains_error(second_sample_completions, second_sample_prompts, second_sample_errors, second_sample_problem_errors)

                total_reward = corr_reward + error_reward
                return total_reward
            
        else:
            if len(errors[len(errors)//2]) == 0:
                # if ground truth code is incorrect, use reward to check if the generated code contains the same errors
                first_sample_completions = completions[:len(completions)//2]
                first_sample_prompts = problem[:len(problem)//2]
                first_sample_errors = errors[:len(errors)//2]
                first_sample_problem_errors = problem_errors[:len(problem_errors)//2]
                error_reward = contains_error(first_sample_completions, first_sample_prompts, first_sample_errors, first_sample_problem_errors)

                second_sample_completions = completions[len(completions)//2:]
                second_sample_prompts = problem[len(problem)//2:]
                corr_reward = evaluate_code_correctness(second_sample_completions, second_sample_prompts)
            
                total_reward = error_reward + corr_reward
                return total_reward
            else:
                error_reward = contains_error(completions, problem, errors, problem_errors)
                return error_reward

    return reward_fn


def match_ability_reward(completions, code, language='java', **kwargs):
    rewards = []
    n = len(completions)

    for completions_i, gt_code_i in zip(completions, code):
        codebleu = compute_code_bleu_modified([gt_code_i.strip()], [completions_i.strip()], language)[0]
        rewards.append(codebleu)

    return rewards


def bind_language(reward_fn, language):
    """Bind a dataset language while retaining the name expected by TRL."""
    def wrapped(*args, **kwargs):
        return reward_fn(*args, language=language, **kwargs)

    wrapped.__name__ = reward_fn.__name__
    return wrapped


def grpo(args, device, sft_time, grpo_time, language):
    if args.log_wandb:
        os.makedirs(os.path.join(args.output_dir, grpo_time), exist_ok=True)

        wandb.login()
        wandb.init(project='grpo')
        wandb.config.update(vars(args), allow_val_change=True)
        print('Run id:', wandb.run.id)

    model = lora_model_load(args, device, sft_time, continue_train=True, load_in_8bit=True)
    tokenizer = create_tokenizer(args, sft_time)

    # Use problem + ability level as prompt, need judge model to evaluate errors
    if args.with_knowledge:
        judge_model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16).to(device)
        judge_model.eval()
        judge_tokenizer = create_tokenizer(args)
        judge_tokenizer.padding_side = "left"

    train_set, valid_set, test_set = read_data(args, test_size=args.test_size, random_state=args.random_state)

    if args.with_knowledge:
        train_ds = load_student_ability_dataset(train_set, tokenizer, language, args.prompt_column)
        valid_ds = load_student_ability_dataset(valid_set, tokenizer, language, args.prompt_column)
        test_ds = load_student_ability_dataset(test_set, tokenizer, language, args.prompt_column)
        error_reward = make_error_reward_fn(judge_model, judge_tokenizer)
        reward_functions = [
            bind_language(match_ability_reward, language),
            bind_language(diversity_reward, language),
            error_reward,
        ]
        reward_weights = [1.0, 1.0, 1.0]

    else:
        train_ds = load_student_dataset(train_set, tokenizer, language, args.prompt_column)
        valid_ds = load_student_dataset(valid_set, tokenizer, language, args.prompt_column)
        test_ds = load_student_dataset(test_set, tokenizer, language, args.prompt_column)
        reward_functions = [
            bind_language(match_reward, language),
            bind_language(diversity_reward, language),
        ]
        reward_weights = [args.reward_weight, 1 - args.reward_weight]


    # GRPO training config
    training_args = GRPOConfig(
        output_dir=f"{args.output_dir}/{grpo_time}",
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        # save_steps=1000,
        save_total_limit=1,
        eval_strategy="epoch",
        # eval_steps=1000,
        load_best_model_at_end=True,
        report_to="wandb" if args.log_wandb else "none",
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_new_tokens,
        temperature=1,
        top_k=40,
        repetition_penalty=1.1,
        reward_weights=reward_weights,
        loss_type=args.loss_type
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        reward_funcs=reward_functions,
    )

    trainer.train()

    if args.save_model:
        model_dir = os.path.join(training_args.output_dir, grpo_time, 'model')
        trainer.save_model(model_dir)

def eval_grpo(args, device, grpo_time, language):
    model, tokenizer = load_model_eval(args, grpo_time, device)

    _, _, test_set = read_data(args, test_size=args.test_size, random_state=args.random_state)

    if not args.with_knowledge:
        test_problem_ls, test_code_ls = get_inputs(test_set)
        test_prompt = [build_prompt_with_special_tokens(prompt, tokenizer) for prompt in test_problem_ls]
    
    else:
        test_problem_ls, test_code_ls, test_ability_ls, test_kc_ls = get_inputs_ability(test_set)
        test_prompt = [build_prompt_with_special_tokens_ability(prompt, tokenizer, ability, kc, language) for prompt, ability, kc in zip(test_problem_ls, test_ability_ls, test_kc_ls)]

    inference_dl = make_dataloader(test_prompt, tokenizer, batch_size=4, train=False)
    generated_output = inference(model, inference_dl, tokenizer, device, args)
    
    save_file = f'generated_output_{args.k}.txt'
    with open(os.path.join(args.model_save_dir, grpo_time, save_file), 'w') as f:
        json.dump(generated_output, f, indent=2)
    
    with open(os.path.join(args.model_save_dir, grpo_time, save_file), 'r') as f:
        generated_output = json.load(f)


    results = evaluate(test_code_ls, generated_output, test_problem_ls, args, grpo_time, lang=language)
    if args.k == 1:
        result = {'codeBLEU': results['codebleu']}
    else:
        result = {'top_{}_codeBLEU'.format(args.k): results['top_{}_codebleu'.format(args.k)]}



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file containing student data.")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts in the data.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=0, help="Random state for data splitting.")
    parser.add_argument("--model_save_dir", type=str, default='sft_checkpoints', help="Directory to load the sft model checkpoints and load grpo model for eval.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Pretrained model name or path.")
    parser.add_argument("--output_dir", type=str, default="grpo_checkpoints", help="Output directory for GRPO training.")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=10, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations for GRPO.")
    parser.add_argument("--k", type=int, default=1, help="Number of generations per sample for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=400, help="Maximum completion length for generation.")
    parser.add_argument("--log_wandb", action='store_true', help="Whether to log training with Weights & Biases.")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta parameter for GRPO.")
    parser.add_argument("--error_labels_path", type=str, default="data/data_inc_error_label_full.pkl", help="First-submission error annotations.")
    parser.add_argument("--problem_errors_path", type=str, default="human_eval/problem_error_dict_10.pkl", help="Problem-level error-label candidates.")
    parser.add_argument("--error_mapping_path", type=str, default="human_eval/problem_error_name_mapping_10.pkl", help="Mapping from raw to clustered error labels.")
    parser.add_argument("--sft_checkpoint", type=str, required=True, help="SFT checkpoint directory name under --model_save_dir.")
    parser.add_argument("--split_by", type=str, default='student', help="Split data by 'student' or 'problem'.")
    parser.add_argument("--reward_weight", type=float, default=1, help="Reward function weights for GRPO.")
    parser.add_argument("--with_knowledge", action='store_true', help="Whether to include knowledge component information in prompts.")
    parser.add_argument("--save_model", action='store_true', help="Whether to save the model.")
    parser.add_argument("--loss_type", type=str, default='dapo', help="Loss type for training.")

    args = parser.parse_args()

    set_random_seed(args.random_state)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('SFT model directory:', args.sft_checkpoint)

    language = 'python' if 'falcon' in args.data_path.lower() else 'java'
    grpo_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print('GRPO training time:', grpo_time)

    grpo(args, device, args.sft_checkpoint, grpo_time, language)

    if args.save_model:
        args.model_save_dir = args.output_dir
        eval_grpo(args, device, grpo_time, language)

    if args.log_wandb:
        wandb.finish()



if __name__ == "__main__":
    main()
