import argparse
import pandas as pd
import json
from trl import GRPOTrainer, GRPOConfig
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
from peft import PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from pdb import set_trace
from sklearn.model_selection import train_test_split
import wandb
from utils import *
from model import *
from eval_sft import *
from datasets import Dataset
from datetime import datetime


def read_data(args, test_size=0.2, random_state=0):
    df = pd.read_pickle(args.data_path)
    
    df.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)
    
    ## Data cleaning
    # df.drop(columns=['CodeStateID', 'Code-ast', 'code-astnn', 'code-embedding', 'embedding', 'astnn', 'prompt-embedding', 'input'], inplace=True)
    # df["Code"] = df["Code"].str.replace(r"/\*.*?\*/", "", regex=True).str.replace(r"/\*[\s\S]*?\*/", "", regex=True).str.replace(r"//.*", "", regex=True).str.strip()

    ## Merging original df with error labels
    # error_df = pd.read_pickle('data/data_inc_error_label_full.pkl')
    # df = df.merge(error_df[['SubjectID', 'ServerTimestamp', 'error_labels']], on=['SubjectID', 'ServerTimestamp'], how='left')
    

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


def load_student_dataset(df, tokenizer, prompt_column = "prompt"):
    ds = Dataset.from_pandas(df, preserve_index=False)

    def to_prompt(ex):
        system_content = "You are an LLM that simulates a student writing Java code. For the given problem, respond the way such a student realistically would: sometimes producing correct code, sometimes making mistakes that students are likely to make. Output code only without any explanations or comments. Do not wrap the code in markdown fences (no ```)."
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


def load_student_ability_dataset(df, tokenizer, prompt_column = "prompt"):
    ds = Dataset.from_pandas(df, preserve_index=False)

    def to_prompt(ex):
        system_content = "You are a student code simulator. Given a programming problem and the student's mastery levels for specific knowledge components (KCs), generate Java code that reflects that understanding, including plausible student errors. Output only the code, with no explanations or comments. Do not wrap the code in markdown fences (no ```)."
        
        kc_ability_list = ex['kc_level']
        problem = ex['prompt']
        kc_ls = ex['knowledge_component']

        errors = list(set([item[-1] for item in ex['error_labels']]))
        
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
               "errors": errors
        }
        
        return out

    ds = ds.map(to_prompt, remove_columns=ds.column_names)

    return ds



def match_reward(completions, code_candidates, **kwargs):
    rewards = []
    n = len(completions)

    for i, group in enumerate(completions):
        generation_i = completions[i].strip()
        code_s = code_candidates[i]

        max_similarity = []
        for gt_code_i in code_s:
            sim = compute_code_bleu_modified([gt_code_i.strip()], [generation_i], 'java')[0]
            max_similarity.append(sim)

        rewards.append(max(max_similarity))

    return rewards


def diversity_reward(completions, **kwargs):
    rewards = []
    n = len(completions)
    codebleu_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            codebleu_matrix[i][j] = compute_code_bleu_modified([completions[i]], [completions[j]], 'java')[0]


    for i in range(n):
        max_sim = max(codebleu_matrix[i])
        div_reward = 1 - max_sim
        rewards.append(div_reward)
    
    return rewards


def make_error_reward_fn(model, tokenizer):
    @torch.inference_mode()
    def contains_error(outputs, problems, error_ls):
        system_content = """You are an experienced code reviewer. Given a programming problem along with a code and a list of errors, your task is to:
1. Examine the code and all the errors in the list.
2. Reason which errors from the list are included in the code and return all that apply based on your reasoning.
3. Return an empty list if none of the errors are present in the code or the code is correct.

Return ONLY valid JSON on a single line. Do not use markdown code fences. Do not include trailing comments.
Output format: {"errors": [<error 1>, <error 2>, ...]}
"""

        prompt_ls = []
        for output, problem, error_i in zip(outputs, problems, error_ls):
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
        out = model.generate(**inputs, do_sample=False)
        
        generated = out[:, inputs["input_ids"].shape[1]:]

        decoded = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        cleaned_out = [s.replace("```json", "").replace("```", "").strip() for s in decoded]

        try:
            ans = [json.loads(s) for s in cleaned_out]
            error_res = [s['errors'] for s in ans]

            error_reward = [len(err_i) / len(error_ls[i]) if len(error_ls[i]) > 0 else 0 for i, err_i in enumerate(error_res)]
        except Exception as e:
            print("Error in parsing JSON:", e)
            # print("Generated outputs:", cleaned_out)
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

    def reward_fn(completions, problem, errors, **kwargs):
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
                error_reward = contains_error(second_sample_completions, second_sample_prompts, second_sample_errors)

                total_reward = corr_reward + error_reward
                return total_reward
            
        else:
            if len(errors[len(errors)//2]) == 0:
                # if ground truth code is incorrect, use reward to check if the generated code contains the same errors
                first_sample_completions = completions[:len(completions)//2]
                first_sample_prompts = problem[:len(problem)//2]
                first_sample_errors = errors[:len(errors)//2]
                error_reward = contains_error(first_sample_completions, first_sample_prompts, first_sample_errors)

                second_sample_completions = completions[len(completions)//2:]
                second_sample_prompts = problem[len(problem)//2:]
                corr_reward = evaluate_code_correctness(second_sample_completions, second_sample_prompts)
            
                total_reward = error_reward + corr_reward
                return total_reward
            else:
                error_reward = contains_error(completions, problem, errors)
                return error_reward

    return reward_fn


def match_ability_reward(completions, code, **kwargs):
    rewards = []
    n = len(completions)

    for completions_i, gt_code_i in zip(completions, code):
        codebleu = compute_code_bleu_modified([gt_code_i.strip()], [completions_i.strip()], 'java')[0]
        rewards.append(codebleu)

    return rewards


def grpo(args, device, sft_time, grpo_time):
    if args.log_wandb:
        os.makedirs(os.path.join(args.output_dir, grpo_time), exist_ok=True)

        wandb.login(key=args.wandb_key, verify=True)
        wandb.init(project='grpo')
        wandb.config.update(vars(args), allow_val_change=True)
        print('Run id:', wandb.run.id)

    model = lora_model_load(args, device, sft_time, continue_train=True, load_in_8bit=True)
    tokenizer = create_tokenizer(args, sft_time)

    # Use problem + ability level as prompt, need judge model to evaluate errors
    if args.with_knowledge:
        judge_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct", dtype=torch.bfloat16).to(device)
        judge_model.eval()
        judge_tokenizer = create_tokenizer(args)
        judge_tokenizer.padding_side = "left"

    train_set, valid_set, test_set = read_data(args, test_size=args.test_size, random_state=args.random_state)

    if args.with_knowledge:
        train_ds = load_student_ability_dataset(train_set, tokenizer, args.prompt_column)
        valid_ds = load_student_ability_dataset(valid_set, tokenizer, args.prompt_column)
        test_ds = load_student_ability_dataset(test_set, tokenizer, args.prompt_column)
        error_reward = make_error_reward_fn(judge_model, judge_tokenizer)
        # reward_functions = [match_ability_reward, diversity_reward, error_reward]
        # reward_weights = [args.reward_weight, 1 - args.reward_weight, 1.0]
        reward_functions = [diversity_reward, error_reward]
        reward_weights = [1.0, 1.0]

    else:
        train_ds = load_student_dataset(train_set, tokenizer, args.prompt_column)
        valid_ds = load_student_dataset(valid_set, tokenizer, args.prompt_column)
        test_ds = load_student_dataset(test_set, tokenizer, args.prompt_column)
        reward_functions = [match_reward, diversity_reward]
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
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=1000,
        load_best_model_at_end=True,
        report_to="wandb" if args.log_wandb else "none",
        num_generations=args.num_generations,
        beta=args.beta,
        max_completion_length=args.max_new_tokens,
        top_k=50,
        repetition_penalty=1.1,
        reward_weights=reward_weights,
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

    if args.log_wandb:
        model_dir = os.path.join(training_args.output_dir, grpo_time, 'model')
        trainer.save_model(model_dir)


def eval_grpo(args, device, grpo_time):
    model, tokenizer = load_model_eval(args, grpo_time, device)

    _, _, test_set = read_data(args, test_size=args.test_size, random_state=args.random_state)

    if not args.with_knowledge:
        test_problem_ls, test_code_ls = get_inputs(test_set)
        test_prompt = [build_prompt_with_special_tokens(prompt, tokenizer) for prompt in test_problem_ls]
    
    else:
        test_problem_ls, test_code_ls, test_ability_ls, test_kc_ls = get_inputs_ability(test_set)
        test_prompt = [build_prompt_with_special_tokens_ability(prompt, tokenizer, ability, kc) for prompt, ability, kc in zip(test_problem_ls, test_ability_ls, test_kc_ls)]

    inference_dl = make_dataloader(test_prompt, tokenizer, batch_size=4, train=False)
    generated_output = inference(model, inference_dl, tokenizer, device, args)
    
    save_file = f'generated_output_{args.k}.txt'
    with open(os.path.join(args.model_save_dir, grpo_time, save_file), 'w') as f:
        json.dump(generated_output, f, indent=2)
    
    results = evaluate(test_code_ls, generated_output, test_problem_ls, args, grpo_time, lang='java')
    if args.k == 1:
        result = {'codeBLEU': results['codebleu']}
    else:
        result = {'top_{}_codeBLEU'.format(args.k): results['top_{}_codebleu'.format(args.k)]}
    
    # # load the saved generated output
    # # save_file = f'generated_output_{args.k}.txt'
    # save_file = f'eval_logs_updated_{args.k}.txt'
    # with open(os.path.join(args.model_save_dir, grpo_time, save_file), 'r') as f:
    #     generated_output = json.load(f)
        
    #     problems = generated_output['problems']
    #     generated_output = generated_output['generated_codes']
    
    error_df = pd.read_pickle('data/data_inc_error_label_full.pkl')

    test_set.drop(columns=['error_labels'], inplace=True)
    eval_res = evaluate_error_coverage_individual(test_set, error_df, generated_output, args)

    with open(os.path.join(args.model_save_dir, grpo_time, 'llm_judge_res.pkl'), 'wb') as f:
        pickle.dump(eval_res, f)

    print('Finished LLM as judge evaluation for individual level error coverage')

    print('Start LLM as judge overall evaluation')

    with open('data/problem_error_dict_30.pkl', 'rb') as f:
        problem_error_dict = pickle.load(f)

    output = evaluate_error_coverage_overall(test_set, problem_error_dict, generated_output, args)
    
    with open(os.path.join(args.model_save_dir, grpo_time, 'llm_judge_overall_res.pkl'), 'wb') as f:
        pickle.dump(output, f)
    
    print('Finished LLM as judge overall evaluation')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file containing student data.")
    parser.add_argument("--prompt_column", type=str, default="prompt", help="Column name for prompts in the data.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument("--random_state", type=int, default=0, help="Random state for data splitting.")
    parser.add_argument("--model_save_dir", type=str, default='sft_checkpoints', help="Directory to save the model checkpoints.")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Pretrained model name or path.")
    parser.add_argument("--output_dir", type=str, default="grpo_checkpoints", help="Output directory for GRPO training.")
    parser.add_argument("--learning_rate", type=float, default=1e-7, help="Learning rate for training.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=10, help="Batch size per device during training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--num_generations", type=int, default=5, help="Number of generations for GRPO.")
    parser.add_argument("--k", type=int, default=1, help="Number of generations per sample for inference.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum completion length for generation.")
    parser.add_argument("--log_wandb", action='store_true', help="Whether to log training with Weights & Biases.")
    parser.add_argument("--beta", type=float, default=0.0, help="Beta parameter for GRPO.")
    parser.add_argument("--split_by", type=str, default='student', help="Split data by 'student' or 'problem'.")
    parser.add_argument("--reward_weight", type=float, default=1, help="Reward function weights for GRPO.")
    parser.add_argument("--with_knowledge", action='store_true', help="Whether to include knowledge component information in prompts.")
    parser.add_argument("--save_model", action='store_true', help="Whether to save the model.")

    args = parser.parse_args()

    set_random_seed(args.random_state)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    # sft_time = '20251130_211626' # split by student

    # sft_time = '20251206_070913' # split by problem

    sft_student_time = '20251204_154225_student_knowledge' # split by student with knowledge

    grpo_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(grpo_time)

    grpo(args, device, sft_student_time, grpo_time)

    # args.model_save_dir = args.output_dir

    # grpo_time = '20251215_212503' # grpo with student knowledge split by student
    # eval_grpo(args, device, grpo_time)

if __name__ == "__main__":
    main()
