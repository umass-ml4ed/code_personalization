import torch
import pickle
import nltk
from nltk import ngrams
import os
import abc
from tqdm import tqdm
from pdb import set_trace
import hydra
import json
from transformers import GenerationConfig
from model import *
import wandb
from collections import Counter, defaultdict
import statistics
import subprocess

from utils import set_random_seed
from trainer import *
from data_loader_sft import *
from evaluator.CodeBLEU import calc_code_bleu
from huggingface_hub import login
from utils import aggregate_metrics
import warnings
from error_labeling import *
from openai_api import *

def inference(model, dataloader, tokenizer, device, configs):
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    tokenizer.padding_side = "left"

    preds = []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc="Inference", leave=False)):
            b_inpids, b_attnids = batch['input_ids'].to(device), batch['attention_mask'].to(device)

            generation_config = GenerationConfig(do_sample=True, temperature=0.7, top_p=1, top_k=40, repetition_penalty=1.1, num_return_sequences=configs.k)

            model_outputs = model.generate(input_ids=b_inpids, attention_mask=b_attnids, max_new_tokens=configs.max_new_tokens, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, generation_config=generation_config)
            # model_outputs = model.generate(input_ids=b_inpids, attention_mask=b_attnids, max_new_tokens=configs.max_new_tokens, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=False)

            decoded_preds = tokenizer.batch_decode(model_outputs[:, b_inpids.size(1):], skip_special_tokens=True)

            decoded_preds_clean = [i.strip() for i in decoded_preds]
            preds.extend(decoded_preds_clean)

    return preds


def evaluate(ground_truth_code, preds, problem, configs, now, lang='java'):
    # Evaluation on top k output to check exact match/codeBLEU
    results = {}

    assert len(ground_truth_code) * configs.k == len(preds)
    exact_cnt = 0
    if configs.k == 1:
        pres_segmented = preds
    else:
        pres_segmented = [preds[i:i + configs.k] for i in range(0, len(preds), configs.k)]

    if configs.k == 1:
        codebleu_score, detailed_codebleu_score = compute_code_bleu(ground_truth_code, preds, lang)
        results['codebleu'] = codebleu_score
        results['detailed_codebleu'] = detailed_codebleu_score
    else:
        # top k codeBLEU
        all_codebleu_scores = []
        for gt_code, gen_codes in zip(ground_truth_code, pres_segmented):
            codebleu_scores = []
            for gen_code in gen_codes:
                codebleu_score, _ = compute_code_bleu_modified([gt_code], [gen_code], lang)
                codebleu_scores.append(codebleu_score)
            all_codebleu_scores.append(max(codebleu_scores))
        
        avg_codebleu = sum(all_codebleu_scores) / len(all_codebleu_scores)
        results['top_{}_codebleu'.format(configs.k)] = avg_codebleu

    print(f"results: {results}")

    ## save results
    results['generated_codes'] = pres_segmented
    results['ground_truth_codes'] = ground_truth_code
    results['problems'] = problem

    # code_analyze(now, configs, results)

    if configs.save_model:
        file_name = f'eval_logs_updated_{configs.k}.txt'
        with open(os.path.join(configs.model_save_dir, now, file_name), 'w') as f:
            json.dump(results, f, indent=2)

    return results



def evaluate_vllm(model, lora_req, tokenizer, prompts, ground_truth_code, now, configs, lang='java'):
    results = {}
    preds = []

    for i in tqdm(range(0, len(prompts), 16), desc="inference", leave=False):
        print(f'Process: {i+16}/{len(prompts)}')
        batch_prompts = prompts[i:i+16]

        sampling_params = SamplingParams(temperature=0.7, max_tokens=400, top_p=1.0, top_k=50, repetition_penalty=1.1)
        outputs = model.generate(batch_prompts, sampling_params=sampling_params, lora_request=lora_req)
        decoded_preds = [output_i.outputs[0].text.strip() for output_i in outputs]

        preds.extend(decoded_preds)

    # Calculate evaluation metrics
    codebleu_score, detailed_codebleu_score = compute_code_bleu(ground_truth_code, preds, lang)
    results['codebleu'] = codebleu_score
    results['detailed_codebleu'] = detailed_codebleu_score

    print(f"results: {results}")

    ## save results
    results['generated_codes'] = preds
    results['ground_truth_codes'] = ground_truth_code
    # results['problems'] = problem

    return results


def compute_code_bleu(ground_truth_codes, generated_codes, lang='java'):
    params='0.25,0.25,0.25,0.25'
    codebleu_score, detailed_codebleu_score = calc_code_bleu.get_codebleu(pre_references=[ground_truth_codes], hypothesis=generated_codes, lang=lang, params=params)
    
    return codebleu_score, detailed_codebleu_score

def compute_code_bleu_modified(ground_truth_codes, generated_codes, lang='java'):
    params='0.25,0.25,0.25,0.25'
    # codebleu_score, detailed_codebleu_score = calc_code_bleu.get_codebleu(pre_references=[ground_truth_codes], hypothesis=generated_codes, lang=lang, params=params)
    codebleu_score, detailed_codebleu_score = calc_code_bleu.get_codebleu_modified(pre_references=[ground_truth_codes], hypothesis=generated_codes, lang=lang, params=params)
    
    return codebleu_score, detailed_codebleu_score



def code_analyze(now, configs, res=None):
    if res:
        results = res
    else:
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'rb') as f:
            results = pickle.load(f)

    ground_truth_codes = results['ground_truth_codes']
    generated_codes = results['generated_codes']
    # problems = results['problems']

    # df = pd.DataFrame({'problem': problems, 'ground_truth_code': ground_truth_codes, 'generated_code': generated_codes})

    # groups = df.groupby(['problem'])

    # for problem, group in groups:
    #     code_check = group.groupby(['generated_code']).size().reset_index(name='count')
    #     print(len(group))
    #     print(code_check)

    total = len(ground_truth_codes)
    exact_match = 0

    for gt_code, gen_code in zip(ground_truth_codes, generated_codes):
        for code_i in gen_code:
            if gt_code.strip() == code_i.strip():
                exact_match += 1
                break

    exact_match_rate = exact_match / total * 100
    print(f"Exact Match Rate: {exact_match_rate:.2f}% ({exact_match}/{total})")

    if configs.k == 1:
        res = code_compilation_analysis(generated_codes)


def code_compilation_analysis(generated_codes):
    compilation_results = []
    print(len(generated_codes))
    for idx, code in enumerate(tqdm(generated_codes, desc="Compiling generated codes", leave=False)):
        class_name = f"StudentSolution"
        compile_result = compile_student_method(code, class_name=class_name, javac_path="javac")
        compilation_results.append(compile_result['ok'])

    compilable_count = sum(compilation_results)
    total_count = len(generated_codes)
    compilable_rate = compilable_count / total_count * 100
    print(f"Compilable Rate: {compilable_rate:.2f}% ({compilable_count}/{total_count})")

    return compilation_results


# Analyze code compilation results part
def wrap_method_into_class(student_code, class_name="StudentSolution"):
    """
    Wrap a standalone Java method into a compilable class with a minimal main().
    Returns the full Java class source as a string.
    """
    if "(" not in student_code or ")" not in student_code or "{" not in student_code:
        raise ValueError("student_code does not look like a Java method.")

    JAVA_MAIN_STUB = """public static void main(String[] args) { 
}
"""

    # method_name = extract_method_name(student_code)

    lines = []
    lines.append(f"public class {class_name} " + "{")
    lines.append(student_code.strip())
    lines.append("")
    # Add a minimal main so javac can compile a standalone file cleanly
    lines.append(JAVA_MAIN_STUB)
    lines.append("}")
    return "\n".join(lines) + "\n"

def save_code(code, class_name="StudentSolution"):
    '''
    Save the wrapped code
    '''
    # Save the code
    with open('compiler_code/{:s}.java'.format(class_name), 'w') as f:
        f.write(code)


def compile_student_method(student_code, class_name="StudentSolution", javac_path="javac"):
    """
    Compile the provided Java method by wrapping it into a class and calling javac.
    """
    source = wrap_method_into_class(student_code, class_name=class_name)

    save_code(source, class_name=class_name)
    java_path = f'compiler_code/{class_name}.java'
    try:
        proc = subprocess.run(
            [javac_path, java_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Error: {e}",
            "java_file": java_path,
            "class_file": None,
        }

    ok = proc.returncode == 0

    return {
        "ok": ok,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "java_file": java_path,
    }


def evaluate_error_coverage_individual(test_set, error_df, generated_output, configs, judge_res=None):
    combined_df = test_set.merge(error_df[['ServerTimestamp', 'Code', 'error_labels']], on=['ServerTimestamp', 'Code'], how='left')

    combined_df['generated_code'] = generated_output

    # filter only incorrect submissions
    combined_df = combined_df[combined_df['error_labels'].notnull()].reset_index(drop=True)

    error_label_list = combined_df['error_labels'].tolist()
    problems = combined_df['prompt'].tolist()
    error_ls = [[item[-1] for item in error_i] for error_i in error_label_list]
    codes = combined_df['generated_code'].tolist()

    if not judge_res:
        system_message = """You are an experienced code reviewer. You will be provided with a programming problem along with a student code and a list of errors. 

    Your task is to:
    1. Carefully examine the student code and all the errors in the list to ensure none are overlooked.
    2. Reason which errors from the list are included in the student code and return all that apply based on your reasoning.
    3. Return an empty list if none of the errors are present in the code or the code is correct.

    Return your output strictly in the following JSON format:
    { "included errors": [<error 1>, <error 2>, ...] // Errors appear in the student code from the error list }
    """

        prompts = []
        for p, c, e in zip(problems, codes, error_ls):
            if configs.k == 1:
                prompt = (
                    f"Problem\n{p}\n\n"
                    f"Student code:\n{c}\n\n"
                    f"The error list is:\n{e}\n\n"
                    "Now follow the instructions in system message, "
                    "select all errors from the list that are present in the student code."
                )
                prompts.append(prompt)
            else:
                for c_i in c:
                    prompt = (
                        f"Problem\n{p}\n\n"
                        f"Student code:\n{c_i}\n\n"
                        f"The error list is:\n{e}\n\n"
                        "Now follow the instructions in system message, "
                        "select all errors from the list that are present in the student code."
                    )
                    prompts.append(prompt)

        start_time = time.perf_counter()

        client = OpenAIClient(False)
        generation_kwargs = {"n": 1, "response_format": {"type": "json_object"}}
        judge_res = client.get_responses(prompts, 'o4-mini', system_message, generation_kwargs, False)
        
        end_time = time.perf_counter()
        print(f'Total LLM judge time: {(end_time - start_time) / 60} mins')
    
    if configs.k > 1:
        judge_res = [judge_res[i:i + configs.k] for i in range(0, len(judge_res), configs.k)]

    summary = []
    for judge_res_i, error_list in zip(judge_res, error_ls):
        if configs.k > 1:
            judge_res_i = [json.loads(judge_res_i_i) if judge_res_i_i else {} for judge_res_i_i in judge_res_i]
            coverage_list = []
            for judge_res_i_j in judge_res_i:
                if len(judge_res_i_j) > 0 and 'included errors' in judge_res_i_j:
                    coverage = len(judge_res_i_j['included errors']) / len(error_list) if len(error_list) > 0 else 0
                    coverage_list.append(coverage)
            coverage = max(coverage_list)
        else:
            if judge_res_i:
                judge_res_i = json.loads(judge_res_i)
                if len(judge_res_i) > 0 and 'included errors' in judge_res_i:
                    coverage = len(judge_res_i['included errors']) / len(error_list) if len(error_list) > 0 else 0
            else:
                coverage = 0
        # print(f'Error coverage: {coverage}')
        summary.append(coverage)
    
    print(f'Average error coverage across submissions: {sum(summary)/len(summary)}')


    prob_summ = defaultdict(list)
    for i, (problem, error, res) in enumerate(zip(problems, error_ls, judge_res)):
        prob_summ[problem].append((error, res))
    
    total = []
    for key, val in prob_summ.items():
        prob = []
        for error_list, judge_res_i in val:
            if configs.k == 1:
                if judge_res_i:
                    judge_res_i = json.loads(judge_res_i)
                    if len(judge_res_i) > 0 and 'included errors' in judge_res_i:
                        coverage = len(judge_res_i['included errors']) / len(error_list) if len(error_list) > 0 else 0
                        prob.append(coverage)
                else:
                    prob.append(0)
            else:
                judge_res_i = [json.loads(judge_res_i_i) if judge_res_i_i else {} for judge_res_i_i in judge_res_i]
                coverage_list = []
                for judge_res_i_j in judge_res_i:
                    if len(judge_res_i_j) > 0 and 'included errors' in judge_res_i_j:
                        coverage = len(judge_res_i_j['included errors']) / len(error_list) if len(error_list) > 0 else 0
                        coverage_list.append(coverage)
                prob.append(max(coverage_list))
        
        avg_cov = sum(prob) / len(prob) if len(prob) > 0 else 0
        total.append(avg_cov)
    
    print(f'Average error coverage across problems: {sum(total)/len(total)}')

    return judge_res


def generate_error_dict():
    error_df = pd.read_pickle('data/data_inc_error_label_full.pkl')
    problem_error_dict = {}
    uniq_problems = error_df['ProblemID'].unique().tolist()

    for problem in uniq_problems:
        sub_error_df = error_df[error_df['ProblemID'] == problem].reset_index(drop=True)

        try:
            nunique = sub_error_df['error_labels'].explode().dropna().unique() #[(type, error)]

            errors = [item[1] for item in nunique]
            clean_errors = [re.sub(r'(?<!^)(?<!-)(?=[A-Z])', ' ', s) for s in errors]
            clean_errors = [re.sub(r"\s+", " ", s) for s in clean_errors]
            clean_errors = [s.lower().strip() for s in clean_errors]

            # optional removing plural 's'
            clean_errors = [s[:-1] if s.endswith('s') and not s.endswith('ss') else s for s in clean_errors]

            error_list = list(set(clean_errors))

            cluster_to_errors = error_cluster_linkage(error_list, "all-mpnet-base-v2", 30)
            error_summarized, map_dict = error_summarize(cluster_to_errors)

            errors = list(set(map_dict.values()))
            print(len(errors))

            # for error in errors:
            #     print(f'- {error}')

            prob_statement = sub_error_df['prompt'].unique()[0]
            problem_error_dict[prob_statement] = errors

        except Exception as e:
            set_trace()
            print(e)
    
    save_name = f'data/problem_error_dict_{30}.pkl'
    with open(save_name, 'wb') as f:
        pickle.dump(problem_error_dict, f)




# TODO; adapt with configs.k > 1
def evaluate_error_coverage_overall(test_set, error_dict, generated_output, configs, output=None):    
    test_set['generated_code'] = generated_output

    incor_subset = test_set[test_set['Score'] < 1].reset_index(drop=True)
    problems = incor_subset['prompt'].tolist()
    codes = incor_subset['generated_code'].tolist()

    if not output:
        client = OpenAIClient(False)

        generation_kwargs = {"n": 1, "response_format": {"type": "json_object"}}
        coverage_cnt_ls = []

        prompts = []
        for p, c in zip(problems, codes):
            error_ls = error_dict[p]
            if configs.k == 1:
                prompt = (
                    f"Problem\n{p}\n\n"
                    f"Student code:\n{c}\n\n"
                    f"The error list is:\n{error_ls}\n\n"
                    "Now follow the instructions in system message, "
                    "select all errors from the list that are present in the student code."
                )
                prompts.append(prompt)
            else:
                for c_i in c:
                    prompt = (
                        f"Problem\n{p}\n\n"
                        f"Student code:\n{c_i}\n\n"
                        f"The error list is:\n{error_ls}\n\n"
                        "Now follow the instructions in system message, "
                        "select all errors from the list that are present in the student code."
                    )
                    prompts.append(prompt)
            
            
        system_message = """You are an experienced code reviewer. You will be provided with a programming problem along with a student code and a list of errors. 

    Your task is to:
    1. Carefully examine the student code and all the errors in the list to ensure none are overlooked.
    2. Reason which errors from the list are included in the student code and return all that apply based on your reasoning.
    3. Return an empty list if none of the errors are present in the code or the code is correct.

    Return your output strictly in the following JSON format:
    { "included errors": [<error 1>, <error 2>, ...] // Errors appear in the student code from the error list }
    """

        output = client.get_responses(prompts, 'o4-mini', system_message, generation_kwargs, False)

        if configs.k > 1:
            output = [output[i:i + configs.k] for i in range(0, len(output), configs.k)]

    error_covered = defaultdict(set)
    for i, (problem, res) in enumerate(zip(problems, output)):
        if configs.k == 1:
            if res:
                judge_res_i = json.loads(res)
                if len(judge_res_i) > 0 and 'included errors' in judge_res_i:
                    included_errors = judge_res_i['included errors']
                    
                    if len(included_errors) > 0:
                        for error in included_errors:
                            error_covered[problem].add(error)
        else:
            judge_res_i = [json.loads(res_i) if res_i else {} for res_i in res]
            for judge_res_i_j in judge_res_i:
                if len(judge_res_i_j) > 0 and 'included errors' in judge_res_i_j:
                    included_errors = judge_res_i_j['included errors']
                    
                    if len(included_errors) > 0:
                        for error in included_errors:
                            error_covered[problem].add(error)
        
    for problem, error_set in error_covered.items():
        error_ls = error_dict[problem]
        coverage = len(error_set) / len(error_ls) if len(error_ls) > 0 else 0
        print(f'Error coverage for problem: {coverage}')
        coverage_cnt_ls.append(coverage)
    
    avg_coverage = sum(coverage_cnt_ls) / len(coverage_cnt_ls)
    print(f'Average error coverage across problems: {avg_coverage}')

    set_trace()
    return output


@hydra.main(version_base=None, config_path=".", config_name="configs_sft")
def main(configs):
    warnings.filterwarnings("ignore")

    # Make reproducible
    set_random_seed(configs.seed)

    now = '20251204_154225_student_knowledge' # sft with student knowledge by student
    # now = '20251218_044312' # grpo with student knowledge split by student

    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if configs.use_cuda: assert device.type == 'cuda', 'No GPU found'

    if configs.log_wandb:
        wandb.login(key=configs.wandb_key, verify=True)
        wandb.init(project=configs.wandb_project, id="wn1lxbx3", resume="must")

    # model, tokenizer = load_model_eval(configs, now, device)

    # model, lora_req, tokenizer = load_model_eval_vllm(configs, now, device)

    # train_stu, valid_stu, test_stu, df, students = read_data('data/data_first_sub.pkl', configs)
    train_stu, valid_stu, test_stu, df, students = read_data('data/data_prob_0_comb.pkl', configs)

    split_col = 'ProblemID' if configs.split_by == 'problem' else 'SubjectID'
    test_set = df[df[split_col].isin(test_stu)].reset_index(drop=True)

    if not configs.student_modeling:
        test_problem_ls, test_code_ls = get_inputs(test_set)

        test_prompt = [build_prompt_with_special_tokens(prompt, tokenizer) for prompt in test_problem_ls]
    
    else:
        test_problem_ls, test_code_ls, test_ability_ls, test_kc_ls = get_inputs_ability(test_set)
        # test_prompt = [build_prompt_with_special_tokens_ability(prompt, tokenizer, ability, kc) for prompt, ability, kc in zip(test_problem_ls, test_ability_ls, test_kc_ls)]

    # # try vllm inference
    # evaluate_vllm(model, lora_req, tokenizer, test_prompt, test_code_ls, now, configs, lang='java')
    

    # inference_dl = make_dataloader(test_prompt, tokenizer, batch_size=4, train=False)
    # generated_output = inference(model, inference_dl, tokenizer, device, configs)
    # if configs.save_model:
    #     save_file = f'generated_output_{configs.k}.txt'
    #     with open(os.path.join(configs.model_save_dir, now, save_file), 'w') as f:
    #         json.dump(generated_output, f, indent=2)

    set_trace()
    # load the saved generated output
    save_file = f'generated_output_{configs.k}.txt'
    # save_file = f'eval_logs_updated_{configs.k}.txt'
    with open(os.path.join(configs.model_save_dir, now, save_file), 'r') as f:
        generated_output = json.load(f)
        if 'eval' in save_file:
            problems = generated_output['problems']
            generated_output = generated_output['generated_codes']


    results = evaluate(test_code_ls, generated_output, test_problem_ls, configs, now, lang='java')
    if configs.k == 1:
        result = {'codeBLEU': results['codebleu']}
    else:
        result = {'top_{}_codeBLEU'.format(configs.k): results['top_{}_codebleu'.format(configs.k)]}

    set_trace()

    if configs.k > 1:
        generated_output = [generated_output[i:i + configs.k] for i in range(0, len(generated_output), configs.k)]

    # Evaluate error coverage with LLM as judge in two manners:
    # 1. evaluate error coverage for each individual submission in test set
    error_df = pd.read_pickle('data/data_inc_error_label_full.pkl')

    eval_res = evaluate_error_coverage_individual(test_set, error_df, generated_output, configs)

    with open(os.path.join(configs.model_save_dir, now, 'llm_judge_res.pkl'), 'wb') as f:
        pickle.dump(eval_res, f)

    print('Finished LLM as judge evaluation for individual level error coverage')
    
    # 2. cluster the errors for each problem in test set, and evaluate coverage based on clustered error set
    print('Start LLM as judge overall evaluation')

    with open('data/problem_error_dict_30.pkl', 'rb') as f:
        problem_error_dict = pickle.load(f)


    output = evaluate_error_coverage_overall(test_set, problem_error_dict, generated_output, configs)
    
    with open(os.path.join(configs.model_save_dir, now, 'llm_judge_overall_res.pkl'), 'wb') as f:
        pickle.dump(output, f)
    
    print('Finished LLM as judge overall evaluation')

    # if configs.log_wandb:
    #     wandb.log(result)
    #     wandb.finish()


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()

    main()
