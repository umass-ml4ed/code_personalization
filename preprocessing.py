import json
import openai
from openai import OpenAI
import pandas as pd
from pdb import set_trace
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import subprocess
import os
import math


# Generate 1 sample solution for each question
def get_sample_solution_gpt(prompt, model='gpt-4o', temperature=0):
    try:
        user_prompt = "Problem: " + prompt[0] + " Method name and parameters: " + prompt[1]
        
        system_content = """You are an experienced computer science teacher and education expert. You are given a Java programming problem with the method name and parameter. Your job is to generate one sample solution code to solve the problem. Please follow these instructions carefully when generating the solution:
- The solution should use concepts appropriate for students in an introductory programming course.
- Do not include any comments in the solution.
- Return the solution in a JSON object using the template: {"code": code_1}.
"""
        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
            ],
            temperature = temperature,
            n=1
        )

        reply = response.choices[0].message.content.strip()
        reply_json = json.loads(reply)

        return reply_json
    except Exception as e:
        print(e)
        return None


def get_sample_solution_llama(prompt, model, tokenizer, device):
    user_prompt = f"Problem: {prompt[0]} Method name and parameters: {prompt[1]}"
        
    system_content = """
You are an experienced computer science teacher. You will be given a Java programming problem along with the method name and parameters. Your job is to generate a single sample solution in Java.

**Very important formatting instructions:**
- Do not include any comments in the solution.
- The code must be valid Java.
- Wrap the code in a JSON object with the key "code".
- The value must be a single-line JSON string with all newline characters escaped as \\n. For example:
  {"code": "public int foo() {\\n  return 0;\\n}"}
"""


    messages = [
        {
         "role": "system",
         "content": system_content
        },
        {"role": "user",
            "content": user_prompt
        }]
    
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )


    input_ids = input_ids.to(device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    res = model.generate(**input_ids, max_new_tokens=1600)
    ans = tokenizer.decode(res[0][input_ids['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
   
    # try:
    #     reply_json = json.loads(ans)
    # except Exception as e:
    #     set_trace()
    #     print(ans)


    return ans


# Use gpt-4o to determine if the generated code is correct
def get_solution_correctness(prompt, model='gpt-4o', temperature=0):
    openai.api_key = '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'
    try:
        user_prompt = "Problem:\n" + prompt[0] + "\n\nCode\n" + prompt[1]
        
        system_content = """You are given a Java programming problem and a student's code submission. Your task is to decide whether the code correctly solves the problem.

Return your answer as a JSON object in the following format:
{"Code correctness": "correct" or "incorrect",
"Corrected code": "correct code to the problem if incorrect, else None"}
"""

        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
            ],
            temperature = temperature,
            n=1
        )

        reply = response.choices[0].message.content.strip()
        reply_json = json.loads(reply)

        return reply_json
    except Exception as e:
        print(e)
        return None



def load_model_and_tokenizer(model_name, device):
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    ).to(device)

    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

# Save sample solutions for each problem
def save_problem_solutions(df, model='gpt-4o', temperature=0, tokenizer=None, device=None):
    solution_dict = {}
    cnt = 0
    cnt = 0
    error_ls = []
    for index, row in df.iterrows():
        problem = row['prompt']
        code = row['Code']

        code_piece = code.split('\r')
        method_sig = code_piece[0]
        
        if model == 'gpt-4o':
            solution_i = get_sample_solution_gpt([problem, method_sig], model=model, temperature=temperature)
        else:
            solution_i = get_sample_solution_llama([problem, method_sig], model, tokenizer, device)
        
        # solution = solution_i['code']
        # if '//' in solution:
        #     set_trace()
        #     print('Found comment')

        try:
            reply_json = json.loads(solution_i)
            solution = reply_json['code']
        except Exception as e:
            # set_trace()
            # print(e)
            # print(solution_i)
            error_ls.append(problem)
            solution = solution_i

        solution_dict[problem] = solution
        cnt += 1
        print(f"{cnt} done")

    print(error_ls)

    with open('solution_llama3_2_3B.json', 'w') as f:
        json.dump(solution_dict, f)
    
    set_trace()


def build_prompt(tokenizer, problem):
    user_prompt = f"Solve the given Java programming problem. Problem:\n{problem}"

    messages = [{"role": "user", "content": user_prompt}]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    return inputs

def extract_method_name(code):
    code_lines = code.split('\n')
    method_signature = code_lines[0].split('(')[0]
    method_parts = method_signature.split(' ')
    method_name = method_parts[-1]
    # print('Method name:', method_name)

    return method_name

def wrap_method_into_class(student_code, class_name="StudentSolution"):
    """
    Wrap a standalone Java method into a compilable class with a minimal main().
    Returns the full Java class source as a string.
    """
    # Minimal sanity: ensure it looks like a method (has '(' and ')', and an opening brace after signature).
    if "(" not in student_code or ")" not in student_code or "{" not in student_code:
        raise ValueError("student_code does not look like a Java method.")

    JAVA_MAIN_STUB = """public static void main(String[] args) { 
}
"""

    method_name = extract_method_name(student_code)

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

def error_process(error_msg):
    if isinstance(error_msg, float):
        return {"error count": 0, "error types": []}

    error_lines = error_msg.strip().split('\n')
    last_line = error_lines[-1]
    error_cnt = last_line.split(' ')[0]

    error_types = set()
    for line in error_lines[:-1]:
        if 'compiler_code/StudentSolution.java' in line:
            error_type = line.split('error:')[-1].strip()
            error_types.add(error_type)

    return {"error count": int(error_cnt), "error types": list(error_types)}


def main():
    ## Run incorrect code through compiler to collect binary compiler indicator and compiler feedback
    # df = pd.read_pickle('data/dataset_wo_emb_time.pkl')

    # problem_sub = df.drop_duplicates(subset='ProblemID')
    # # code_dict = problem_sub.set_index('prompt')['Code'].to_dict()
    # # problem_method_dict = extract_method_name(code_dict)

    # subset = df.drop_duplicates(subset=['SubjectID', 'ProblemID'],keep='first').reset_index(drop=True)
    # subset = subset[subset['Score_x'] != 1]

    # # subset = subset[subset['Score_x'] == 0]
    # set_trace()
    # sub_zero = subset[subset['Score_x'] == 0.0].copy()
    # sub_nonzero = subset[subset['Score_x'] != 0.0].copy()

    # sub_nonzero['compilable'] = 1
    # sub_nonzero['compile_stderr'] = ""

    # compile_results = sub_zero['Code'].apply(compile_student_method)

    # sub_zero['compilable'] = compile_results.apply(lambda x: 1 if x['ok'] else 0)
    # sub_zero['compile_stderr'] = compile_results.apply(lambda x: x['stderr'])

    # df_final = pd.concat([sub_zero, sub_nonzero], ignore_index=True)
   
    # Preprocess compiler dataset to create column: 'error cnt' and 'error types'
    df = pd.read_csv('data/data_compile_res.csv')
    check = df['compile_stderr'].tolist()

    error_info = df['compile_stderr'].apply(error_process)
    df['error_cnt'] = error_info.apply(lambda x: x['error count'])
    df['error_types'] = error_info.apply(lambda x: x['error types'])

    set_trace()
    df.to_csv('data/data_compile_res_processed.csv')


    ## Generate sample solutions using Llama 3.2 3B Instruct
    # device = torch.device('cuda')
    # model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-3.2-3B-Instruct", device)
    # save_problem_solutions(problem_sub, model=model, tokenizer=tokenizer, device=device)

    ## Evaluate correctness of generated solutions using gpt-4o
    # # with open('solution_gpt4o.json', 'r') as f:
    # #     gen_sol = json.load(f)

    # #     cor_dict = {}
    # #     cnt = 1
    # #     for key, val in gen_sol.items():
    # #         reply_json = get_solution_correctness([key, val])
    # #         decision = reply_json['Code correctness']
    # #         cor_dict[key] = decision.lower()

    # #         if decision.lower() == 'incorrect':
    # #             print(f'{cnt} Incorrect Problem: {key}\n\nCode: {val}\n')
    # #             fixed_code = reply_json["Corrected code"]
    # #             # gen_sol[key] = fixed_code
            
    # #         print(f'{cnt} processed')
    # #         cnt += 1
        
    # #     res = list(cor_dict.values())
    # #     cor_cnt = res.count('correct')

    #     # with open('solution_llama_32_fixed.json', 'w') as sa:
    #     #     json.dump(gen_sol, sa)

    #     # set_trace()
    #     # print(cor_cnt)

    #     # TODO: try new Qwen model

# main()
