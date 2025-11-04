from preprocessing import *
import random
from tqdm import tqdm

def retrieve_activation(model, tokenizer, prompt_ls, response_ls):
    max_layer = model.config.num_hidden_layers
    texts = [p+a for p, a in zip(prompt_ls, response_ls)]
    layer_list = list(range(max_layer+1))

    response_avg = [[] for _ in range(max_layer+1)]

    for i in tqdm(range(len(texts)), desc="Generating activations"):
        text = texts[i]
        prompt = prompt_ls[i]
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

            # Start with average of response tokens first
            for layer in layer_list:
                response_avg[layer].append(outputs.hidden_states[layer][:, prompt_len:, :].mean(dim=1).detach().cpu())

            del outputs

    for layer in layer_list:
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)

    return response_avg



def main():
    ## Preprocessing to get binary compiler results with first submission
    # df = pd.read_pickle('data/dataset_wo_emb_time.pkl')

    # subset = df.drop_duplicates(subset=['SubjectID', 'ProblemID'],keep='first').reset_index(drop=True)
    # subset = subset[subset['Score_x'] != 1]


    # Load the saved df with compiled results
    subset = pd.read_csv('data/data_compile_res.csv', index_col="Unnamed: 0")
    subset.sort_values(by=['SubjectID', 'ServerTimestamp'], inplace=True)

    with open('solution_gpt4o.json', 'r') as sol:
        solution = json.load(sol)

        device = torch.device('cuda')
        model, tokenizer = load_model_and_tokenizer("meta-llama/Meta-Llama-3-8B-Instruct", device)


        combined_tensor = None
        scores_gt, code_gt_ls, problem_gt_ls, compilability_ls, error_cnt_ls, error_ls = [], [], [], [], [], []

        len_dff_ls = []

        for problem, sol_i in solution.items():
            prompt = build_prompt(tokenizer, problem)

            # Get the generated code activation
            gen_code_repr = retrieve_activation(model, tokenizer, [prompt], [sol_i])

            # Collect corresponding metadata: score, code, problem, error_cnt, error_types, compilable
            subset_i = subset[subset['prompt'] == problem]
            gt_scores = subset_i['Score_x'].tolist()
            scores_gt.extend(gt_scores)

            codes = subset_i['Code'].tolist()
            code_gt_ls.extend(codes)



            compilability = subset_i['compilable'].tolist()
            compilability_ls.extend(compilability)

            error_cnt = subset_i['error_cnt'].tolist()
            error_cnt_ls.extend(error_cnt)

            error_types = subset_i['error_types'].tolist()
            error_ls.extend(error_types)

            problem_gt_ls.extend([problem] * len(codes))

            # Get the student code activation
            problem_ls = [prompt] * len(subset_i)
            student_code_repr = retrieve_activation(model, tokenizer, problem_ls, codes)

            code_repr_diff = []

            for layer in reversed(range(len(gen_code_repr))):
                gen_repr_layer_i = gen_code_repr[layer]
                gen_code_repr_exp = gen_repr_layer_i.repeat(len(problem_ls), 1).float()

                gen_code_repr_exp.sub_(student_code_repr[layer].float())

                code_repr_diff.insert(0, gen_code_repr_exp)

                del gen_code_repr[layer]
                del student_code_repr[layer]
                torch.cuda.empty_cache()


            # Stack the results
            student_code_repr_diff = torch.stack(code_repr_diff, dim=0)
            
            if combined_tensor is None:
                combined_tensor = student_code_repr_diff   # dimension: [num_layer, n_sample, 4096]

            else:
                combined_tensor = torch.cat([combined_tensor, student_code_repr_diff], dim=1)


        torch.save(combined_tensor, f"response_avg_diff_inc.pt")

        # Save the metadata
        with open('gt_scores_all_inc.json', 'w') as f:
            json.dump(scores_gt, f)
        
        with open('problems_all_inc.json', 'w') as f:
            json.dump(problem_gt_ls, f)

        with open('codes_all_inc.json', 'w') as f:
            json.dump(code_gt_ls, f)

        with open('compilability_all_inc.json', 'w') as f:
            json.dump(compilability_ls, f)
        
        with open('error_cnt_all_inc.json', 'w') as f:
            json.dump(error_cnt_ls, f)
        
        with open('error_types_all_inc.json', 'w') as f:
            json.dump(error_ls, f)


# main()