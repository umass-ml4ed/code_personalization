import os
import json
from datetime import datetime

import torch
import torch.optim as optim
import transformers
import hydra
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm

from data_loader_sft import (
    build_input_with_special_tokens,
    build_input_with_special_tokens_ability,
    build_prompt_with_special_tokens,
    build_prompt_with_special_tokens_ability,
    get_inputs,
    get_inputs_ability,
    make_dataloader,
    read_data,
)
from eval_sft import evaluate, inference
from model import create_model, load_model_eval
from trainer import generator_step
from utils import aggregate_metrics, set_random_seed


@hydra.main(version_base=None, config_path=".", config_name="configs_sft")
def main(configs):
    torch.cuda.empty_cache()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(now)

    set_random_seed(configs.seed)
    
    # Use wandb to track experiment
    if configs.log_wandb:
        wandb.login()
        wandb.init(project=configs.wandb_project)
        print('Run id:', wandb.run.id)
        wandb.config.update(OmegaConf.to_container(configs, resolve=True))
    
    if configs.save_model:
        save_dir = f"{now}_{configs.split_by}"
        os.makedirs(os.path.join(configs.model_save_dir, save_dir), exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create LLM and tokenizer
    model, tokenizer = create_model(configs, device)

    # The submission experiments use only the first AST-convertible attempt for
    # each student/problem pair (enforced by data_loader_sft.read_data).
    train_ids, valid_ids, test_ids, df, _ = read_data(configs.data_file, configs)

    language = 'python' if 'falcon' in configs.kc_path.lower() else 'java'
    print('Dataset language:', language)
       
    print('Splitting based on {}.'.format(configs.split_by))

    split_col = 'ProblemID' if configs.split_by == 'problem' else 'SubjectID'

    train_set = df[df[split_col].isin(train_ids)].reset_index(drop=True)
    valid_set = df[df[split_col].isin(valid_ids)].reset_index(drop=True)
    test_set = df[df[split_col].isin(test_ids)].reset_index(drop=True)


    # Raw input SFT with problem statement only
    if not configs.student_modeling:
        train_problem_ls, train_code_ls = get_inputs(train_set)
        valid_problem_ls, valid_code_ls = get_inputs(valid_set)
        test_problem_ls, test_code_ls = get_inputs(test_set)

        train_completions = [build_input_with_special_tokens(prompt, code, tokenizer) for prompt, code in zip(train_problem_ls, train_code_ls)]
        valid_completions = [build_input_with_special_tokens(prompt, code, tokenizer) for prompt, code in zip(valid_problem_ls, valid_code_ls)]
        test_completions = [build_input_with_special_tokens(prompt, code, tokenizer) for prompt, code in zip(test_problem_ls, test_code_ls)]

    else:
        # Input SFT with problem statement + ability level + knowledge component
        train_problem_ls, train_code_ls, train_ability_ls, train_kc_ls = get_inputs_ability(train_set)
        valid_problem_ls, valid_code_ls, valid_ability_ls, valid_kc_ls = get_inputs_ability(valid_set)
        test_problem_ls, test_code_ls, test_ability_ls, test_kc_ls = get_inputs_ability(test_set)

        if configs.testing:
            train_problem_ls, train_code_ls, train_ability_ls, train_kc_ls = train_problem_ls[:3], train_code_ls[:3], train_ability_ls[:3], train_kc_ls[:3]
            valid_problem_ls, valid_code_ls, valid_ability_ls, valid_kc_ls = valid_problem_ls[:3], valid_code_ls[:3], valid_ability_ls[:3], valid_kc_ls[:3]
            test_problem_ls, test_code_ls, test_ability_ls, test_kc_ls = test_problem_ls[:3], test_code_ls[:3], test_ability_ls[:3], test_kc_ls[:3]
            configs.epochs = 1
            print("Testing mode: using a small subset of the data for quick runs.")


        train_completions = [build_input_with_special_tokens_ability(prompt, code, tokenizer, ability, kc, language=language) for prompt, code, ability, kc in zip(train_problem_ls, train_code_ls, train_ability_ls, train_kc_ls)]
        valid_completions = [build_input_with_special_tokens_ability(prompt, code, tokenizer, ability, kc, language=language) for prompt, code, ability, kc in zip(valid_problem_ls, valid_code_ls, valid_ability_ls, valid_kc_ls)]
        test_completions = [build_input_with_special_tokens_ability(prompt, code, tokenizer, ability, kc, language=language) for prompt, code, ability, kc in zip(test_problem_ls, test_code_ls, test_ability_ls, test_kc_ls)]

    train_dl = make_dataloader(train_completions, tokenizer, batch_size=configs.batch_size, train=True)
    valid_dl = make_dataloader(valid_completions, tokenizer, batch_size=configs.batch_size, train=True)
    test_dl = make_dataloader(test_completions, tokenizer, batch_size=configs.batch_size, train=True)

    optimizers_generator = []
    optimizer_lm = optim.AdamW(model.parameters(), lr=configs.lr)
    optimizers_generator.append(optimizer_lm)

    # LR scheduler
    num_training_steps = len(train_dl) * configs.epochs
    num_warmup_steps = configs.warmup_ratio * num_training_steps
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer_lm, num_warmup_steps, num_training_steps)


    best_valid_metrics =  {'loss': float('inf')} 
    best_test_metrics =  {'loss': float('inf')} 
    best_metrics_with_valid =  {'loss': float('inf')} 
    train_dl_len = len(train_dl)

    for ep in tqdm(range(configs.epochs), desc="epochs", mininterval=20.0):
        train_logs, test_logs, valid_logs = [], [], []

        # training
        for idx, batch in enumerate(tqdm(train_dl, desc="training", leave=False)):
            train_log = generator_step(idx, batch, model, tokenizer, device, optimizers=optimizers_generator, configs=configs, train_dl_len=train_dl_len, train=True, scheduler=scheduler)
            train_logs.append(train_log)

            ## save results to wandb
            if configs.log_train_every_itr and configs.log_wandb:
                if (idx+1) % configs.log_train_every_itr == 0:
                    itr_train_logs = aggregate_metrics(train_logs, configs)
                    for key in itr_train_logs:
                        wandb.log({"metrics/train_every_{}_itr/{}".format(configs.log_train_every_itr,key): itr_train_logs[key]})
        
        # validation
        for idx, batch in enumerate(tqdm(valid_dl, desc="validation", leave=False)):
            valid_log = generator_step(idx, batch, model, tokenizer, device, configs=configs, train=False)
            valid_logs.append(valid_log)

        # testing
        for idx, batch in enumerate(tqdm(test_dl, desc="testing", leave=False)):
            test_log = generator_step(idx, batch, model, tokenizer, device, configs=configs, train=False)
            test_logs.append(test_log)

       
        train_logs = aggregate_metrics(train_logs, configs)
        valid_logs = aggregate_metrics(valid_logs, configs)
        test_logs  = aggregate_metrics(test_logs, configs)

        ## log the results and save models
        for key in valid_logs:
            if key == 'loss':
                if( float(valid_logs[key]) < best_valid_metrics[key] ):
                    best_valid_metrics[key] = float(valid_logs[key])
                    for key_best_metric in best_metrics_with_valid:
                        best_metrics_with_valid[key_best_metric] = float(test_logs[key_best_metric])
                    ## Save the model with lowest validation loss
                    print('Saved at Epoch:', ep)
                    print('Best model stats:', test_logs)
                    if configs.save_model:
                        if configs.log_wandb:
                            wandb.log({"best_model_at_epoch": ep, "best_valid_loss": best_valid_metrics[key]})

                        model_dir = os.path.join(configs.model_save_dir, save_dir, 'model')
                        model.save_pretrained(model_dir)
                        tokenizer_dir = os.path.join(configs.model_save_dir, save_dir, 'tokenizer')
                        tokenizer.save_pretrained(tokenizer_dir)

                        scheduler_dir = os.path.join(configs.model_save_dir, save_dir, 'scheduler.pth')
                        torch.save(scheduler.state_dict(), scheduler_dir)

                        optimizer_lm_dir = os.path.join(configs.model_save_dir, save_dir, 'optimizer_lm.pth')
                        torch.save(optimizer_lm.state_dict(), optimizer_lm_dir)

        for key in test_logs:
            if key == 'loss':
                if float(test_logs[key])<best_test_metrics[key]:
                    best_test_metrics[key] = float(test_logs[key])
        
        ## save results to wandb:
        if configs.log_wandb:
            saved_stats = {}
            for key in train_logs:
                saved_stats["metrics/train/"+key] = train_logs[key]
            for key in valid_logs:
                saved_stats["metrics/valid/"+key] = valid_logs[key]
            for key in test_logs:
                saved_stats["metrics/test/"+key] = test_logs[key]
            for key in best_test_metrics:
                saved_stats["metrics/test/best_"+key] = best_test_metrics[key]
            for key in best_metrics_with_valid:
                saved_stats["metrics/test/best_"+key+"_with_valid"] = best_metrics_with_valid[key]
            saved_stats["epoch"] = ep

            wandb.log(saved_stats)


    del model
    torch.cuda.empty_cache()
    print("Training completed. Start Evaluation...")

    if not configs.save_model:
        if configs.log_wandb:
            wandb.finish()
        return

    model, tokenizer = load_model_eval(configs, save_dir, device)
    if not configs.student_modeling:
        test_prompt = [build_prompt_with_special_tokens(prompt, tokenizer) for prompt in test_problem_ls]
    else:
        test_prompt = [build_prompt_with_special_tokens_ability(prompt, tokenizer, ability, kc, language=language) for prompt, ability, kc in zip(test_problem_ls, test_ability_ls, test_kc_ls)]
    
    inference_dl = make_dataloader(test_prompt, tokenizer, batch_size=4, train=False)
    generated_output = inference(model, inference_dl, tokenizer, device, configs)

    if configs.save_model:
        with open(os.path.join(configs.model_save_dir, save_dir, 'generated_output.txt'), 'w') as f:
            json.dump(generated_output, f, indent=2)

    results = evaluate(test_code_ls, generated_output, test_problem_ls, configs, save_dir, lang=language)
    if configs.k == 1:
        result = {'codeBLEU': results['codebleu']}
    else:
        result = {'top_{}_codeBLEU'.format(configs.k): results['top_{}_codebleu'.format(configs.k)]}


    if configs.log_wandb:
        wandb.log(result)
        wandb.finish()


if __name__ == "__main__":
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
