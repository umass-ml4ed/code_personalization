"""Generation and CodeBLEU evaluation used by the released experiments."""

import json
import os

import torch
from tqdm import tqdm
from transformers import GenerationConfig
from vllm import SamplingParams

from evaluator.CodeBLEU import calc_code_bleu
from error_metrics import (
    error_set_iou,
    evaluate_error_metrics,
    problem_error_coverage,
    submission_error_iou,
)


def inference(model, dataloader, tokenizer, device, configs):
    """Generate ``configs.k`` code submissions for every input prompt."""
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    tokenizer.padding_side = "left"

    predictions = []
    generation_config = GenerationConfig(
        do_sample=configs.k > 1,
        temperature=0.7 if configs.k > 1 else None,
        top_p=1.0,
        top_k=40,
        repetition_penalty=1.1,
        num_return_sequences=configs.k,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=configs.max_new_tokens,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                generation_config=generation_config,
            )
            decoded = tokenizer.batch_decode(
                outputs[:, input_ids.size(1):], skip_special_tokens=True
            )
            predictions.extend(item.strip() for item in decoded)

    return predictions


def evaluate(ground_truth_code, predictions, problems, configs, run_name, lang="java"):
    """Calculate top-1 or oracle top-k CodeBLEU without case selection."""
    expected = len(ground_truth_code) * configs.k
    if len(predictions) != expected:
        raise ValueError(
            f"Expected {expected} predictions for {len(ground_truth_code)} examples "
            f"with k={configs.k}, received {len(predictions)}."
        )

    if configs.k == 1:
        grouped_predictions = predictions
        codebleu, detailed = compute_code_bleu(
            ground_truth_code, grouped_predictions, lang
        )
        metrics = {"codebleu": codebleu, "detailed_codebleu": detailed}
    else:
        grouped_predictions = [
            predictions[index:index + configs.k]
            for index in range(0, len(predictions), configs.k)
        ]
        per_example_scores = [
            max(
                compute_code_bleu([target], [candidate], lang)[0]
                for candidate in candidates
            )
            for target, candidates in zip(ground_truth_code, grouped_predictions)
        ]
        metrics = {f"top_{configs.k}_codebleu": sum(per_example_scores) / len(per_example_scores)}

    print(f"results: {metrics}")
    results = {
        **metrics,
        "generated_codes": grouped_predictions,
        "ground_truth_codes": ground_truth_code,
        "problems": problems,
    }

    if configs.save_model:
        output_dir = os.path.join(configs.model_save_dir, run_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"evaluation_k{configs.k}.json")
        with open(output_path, "w") as output_file:
            json.dump(results, output_file, indent=2)

    return results


def evaluate_vllm(
    model,
    lora_request,
    tokenizer,
    prompts,
    ground_truth_code,
    run_name,
    configs,
    lang="java",
):
    """Generate with vLLM and evaluate every dataset example."""
    del tokenizer, run_name  # Kept in the interface for compatibility.
    predictions = []
    sampling_params = SamplingParams(
        n=configs.k,
        temperature=0.7,
        max_tokens=configs.max_new_tokens,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.1,
    )

    for index in tqdm(range(0, len(prompts), 16), desc="inference", leave=False):
        outputs = model.generate(
            prompts[index:index + 16],
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        for output in outputs:
            predictions.extend(item.text.strip() for item in output.outputs)

    return evaluate(
        ground_truth_code,
        predictions,
        [None] * len(ground_truth_code),
        configs,
        "",
        lang=lang,
    )


def compute_code_bleu(ground_truth_codes, generated_codes, lang="java"):
    params = "0.25,0.25,0.25,0.25"
    return calc_code_bleu.get_codebleu(
        pre_references=[ground_truth_codes],
        hypothesis=generated_codes,
        lang=lang,
        params=params,
    )


def compute_code_bleu_modified(ground_truth_codes, generated_codes, lang="java"):
    params = "0.25,0.25,0.25,0.25"
    return calc_code_bleu.get_codebleu_modified(
        pre_references=[ground_truth_codes],
        hypothesis=generated_codes,
        lang=lang,
        params=params,
    )
