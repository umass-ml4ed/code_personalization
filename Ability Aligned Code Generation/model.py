import torch
import torch.optim as optim
import os
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from pdb import set_trace
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# LSTM used for learning student knowledge space
def create_lstm_model(configs, device):
    lstm = nn.LSTM(configs.lstm_inp_dim, configs.transition_dim, num_layers=configs.num_layers)
    lstm.to(device)
    
    return lstm

def create_tokenizer(configs, now=None, folder_name=None):
    if now:
        if configs.model_save_dir == 'sft_checkpoints':
            tokenizer_dir = os.path.join(configs.model_save_dir, now, 'tokenizer')
        else:
            if folder_name:
                tokenizer_dir = os.path.join(configs.model_save_dir, now, folder_name)
            else:
                tokenizer_dir = os.path.join(configs.model_save_dir, now, now, 'model')

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(configs.base_model) 
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"

    return tokenizer

def create_model(configs, device):
    tokenizer = create_tokenizer(configs)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
        )
        
    model = AutoModelForCausalLM.from_pretrained(
            configs.base_model,
            quantization_config=bnb_config
        )
    
    lora_config = LoraConfig(
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            r=configs.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            inference_mode=False
        )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.to(device)

    return model, tokenizer

# A single layer classifier head for KT prediction
def create_dkt_classifier(device):
    classifier = nn.Linear(4096, 1) #3584 or 4096
    classifier.to(device)
    nn.init.xavier_uniform_(classifier.weight)

    return classifier


def lora_model_load(configs, device, now, continue_train, load_in_8bit=True, dpo_training=False):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        bnb_8bit_compute_dtype=torch.bfloat16 if continue_train else torch.float16
    )

    if configs.model_save_dir == 'sft_checkpoints':
        model_dir = os.path.join(configs.model_save_dir, now, 'model')
    elif configs.model_save_dir == 'grpo_checkpoints' or configs.model_save_dir == 'dpo_checkpoints':
        model_dir = os.path.join(configs.model_save_dir, now, now, 'model')


    peft_config = PeftConfig.from_pretrained(model_dir)
    _hf_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(_hf_model, model_dir, is_trainable=continue_train, adapter_name="default").to(device)


    if dpo_training:
        model.load_adapter(model_dir, is_trainable=False, adapter_name="lora_ref")

    return model



def load_model_eval(configs, now, device):
    model = lora_model_load(configs, device, now, False, load_in_8bit=True)
    model.eval()

    tokenizer = create_tokenizer(configs, now, folder_name='')
    tokenizer.padding_side = "left"

    return model, tokenizer

def create_knowledge_linear(device, hid_dim, transition_dim):
    if transition_dim == 64:
        linear = nn.Sequential(
        nn.ReLU(),
        nn.Linear(transition_dim, hid_dim),
    )

    else:
        linear = nn.Sequential(
            nn.Linear(transition_dim, 64),
            nn.ReLU(),  
            nn.Linear(64, hid_dim),
        )

    linear = linear.to(device)

    return linear


def load_kcgen_eval(configs, now, device, no_kc):
    if configs.save_model:
        model = lora_model_load(configs, device, now, False, load_in_8bit=True)
        model.eval()

        tokenizer = create_tokenizer(configs)

        lstm_hid_dim = no_kc
        if configs.transition:
            lstm = create_lstm_model(configs, device)
            lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm'), weights_only=True))
            trans_linear = create_knowledge_linear(device, lstm_hid_dim, configs.transition_dim)
            trans_linear.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'transition'), weights_only=True))

        
        predictor = create_dkt_classifier(device)
        predictor.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'predictor'), weights_only=True))
    
    return model, lstm, predictor, tokenizer, trans_linear


def load_model_eval_vllm(configs, now, device):
    llm = LLM(
        model=configs.base_model,
        enable_lora=True,
        max_lora_rank=16
    )

    adapter_path = os.path.join(configs.model_save_dir, now, 'model')
    lora_request = LoRARequest("default", 1, adapter_path)

    tokenizer_dir = os.path.join(configs.model_save_dir, now, 'tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    tokenizer.padding_side = "left"

    return llm, lora_request, tokenizer
