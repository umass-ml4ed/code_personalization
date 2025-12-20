import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace
import numpy as np
from torch.utils.checkpoint import checkpoint



# SFT training step
def generator_step(idx, batch, model, tokenizer, device, optimizers=None, configs=None, train_dl_len=None, train=True, scheduler=None):
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if train:
        model.train()
    else:
        model.eval()
    
    b_inpids, b_attnids, b_labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
    
    if train:
        model_outputs = model(input_ids=b_inpids, attention_mask=b_attnids, labels=b_labels)
    else:
        with torch.no_grad():
            model_outputs = model(input_ids=b_inpids, attention_mask=b_attnids, labels=b_labels)

    loss_batch = model_outputs.loss

    log = {'loss': loss_batch.cpu().detach()}

    if train:
        loss_batch = loss_batch / configs.accum_iter
        loss_batch.backward()

    if train:
        if (idx+1) % configs.accum_iter == 0 or idx == train_dl_len - 1:
            for optimizer in optimizers:
                optimizer.step()
            if configs.use_scheduler:
                scheduler.step()
            for optimizer in optimizers:
                optimizer.zero_grad()
    
    
    return log
    

# KCGEN-KT training step
def generator_kc_step(idx, batch, model, lstm, tokenizer, optimizers=None, optimizers_lstm=None,
                   configs=None, train_dl_len=None, train=True, scheduler=None, device=None, 
                   group_size=2, multitask=False, predictor=None, pred_loss_fn=None, 
                   optimizers_multitask=None, kc_loss_fn=None, trans_linear=None, optimizers_trans=None):
    
    eps = 1e-8
    if train:
        assert(optimizers != None)
        assert(optimizers_lstm != None)
        model.train()
        lstm.train()
        predictor.train()
    else:
        model.eval()
        lstm.eval()
        predictor.eval()

    # assemble generator input, make the timestep alignment for inputs to LSTM and generatot model later
    padded_scores, padded_inputs, padded_input_ids_ls, padded_attention_mask_ls, padded_labels_ls, padded_prompt_id_lens_ls, padded_kc, level_loc_ls = batch[0][1:], batch[1][:-1], batch[2][:, 1:], batch[3][1:], batch[4][1:], batch[5][1:], batch[6][1:], batch[7]
    set_trace()

    range_tensor = torch.arange(padded_attention_mask_ls.size(2), device=device).unsqueeze(0).unsqueeze(0) # Shape = [1, 1, max_length]
    range_tensor = range_tensor.repeat(padded_attention_mask_ls.size(0), padded_attention_mask_ls.size(1), 1) # Shape = [T, B, max_length]
    mask_tensor = (range_tensor >= padded_prompt_id_lens_ls.unsqueeze(-1)) # Shape = [T, B, max_length]

    input_wte, kc_prod = update_input_weight(padded_input_ids_ls, padded_inputs, padded_kc, level_loc_ls, device, model, lstm, tokenizer, configs.transition, trans_linear, configs.kc_loss_method)

    T, B, max_length, D = input_wte.shape
    input_wte = input_wte.reshape((T * B), max_length, D)
    padded_attention_mask = padded_attention_mask_ls.reshape((T * B), -1)
    padded_label = padded_labels_ls.reshape((T * B), max_length)

    input_wte_groups = torch.split(input_wte, group_size)
    attention_mask_groups = torch.split(padded_attention_mask, group_size)
    label_groups = torch.split(padded_label, group_size)

    padded_scores = torch.unsqueeze(padded_scores, -1)
    padded_scores = padded_scores.reshape((T * B), -1)
    score_groups = torch.split(padded_scores, group_size)

    kc_prod = torch.unsqueeze(kc_prod, -1)
    kc_prod = kc_prod.reshape((T * B), -1)
    kc_prod_groups = torch.split(kc_prod, group_size)

    # Mask hidden states for label embeddings
    padded_mask = mask_tensor.reshape((T * B), -1)
    mask_groups = torch.split(padded_mask, group_size)

    pred_cum_loss, kc_cum_loss = 0.0, 0.0
    pred_cnt, kc_cnt = 0, 0

    pred_total = torch.tensor([]).to(device)
    gt_total = torch.tensor([]).to(device)
    logits_total = torch.tensor([]).to(device)

    for i in range(len(input_wte_groups)):
        input_wte_sub = input_wte_groups[i]
        attention_mask_sub = attention_mask_groups[i]
        label_sub = label_groups[i]

        # forward generator
        if train:
            outputs = model(inputs_embeds=input_wte_sub, attention_mask=attention_mask_sub, labels=label_sub, output_hidden_states=True, return_dict=True)

            activation = outputs['hidden_states']

            mask_sub = mask_groups[i]
            hidden_states = outputs['hidden_states'][-1]

            # Question emebedding 
            mask_expand = torch.unsqueeze(mask_sub, -1)
            hidden_states_question = hidden_states * ~mask_expand
            pooled_out = hidden_states_question.sum(dim=1)
            ques_cnt = torch.sum(~mask_expand, dim=1)
            pooled_out = pooled_out / (ques_cnt + eps)

            logits = predictor(pooled_out)
        
        else:
            with torch.no_grad():
                outputs = model(inputs_embeds=input_wte_sub, attention_mask=attention_mask_sub, labels=label_sub, output_hidden_states=True, return_dict=True)

                mask_sub = mask_groups[i]
                hidden_states = outputs['hidden_states'][-1]

                mask_expand = torch.unsqueeze(mask_sub, -1)
                hidden_states_question = hidden_states * ~mask_expand
                pooled_out = hidden_states_question.sum(dim=1)
                ques_cnt = torch.sum(~mask_expand, dim=1)
                pooled_out = pooled_out / (ques_cnt + eps)

                logits = predictor(pooled_out)
        

        # KC error rate loss
        score_sub = score_groups[i]
        kc_prod_sub = kc_prod_groups[i]

        kc_loss_sub = kc_loss_fn(kc_prod_sub[score_sub != -100], score_sub[score_sub != -100]).sum()

        kc_cum_loss += kc_loss_sub
        kc_cnt += score_sub[score_sub != -100].shape[-1]

        # prediction loss
        gt_total = torch.cat((gt_total, score_sub), 0)
        pred = (torch.sigmoid(logits) > 0.5) * 1
        pred_total = torch.cat((pred_total, pred), 0)
        logits_total = torch.cat((logits_total, logits), 0)

        pred_loss_sub = pred_loss_fn(logits[score_sub != -100], score_sub[score_sub != -100]).sum()
        pred_cum_loss += pred_loss_sub
        pred_cnt += logits[score_sub != -100].shape[-1]


    kc_cum_loss = kc_cum_loss / kc_cnt

    if multitask:
        pred_cum_loss = pred_cum_loss / pred_cnt
    

    # Code Prediction loss + kc loss + correctness loss
    total_loss = kc_cum_loss + pred_cum_loss

    norm_loss = configs.alpha * (pred_cum_loss / (pred_cum_loss.detach() + eps)) + (1 - configs.alpha) * kc_cum_loss / (kc_cum_loss.detach() + eps)
        

    # Adding gradient accumulation for training
    if train:
        norm_loss.backward()

    # optimization
    if train:
        if (idx+1) % configs.accum_iter == 0 or idx == train_dl_len - 1:
            for optimizer in optimizers:
                optimizer.step()
            if configs.use_scheduler:
                scheduler.step()
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            for optimizer in optimizers_lstm:
                optimizer.step()
            for optimizer in optimizers_lstm:
                optimizer.zero_grad()
            
            if configs.transition:
                for optimizer in optimizers_trans:
                    optimizer.step()
                for optimizer in optimizers_trans:
                    optimizer.zero_grad()

            for optimizer in optimizers_multitask:
                optimizer.step()
            for optimizer in optimizers_multitask:
                optimizer.zero_grad()


    # # random baseline:
    # logits_total = torch.randn(pred_total.shape).to(device)
    # pred_total = (torch.sigmoid(logits_total) > 0.5) * 1

    # # # majority baseline:
    # logits_total = torch.ones(pred_total.shape).to(device)
    # pred_total = torch.ones(pred_total.shape).to(device)

    log = {'loss': total_loss.cpu().detach(), 'kc_loss': kc_cum_loss.cpu().detach(), 'predictor_loss': pred_cum_loss.cpu().detach()}
    
    pred_res = pred_total[gt_total != -100].detach().cpu() == gt_total[gt_total != -100].detach().cpu()
    log['acc'] = pred_res
    if configs.binary_loss_fn == 'BCE':
        log['auc'] = {'logits': logits_total[gt_total != -100].detach().cpu(), 'scores': gt_total[gt_total != -100].detach().cpu()}

    return log


def predict_mastery_level(padded_inputs, padded_kc, lstm, trans=False, trans_linear=None):
    ks, hidden = lstm(padded_inputs)

    if trans:
        ks = trans_linear(ks)
    
    ks = torch.sigmoid(ks)

    padding_mask = padded_kc == -1
    safe_indices = padded_kc.clone().to(dtype=torch.int64)
    safe_indices[padding_mask] = 0

    result_kc = torch.gather(ks, 2, safe_indices)
    result_kc[padding_mask] = -1
    result_kc = result_kc.transpose(0, 1)   # shape: (B, T, max_kc_len)
    ks = ks.transpose(0, 1)

    return result_kc, ks



def update_input_weight(padded_input_ids_ls, padded_inputs, padded_kc, level_loc_ls, device, model, lstm, tokenizer, trans=False, trans_linear=None, kc_loss_method='mean', dpo_training=False):
    # Get embedding for True & False for replacing the embedding of ? to actual mastery level
    true_token = tokenizer.convert_tokens_to_ids('True')
    false_token = tokenizer.convert_tokens_to_ids('False')
    
    # Convert padded_input_ids to input_weight
    generator_input_wte = model.base_model.model.model.embed_tokens(padded_input_ids_ls)  # shape: (B, T, max_len, 4096), T is not aligned yet
    true_emb = model.base_model.model.model.embed_tokens(torch.tensor(true_token).to(device))
    false_emb = model.base_model.model.model.embed_tokens(torch.tensor(false_token).to(device))

    result_kc, kc = predict_mastery_level(padded_inputs, padded_kc, lstm, trans, trans_linear)

    B, T = padded_input_ids_ls.shape[0], padded_input_ids_ls.shape[1]

    # Start updating
    input_wte = generator_input_wte.clone()
   
    kc_prod_total = torch.zeros(B, T, requires_grad=True).to(device)
    kc_prod_total = kc_prod_total.clone()

    for i in range(B):        
        sent_ind_i, token_ind_i = level_loc_ls[i]
        res_kc_i = result_kc[i]  # shape: (T, max_kc_len)
        input_weight_copy = input_wte[i]
        kc_prod_i = kc_prod_total[i]

        position_dict = {}
        for s, t in zip(sent_ind_i.tolist(), token_ind_i.tolist()):
            if s not in position_dict:
                position_dict[s] = ([], [])
            position_dict[s][0].append(s)
            position_dict[s][1].append(t)

        for key_sent_ind, (sub_sent_ind, sub_token_ind) in position_dict.items():
            sub_sent_ind = torch.tensor(sub_sent_ind)
            sub_token_ind = torch.tensor(sub_token_ind)
            
            k = sub_token_ind.size(0)

            selected_values = res_kc_i[key_sent_ind, :k]

            if kc_loss_method == 'prod':
                kc_prod = selected_values.prod()
                kc_prod_i[key_sent_ind] = kc_prod
            
            elif kc_loss_method == 'mean':
                kc_mean = selected_values.mean()
                kc_prod_i[key_sent_ind] = kc_mean

            else:
                kc_prod = selected_values.prod()
                kc_prod_mean = torch.pow(kc_prod, 1/k)
                kc_prod_i[key_sent_ind] = kc_prod_mean

            true_weighted = selected_values.unsqueeze(-1) * true_emb
            false_weighted = (1 - selected_values.unsqueeze(-1)) * false_emb
            new_values = true_weighted + false_weighted

            input_weight_copy[sub_sent_ind, sub_token_ind] = new_values

    input_wte = torch.transpose(input_wte, 0, 1)
    kc_prod_tensor = torch.transpose(kc_prod_total, 0, 1)

    if dpo_training:
        return input_wte, kc_prod_tensor, result_kc.transpose(0, 1)
    else:
        return input_wte, kc_prod_tensor



def trainer_kc_only(idx, batch, device, lstm, trans=False, trans_linear=None, optimizers_lstm=None, configs=None, train_dl_len=None, train=True, optimizers_trans=None, collecting_kc=False):
    eps = 1e-8
    if train:
        assert(optimizers_lstm != None)
        lstm.train()
    else:
        lstm.eval()

    # assemble generator input, make the timestep alignment for inputs to LSTM and generatot model later
    padded_scores, padded_inputs, padded_kc, padded_prompts, padded_students = batch[0][1:], batch[1][:-1], batch[2][1:], batch[3][1:], batch[4][1:]

    pred_score_ls, student_ls, prompt_ls, kc_ls = [], [], [], []
    result_kc, kc = predict_mastery_level(padded_inputs, padded_kc, lstm, trans, trans_linear)

    result_kc = result_kc.transpose(0, 1)  
    valid = padded_kc != -1
    denom = valid.sum(dim=2).clamp(min=1).float() 
    padding_mask = padded_kc == -1
    result_kc[padding_mask] = 0
    mean_prob = result_kc.sum(dim=2) / denom


    if collecting_kc:
        T, B, dim = padded_inputs.shape
        padded_result_kc = result_kc.reshape((T * B), -1)

        est_prob = torch.unsqueeze(mean_prob, -1).reshape((T * B), -1)

        flattened_students = [student_i for substudent in padded_students for student_i in substudent]
        flattened_prompts = [prompt_i for subprompt in padded_prompts for prompt_i in subprompt]

        kc_subset = torch.split(padded_result_kc, 1)
        prob_subset = torch.split(est_prob, 1)

        for i in range(len(kc_subset)):
            student = flattened_students[i]
            prompt = flattened_prompts[i]
            if student:
                kc_i = kc_subset[i]
                
                student_ls.append(student)
                prompt_ls.append(prompt)
                kc_ls.append(kc_i[0].tolist())

                predicted_score = prob_subset[i][0][0].item()
                pred_score_ls.append(predicted_score)

        return pred_score_ls, student_ls, prompt_ls, kc_ls



    # padded_scores has -100 for padded timestep positions
    valid_pos = padded_scores != -100  # shape: (T, B)

    # flatten valid predictions and targets
    preds = mean_prob[valid_pos]
    targets = padded_scores[valid_pos].to(device)

    bce_fn = nn.BCELoss(reduction='mean')
    loss = bce_fn(preds, targets)

    if train:
        loss.backward()

    if train:
        if (idx+1) % configs.accum_iter == 0 or idx == train_dl_len - 1:
            for optimizer in optimizers_lstm:
                optimizer.step()
            for optimizer in optimizers_lstm:
                optimizer.zero_grad()
            
            if configs.transition:
                for optimizer in optimizers_trans:
                    optimizer.step()
                for optimizer in optimizers_trans:
                    optimizer.zero_grad()
        
    log = {'loss': loss.detach().cpu()}

    pred_binary = (preds > 0.5) * 1
    pred_res = pred_binary.detach().cpu() == targets.detach().cpu()
    log['acc'] = pred_res
    log['auc'] = {'logits': preds.detach().cpu(), 'scores': targets.detach().cpu()}

    return log

