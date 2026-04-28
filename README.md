# KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks
This repo contains the code for the paper, <a href="https://https://arxiv.org/abs/2601.06633//">KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks</a>, by Zhangqi Duan, Nigel Fernandez, and Andrew Lan, published at ACL 2026. We will release the cleaned code soon. 

## Training

The following will train, test, and evaluate the model. We use Qwen2.5-Coder-7B-Instruct as the base model. 
### SFT
```
python sft.py
```

### GRPO
```
python grpo.py --data_path data/grpo_student_knowledge_notrunc.pkl --with_knowledge
```

