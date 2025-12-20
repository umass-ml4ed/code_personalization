Code for Ability Aligned Code Generation

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

