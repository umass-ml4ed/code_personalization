# KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks
This repo contains the code for the paper, <a href="https://arxiv.org/abs/2601.06633//">KASER: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks</a>, by Zhangqi Duan, Nigel Fernandez, and Andrew Lan, published at ACL 2026. In this repo, we release the code for SFT training and GRPO training with our designed reward functions. 

If you find this code useful, please cite us!
```
@inproceedings{duan-etal-2026-kaser,
    title = "{KASER}: Knowledge-Aligned Student Error Simulator for Open-Ended Coding Tasks",
    author = "Duan, Zhangqi  and
      Fernandez, Nigel  and
      Lan, Andrew",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1858/",
    doi = "10.18653/v1/2026.acl-long.1858",
    pages = "39988--40006",
    ISBN = "979-8-89176-390-6"
}
```

## Training

The following will train, test, and evaluate the model. We use Qwen2.5-Coder-7B-Instruct as the base model. 
### SFT
```
python sft.py
```

### GRPO
```
python grpo.py --data_path data/grpo_student_knowledge_notrunc.pkl --with_knowledge --sft_checkpoint <SFT_CHECKPOINT_NAME>
```
