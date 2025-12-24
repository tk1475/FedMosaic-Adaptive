import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import random

class RealWorldDataset(Dataset):
    def __init__(self, hf_data, task_type):
        self.data = hf_data
        self.task_type = task_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # --- MAPPING VISUAL TASKS TO TEXT PROXIES ---
        
        # 1. Relation Tasks (Proxy for Visual Relation) -> MNLI
        if self.task_type == "relation":
            premise = item['premise']
            hypothesis = item['hypothesis']
            # Labels: 0=entailment, 1=neutral, 2=contradiction
            label_map = {0: "Yes", 1: "Maybe", 2: "No"}
            text = f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise imply the hypothesis? Answer:"
            label = label_map.get(item['label'], "Unknown")

        # 2. Reasoning Tasks (Proxy for Multi-modal Reasoning) -> CommonsenseQA
        elif self.task_type == "reasoning":
            question = item['question']
            choices = item['choices'] # dict with 'label' and 'text' lists
            # Format choices nicely
            choice_str = ", ".join([f"({l}) {t}" for l, t in zip(choices['label'], choices['text'])])
            text = f"Question: {question}\nOptions: {choice_str}\nChoose the best answer:"
            label = item['answerKey']

        # 3. Fact Retrieval (Proxy for VQA) -> SQuAD
        elif self.task_type == "vqa_proxy":
            context = item['context']
            question = item['question']
            text = f"Context: {context}\nQuestion: {question}\nAnswer:"
            # SQuAD answers are lists, take the first valid one
            label = item['answers']['text'][0] if len(item['answers']['text']) > 0 else "Unknown"

        # 4. Math/Logic (Proxy for Complex Reasoning) -> GSM8K
        elif self.task_type == "math":
            question = item['question']
            text = f"Solve this step-by-step: {question}\nAnswer:"
            label = item['answer']

        return f"{text} {label}"

class DataManager:
    def __init__(self, batch_size=4):
        self.batch_size = batch_size
        
    def setup_real_world_benchmark(self):
        print("--- Initializing 'Scaled-Up' Benchmark (Mimicking DRAKE) ---")
        
        datasets = []
        
        # --- Task Group A: Relation Understanding (Clients 0, 1, 2) ---
        # Mimics: Fashion Relation, Spatial Relation
        print("Loading MNLI (Relation Proxy)...")
        ds_mnli = load_dataset("glue", "mnli", split="train[:1000]")
        # Partition 1000 samples into 3 chunks
        datasets.append(RealWorldDataset(ds_mnli.select(range(0, 300)), "relation"))
        datasets.append(RealWorldDataset(ds_mnli.select(range(300, 600)), "relation"))
        datasets.append(RealWorldDataset(ds_mnli.select(range(600, 900)), "relation"))

        # --- Task Group B: Reasoning (Clients 3, 4, 5) ---
        # Mimics: Visual Figurative, Context-dependent Reasoning
        print("Loading CommonsenseQA (Reasoning Proxy)...")
        ds_cqa = load_dataset("tau/commonsense_qa", split="train[:1000]")
        datasets.append(RealWorldDataset(ds_cqa.select(range(0, 300)), "reasoning"))
        datasets.append(RealWorldDataset(ds_cqa.select(range(300, 600)), "reasoning"))
        datasets.append(RealWorldDataset(ds_cqa.select(range(600, 900)), "reasoning"))

        # --- Task Group C: Fact Retrieval (Clients 6, 7, 8) ---
        # Mimics: General VQA, TextVQA
        print("Loading SQuAD (VQA Proxy)...")
        ds_squad = load_dataset("rajpurkar/squad", split="train[:1000]")
        datasets.append(RealWorldDataset(ds_squad.select(range(0, 300)), "vqa_proxy"))
        datasets.append(RealWorldDataset(ds_squad.select(range(300, 600)), "vqa_proxy"))
        datasets.append(RealWorldDataset(ds_squad.select(range(600, 900)), "vqa_proxy"))

        # --- Task Group D: Complex Logic (Client 9 - The 'Expert') ---
        # Mimics: Unseen Tasks / Expert Domains
        print("Loading GSM8K (Math Expert)...")
        ds_math = load_dataset("gsm8k", "main", split="train[:300]")
        datasets.append(RealWorldDataset(ds_math, "math"))
        
        print(f"✅ Created 10 Clients covering 4 Distinct Cognitive Domains.")
        return datasets