import torch
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset

class SimpleTextDataset(Dataset):
    """Wraps HF datasets into simple (text_input, label) pairs."""
    def __init__(self, hf_data, task_prompt):
        self.data = hf_data
        self.prompt = task_prompt
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        # Robust handling for different dataset columns
        text = item.get('question', item.get('sentence', ''))
        # For demo: just returning text. In real FL, we tokenize here.
        return f"{self.prompt} {text}"

class DataManager:
    def __init__(self, num_clients=10, batch_size=4):
        self.num_clients = num_clients
        self.batch_size = batch_size
        
    def setup_mini_drake(self):
        print("--- Setting up Mini-DRAKE Data ---")
        
        # 1. Download small slices of 4 distinct datasets
        # Task A: QA (Simulating VQA text part)
        ds_qa = load_dataset("tau/commonsense_qa", split="train[:200]")
        
        # Task B: Reasoning (ScienceQA)
        ds_sci = load_dataset("derek-thomas/ScienceQA", split="train[:200]")
        
        # Task C: Summary (CNN/DailyMail)
        ds_sum = load_dataset("cnn_dailymail", '3.0.0', split="train[:200]")
        
        # Task D: Unseen/Odd (Rotten Tomatoes)
        ds_odd = load_dataset("rotten_tomatoes", split="train[:200]")
        
        # 2. Assign to Clients (Heterogeneity)
        datasets = []
        
        # Clients 0-2: QA
        datasets.extend([SimpleTextDataset(ds_qa, "Answer:") for _ in range(3)])
        # Clients 3-5: Science
        datasets.extend([SimpleTextDataset(ds_sci, "Explain:") for _ in range(3)])
        # Clients 6-7: Summary
        datasets.extend([SimpleTextDataset(ds_sum, "Summarize:") for _ in range(2)])
        # Clients 8-9: Oddball
        datasets.extend([SimpleTextDataset(ds_odd, "Classify:") for _ in range(2)])
        
        print(f"Created {len(datasets)} heterogeneous client datasets.")
        return datasets