import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pqlora import PQLoRALayer
import gc
import os
import random
import time

# Constants
MODEL_SMALL = "Qwen/Qwen2.5-0.5B-Instruct" 
MODEL_LARGE = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SHARED_Ws = "Qwen/Qwen2.5-0.5B-Instruct" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ADAPTER_DTYPE = torch.float32 

class FederatedClientBlockWise:
    def __init__(self, client_id, model_type, dataset):
        self.id = client_id
        self.model_type = model_type
        self.dataset = dataset
        self.r = 16
        self.history = {"loss": []} 
        self.local_state = {} 

    def get_model_name(self):
        return MODEL_SMALL if self.model_type == "small" else MODEL_LARGE

    def _get_bnb_config(self):
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

    def train_and_rela(self, global_weights_dict, mode="grad"):
        # --- LITE CONFIG (Fast but Deep) ---
        BATCH_SIZE = 4  # Reduced back to 4 for speed
        
        # Randomly sample 4 examples
        indices = random.sample(range(len(self.dataset)), min(BATCH_SIZE, len(self.dataset)))
        raw_texts = [self.dataset[i] for i in indices]
        
        print(f"  > [C{self.id}] Training on 4 examples...")
        loss, self.local_state = self._phase_1_train(raw_texts, global_weights_dict)
        
        gc.collect(); torch.cuda.empty_cache()
        self.history["loss"].append(loss)
        print(f"    - Final Loss: {loss:.4f}")

        # Probe Phase
        if mode == "grad":
            vector = self._phase_2_gradient(raw_texts)
        else:
            vector = self._phase_2_representation(raw_texts)
            
        gc.collect(); torch.cuda.empty_cache()
        return {"weights": self.local_state, "vector": vector}

    def _phase_1_train(self, raw_texts, global_weights_dict):
        model_name = self.get_model_name()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        # Load Model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=self._get_bnb_config(), device_map=DEVICE
        )
        model.gradient_checkpointing_enable()
        model.eval() 

        # --- DEEP ADAPTERS (The key to learning) ---
        num_layers = len(model.model.layers)
        target_layers = range(num_layers - 4, num_layers) 
        self.active_adapters = nn.ModuleDict()
        self.hooks = []
        
        def get_hook(adapter_name):
            def hook_fn(module, input, output):
                if isinstance(output, tuple): h = output[0]
                else: h = output
                x = input[0].to(ADAPTER_DTYPE)
                delta = self.active_adapters[adapter_name](x).to(h.dtype)
                if isinstance(output, tuple): return (h + delta,) + output[1:]
                else: return h + delta
            return hook_fn

        for layer_idx in target_layers:
            block = model.model.layers[layer_idx]
            targets = {f"l{layer_idx}_q": block.self_attn.q_proj, f"l{layer_idx}_v": block.self_attn.v_proj}
            for name, module in targets.items():
                pq_layer = PQLoRALayer(module.in_features, module.out_features, r=self.r).to(device=DEVICE, dtype=ADAPTER_DTYPE)
                if name in self.local_state:
                    with torch.no_grad():
                        pq_layer.P_local.copy_(self.local_state[name]["P"])
                        pq_layer.Q_local.copy_(self.local_state[name]["Q"])
                        pq_layer.beta.copy_(self.local_state[name]["beta"])
                if global_weights_dict and name in global_weights_dict:
                    pq_layer.update_global_weights(global_weights_dict[name]["P"].to(DEVICE), global_weights_dict[name]["Q"].to(DEVICE))
                self.active_adapters[name] = pq_layer
                self.hooks.append(module.register_forward_hook(get_hook(name)))

        optimizer = optim.AdamW(self.active_adapters.parameters(), lr=1e-4)
        inputs = tokenizer(raw_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        
        model.train()
        total_loss = 0
        
        # Training Loop (5 Steps to ensure convergence on these 4 examples)
        for step in range(5):
            optimizer.zero_grad()
            outputs = model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        new_state = {}
        for name, adapter in self.active_adapters.items():
            new_state[name] = {"P": adapter.P_local.detach().cpu(), "Q": adapter.Q_local.detach().cpu(), "beta": adapter.beta.detach().cpu()}
        
        for h in self.hooks: h.remove()
        del model, optimizer, inputs, loss
        return total_loss / 5, new_state

    def _phase_2_representation(self, raw_texts):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SHARED_Ws)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_SHARED_Ws, quantization_config=self._get_bnb_config(), device_map=DEVICE)
        model.eval()
        inputs = tokenizer(raw_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=inputs.input_ids, output_hidden_states=True)
            vector = outputs.hidden_states[-1].mean(dim=(0, 1))
        return vector.cpu().to(torch.float32)

    def _phase_2_gradient(self, raw_texts):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SHARED_Ws)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_SHARED_Ws, quantization_config=self._get_bnb_config(), device_map=DEVICE)
        model.eval()
        inputs = tokenizer(raw_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        model.zero_grad()
        loss = model(input_ids=inputs.input_ids, labels=inputs.input_ids).loss
        loss.backward()
        vector = model.lm_head.weight.grad.detach().cpu().flatten()[::20]
        del model, tokenizer, inputs, loss
        return vector.to(torch.float32)

class ServerBlockWise:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.tmp_dir = "server_block_tmp"
        os.makedirs(self.tmp_dir, exist_ok=True)
        
    def aggregate(self, client_updates):
        print(f"Server (BlockWise): Aggregating...")
        for i, up in enumerate(client_updates):
            torch.save(up, f"{self.tmp_dir}/update_{i}.pt")
        del client_updates
        gc.collect()
        
        vectors = []
        for i in range(self.num_clients):
            data = torch.load(f"{self.tmp_dir}/update_{i}.pt", map_location="cpu")
            vectors.append(data["vector"])
            del data
            
        vec_stack = torch.stack(vectors)
        vec_norm = F.normalize(vec_stack, p=2, dim=1)
        sim_matrix = torch.mm(vec_norm, vec_norm.t())
        weights = F.softmax(sim_matrix / 0.5, dim=1)
        del vec_stack, vec_norm, sim_matrix
        
        personalized_globals = []
        meta = torch.load(f"{self.tmp_dir}/update_0.pt", map_location="cpu")
        layer_names = list(meta["weights"].keys())
        del meta
        
        for i in range(self.num_clients):
            client_weights = weights[i]
            client_global_dict = {}
            for layer in layer_names:
                client_global_dict[layer] = {"P": 0, "Q": 0}
            
            for j in range(self.num_clients):
                w = client_weights[j].item()
                if w > 1e-4:
                    data = torch.load(f"{self.tmp_dir}/update_{j}.pt", map_location="cpu")
                    w_dict = data["weights"]
                    for layer in layer_names:
                        if layer in w_dict:
                            if isinstance(client_global_dict[layer]["P"], int):
                                client_global_dict[layer]["P"] = w_dict[layer]["P"] * w
                                client_global_dict[layer]["Q"] = w_dict[layer]["Q"] * w
                            else:
                                client_global_dict[layer]["P"] += w_dict[layer]["P"] * w
                                client_global_dict[layer]["Q"] += w_dict[layer]["Q"] * w
                    del data
            personalized_globals.append(client_global_dict)
            gc.collect()
            
        return personalized_globals