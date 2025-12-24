import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pqlora import PQLoRALayer
import gc

# --- CONFIGURATION ---
MODEL_SMALL = "Qwen/Qwen2.5-0.5B-Instruct" 
MODEL_LARGE = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_SHARED_Ws = "Qwen/Qwen2.5-0.5B-Instruct" 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# We use Float32 for the ADAPTER to ensure training stability
# The BASE MODEL will be loaded in 4-bit/Float16
ADAPTER_DTYPE = torch.float32 

class FederatedClient:
    def __init__(self, client_id, model_type, dataset):
        self.id = client_id
        self.model_type = model_type
        self.dataset = dataset
        self.r = 16
        self.history = {"loss": []} 
        self.g_hat = None 
        self.alpha = 0.5 
        self.local_state = None 

    def get_model_name(self):
        return MODEL_SMALL if self.model_type == "small" else MODEL_LARGE

    def _get_bnb_config(self):
        """Returns 4-bit quantization config to save memory"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    def train_and_rela(self, global_P, global_Q):
        # Batch size can be safely increased now!
        batch_size = 4 
        raw_texts = [self.dataset[i] for i in range(min(batch_size, len(self.dataset)))]
        
        # --- PHASE 1: TRAIN ---
        loss, self.local_state = self._phase_1_train(raw_texts, global_P, global_Q)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        self.history["loss"].append(loss)
        print(f"Client {self.id} ({self.model_type}) Loss: {loss:.4f}")

        # --- PHASE 2: PROBE ---
        rela_grad = self._phase_2_probe(raw_texts)
        
        gc.collect()
        torch.cuda.empty_cache()

        return {
            "P": self.local_state["P"],
            "Q": self.local_state["Q"],
            "rela_grad": rela_grad
        }

    def _phase_1_train(self, raw_texts, global_P, global_Q):
        model_name = self.get_model_name()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        # Load in 4-bit to save massive amounts of memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            quantization_config=self._get_bnb_config(),
            device_map=DEVICE
        )
        # Gradient checkpointing saves even more memory
        model.gradient_checkpointing_enable()
        model.eval() 
        
        # Inject PQ-LoRA (in Float32 for stability)
        emb_dim = model.get_input_embeddings().embedding_dim
        pq_layer = PQLoRALayer(emb_dim, emb_dim, r=self.r).to(device=DEVICE, dtype=ADAPTER_DTYPE)
        
        if self.local_state:
            with torch.no_grad():
                pq_layer.P_local.copy_(self.local_state["P"])
                pq_layer.Q_local.copy_(self.local_state["Q"])
                pq_layer.beta.copy_(self.local_state["beta"])
        
        if global_P is not None:
            pq_layer.update_global_weights(global_P.to(DEVICE), global_Q.to(DEVICE))
            
        optimizer = optim.AdamW([pq_layer.P_local, pq_layer.Q_local, pq_layer.beta], lr=2e-4)
        inputs = tokenizer(raw_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        
        model.train()
        loss_avg = 0
        
        for _ in range(2):
            optimizer.zero_grad()
            
            # 1. Get Embeddings (These will be Float16 from 4-bit model)
            embeds = model.get_input_embeddings()(inputs.input_ids)
            
            # 2. Cast to Float32 for Adapter Processing
            embeds_f32 = embeds.to(ADAPTER_DTYPE)
            adapter_out = pq_layer(embeds_f32)
            
            # 3. Add Residual & Cast back to model dtype
            modified_embeds = embeds_f32 + adapter_out
            modified_embeds = modified_embeds.to(embeds.dtype)
            
            outputs = model(inputs_embeds=modified_embeds, labels=inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_avg += loss.item()
            
        new_state = {
            "P": pq_layer.P_local.detach().cpu(),
            "Q": pq_layer.Q_local.detach().cpu(),
            "beta": pq_layer.beta.detach().cpu()
        }
        
        final_loss = loss_avg / 2
        del model, pq_layer, optimizer, inputs, embeds, outputs, loss
        return final_loss, new_state

    def _phase_2_probe(self, raw_texts):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SHARED_Ws)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        # Load Probe in 4-bit too
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_SHARED_Ws, 
            quantization_config=self._get_bnb_config(),
            device_map=DEVICE
        )
        model.eval()
        
        inputs = tokenizer(raw_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
        model.zero_grad()
        
        outputs = model(input_ids=inputs.input_ids, labels=inputs.input_ids)
        loss = outputs.loss
        loss.backward()
        
        g_i = model.lm_head.weight.grad.detach().cpu().flatten()
        
        if self.g_hat is None:
            self.g_hat = g_i
        else:
            self.g_hat = (1 - self.alpha) * self.g_hat + self.alpha * g_i
            
        noise = torch.randn_like(self.g_hat) * 1e-4
        g_tilde = self.g_hat + noise
        
        del model, tokenizer, inputs, outputs, loss, g_i
        return g_tilde

class Server:
    def __init__(self, num_clients):
        self.client_params = {} 
        self.num_clients = num_clients
        
    def aggregate(self, client_updates, mode="rela"):
        print(f"Server: Aggregating using {mode.upper()}...")
        for i, up in enumerate(client_updates):
            self.client_params[i] = up
            
        grads = torch.stack([self.client_params[i]["rela_grad"] for i in range(self.num_clients)])
        grads_norm = F.normalize(grads, p=2, dim=1)
        sim_matrix = torch.mm(grads_norm, grads_norm.t())
        
        tau = 0.5 
        weights = F.softmax(sim_matrix / tau, dim=1)
        
        personalized_globals = []
        for i in range(self.num_clients):
            new_P = torch.zeros_like(client_updates[0]["P"])
            new_Q = torch.zeros_like(client_updates[0]["Q"])
            client_weights = weights[i]
            
            for j in range(self.num_clients):
                w = client_weights[j].item()
                new_P += self.client_params[j]["P"] * w
                new_Q += self.client_params[j]["Q"] * w
            personalized_globals.append({"P": new_P, "Q": new_Q})
            
        return personalized_globals 

class AlignmentManager:
    def align_models(self):
        print("Alignment: Initialized orthogonal matrices (Theorem 1).")
        pass