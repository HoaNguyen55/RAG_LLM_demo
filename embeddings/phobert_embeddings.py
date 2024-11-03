import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Union
from tqdm import tqdm
from config.config import Config

class PhoBERTEmbedding:
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_config["model_name"])
        self.model = AutoModel.from_pretrained(config.model_config["model_name"])
        
        self.device = torch.device(config.model_config["device"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"PhoBERT model loaded successfully on {self.device}")

    def get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        batch_size = self.config.model_config["batch_size"]
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.model_config["max_length"],
                return_tensors='pt'
            )
            
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.extend(embeddings)
        
        embeddings_array = np.array(all_embeddings)
        return embeddings_array[0] if len(texts) == 1 else embeddings_array