import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel

class CrossAttention(nn.Module):
    def __init__(self, dim, n_outputs, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, n_outputs)
        
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(projection_dropout)

    def forward(self, x, concepts):
        B, N, C = x.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(concepts).reshape(1, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(concepts).reshape(1, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        x = self.proj_dropout(x)
        
        # Return attention scores averaged over heads
        return x, attn.mean(dim=1)  # This will give us [B, N, num_concepts]

class TextCTC(nn.Module):
    def __init__(
        self,
        model_name='t5-base',
        embedding_dim=768,
        max_length=512,
        n_unsup_concepts=10,
        num_classes=2,
        num_heads=2,
        attention_dropout=0.1,
        projection_dropout=0.1,
    ):
        super().__init__()
        
        self.n_unsup_concepts = n_unsup_concepts
        
        # T5 components
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.t5_model = T5EncoderModel.from_pretrained(model_name)
        
        # Freeze T5
        for param in self.t5_model.parameters():
            param.requires_grad = False
            
        # Project T5 embeddings if needed
        self.proj = nn.Linear(self.t5_model.config.hidden_size, embedding_dim) if embedding_dim != self.t5_model.config.hidden_size else nn.Identity()
        
        # Concept learning components
        self.unsup_concepts = nn.Parameter(torch.zeros(1, n_unsup_concepts, embedding_dim))
        nn.init.normal_(self.unsup_concepts, std=0.02)
        
        self.concept_transformer = CrossAttention(
            dim=embedding_dim,
            n_outputs=num_classes,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        
        self.token_attention_pool = nn.Linear(embedding_dim, 1)
        
    def forward(self, texts):
        # Move inputs to the same device as the model
        device = next(self.parameters()).device
        
        # Tokenize and get embeddings
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.t5_model(**encodings)
            x = outputs.last_hidden_state
            
        x = self.proj(x)
        
        # Pool tokens using attention
        token_attn = F.softmax(self.token_attention_pool(x), dim=1).transpose(-1, -2)
        x_pooled = torch.matmul(token_attn, x)
        
        # Get concepts and logits
        logits, concept_attn = self.concept_transformer(x_pooled, self.unsup_concepts)
        
        # Ensure concept_attn has the right shape [batch_size, num_concepts]
        if len(concept_attn.shape) == 3:
            concept_attn = concept_attn.mean(1)  # Average over sequence length if needed
            
        return logits.squeeze(1), concept_attn

    def interpret_concepts(self, texts, top_k=None):
        self.eval()
        with torch.no_grad():
            _, concept_attn = self(texts)
            
            # Ensure concept_attn is 2D [batch_size, num_concepts]
            if len(concept_attn.shape) == 3:
                concept_attn = concept_attn.mean(1)
            
            # Print debugging info
            print(f"Concept attention shape: {concept_attn.shape}")
            print(f"Concept attention values:\n{concept_attn}")
            
            # Ensure valid top_k
            if top_k is None or top_k > self.n_unsup_concepts:
                top_k = min(3, self.n_unsup_concepts)
            
            try:
                values, indices = torch.topk(concept_attn, k=top_k, dim=-1)
                return values, indices
            except RuntimeError as e:
                print(f"Error in topk: shape={concept_attn.shape}, top_k={top_k}")
                raise e