# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel

class CrossAttention(nn.Module):
    def __init__(self, dim, n_outputs, num_heads=8, attention_dropout=0.1, projection_dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dim = dim
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, n_outputs)
        
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(projection_dropout)

    def forward(self, x, concepts):
        B = x.shape[0]  # batch size
        
        # Project queries, keys, and values
        q = self.q(x)          # [B, N, dim]
        k = self.k(concepts)   # [B, M, dim]
        v = self.v(concepts)   # [B, M, dim]
        
        # # Debug prints
        # print(f"q shape: {q.shape}")
        # print(f"k shape: {k.shape}")
        # print(f"v shape: {v.shape}")
        
        # Reshape to include heads
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, head_dim]
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, M, head_dim]

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, M]
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v)  # [B, heads, N, head_dim]
        x = x.transpose(1, 2)  # [B, N, heads, head_dim]
        x = x.reshape(B, -1, self.dim)  # [B, N, dim]
        
        x = self.out(x)
        x = self.proj_dropout(x)
        
        return x, attn.mean(dim=1)  # Average attention across heads

class TextCTC(nn.Module):
    def __init__(
        self,
        model_name='t5-base',
        embedding_dim=768,
        num_classes=2,
        num_heads=2,
        attention_dropout=0.1,
        projection_dropout=0.1,
    ):
        super().__init__()
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.t5_model = T5EncoderModel.from_pretrained(model_name)
        
        # Freeze T5
        for param in self.t5_model.parameters():
            param.requires_grad = False
            
        self.proj = nn.Linear(self.t5_model.config.hidden_size, embedding_dim)
        
        self.concept_attention = CrossAttention(
            dim=embedding_dim,
            n_outputs=num_classes,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=projection_dropout,
        )
        
        self.token_attention_pool = nn.Linear(embedding_dim, 1)

    def encode_text(self, texts):
        device = next(self.parameters()).device
        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.t5_model(**encodings)
        return self.proj(outputs.last_hidden_state)

    def encode_concepts(self, concept_list):
        device = next(self.parameters()).device
        # Join concepts with spaces for each example in the batch
        concept_texts = [" ".join(concepts) for concepts in concept_list]
        encodings = self.tokenizer(concept_texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.t5_model(**encodings)
        return self.proj(outputs.last_hidden_state)
        
    def forward(self, texts, concept_list):
        # Encode text and concepts
        text_embeds = self.encode_text(texts)      # [B, N, dim]
        concept_embeds = self.encode_concepts(concept_list)  # [B, M, dim]

        # Debug prints
        print(f"text_embeds shape: {text_embeds.shape}")
        print(f"concept_embeds shape: {concept_embeds.shape}")

        # Pool text tokens using attention
        text_attn = F.softmax(self.token_attention_pool(text_embeds), dim=1).transpose(-1, -2)
        text_pooled = torch.matmul(text_attn, text_embeds)

        # Cross attention between text and concepts
        logits, concept_attn = self.concept_attention(text_pooled, concept_embeds)

        return logits.squeeze(1), concept_attn

    def interpret_concepts(self, texts, concept_list):
        self.eval()
        with torch.no_grad():
            _, concept_attn = self(texts, concept_list)
            return concept_attn