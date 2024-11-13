# train.py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np
from model_supervised import TextCTC  # Make sure model.py is in the same directory
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class TextConceptDataset(Dataset):
    def __init__(self, texts, concept_list, labels):
        assert len(texts) == len(concept_list) == len(labels), "All inputs must have the same length"
        self.texts = texts
        self.concept_list = concept_list
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.concept_list[idx], self.labels[idx]

class ConceptTrainer:
    def __init__(self, model, learning_rate=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate
        )
    
    def train_step(self, texts, concepts, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        labels = labels.to(self.device)
        logits, concept_attn = self.model(texts, concepts)
        
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, concepts, labels in dataloader:
                labels = labels.to(self.device)
                logits, _ = self.model(texts, concepts)
                loss = F.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(dataloader), correct / total

def main():
    # Set random seed for reproducibility
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Example dataset
    texts = [
        "This product exceeded my expectations",
        "The quality is terrible and it broke easily",
        "Average product, nothing special",
        "Great value for money, highly recommend",
        "Don't waste your money on this",
        "Absolutely love this product",
        "Poor customer service and product quality",
        "Decent product for the price"
    ]
    
    # Concept vocabulary groups
    concept_groups = {
        'quality': ['quality', 'excellence', 'standard'],
        'price': ['price', 'cost', 'value', 'expensive', 'cheap'],
        'durability': ['durability', 'lasting', 'robust'],
        'service': ['service', 'support', 'assistance'],
        'performance': ['performance', 'efficiency', 'effectiveness']
    }
    
    # Ensure concept_list has the same length as texts
    concept_list = [list(concept_groups.keys()) for _ in range(len(texts))]
    
    # Labels: 1 for positive, 0 for negative
    labels = [1, 0, 0, 1, 0, 1, 0, 1]
    
    # Create dataset and dataloader
    dataset = TextConceptDataset(texts, concept_list, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    try:
        model = TextCTC(
            model_name='t5-base',
            embedding_dim=768,
            num_classes=2,
            num_heads=2
        )
        
        trainer = ConceptTrainer(model, device=device)
        
        # Training loop
        num_epochs = 10
        logging.info("Starting training...")
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch_texts, batch_concepts, batch_labels in tqdm(dataloader, desc=f'Epoch {epoch+1}'):
                loss = trainer.train_step(batch_texts, batch_concepts, batch_labels)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(dataloader)
            if (epoch + 1) % 2 == 0:
                # Evaluate model
                val_loss, val_acc = trainer.evaluate(dataloader)
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        logging.info("Training completed.")
        
        # Test concept interpretation
        logging.info("Analyzing concepts for example texts...")
        test_texts = ["This product is amazing!", "This product is terrible."]
        test_concepts = [list(concept_groups.keys())]  # Use the same concepts for testing
        
        model.eval()
        with torch.no_grad():
            for text in test_texts:
                concept_attn = model.interpret_concepts([text], test_concepts)
                values, indices = torch.topk(concept_attn, k=3, dim=1)
                
                print(f"\nText: {text}")
                concepts = [f"{list(concept_groups.keys())[idx.item()]}:{val.item():.3f}" 
                           for val, idx in zip(values[0], indices[0])]
                print(f"Top concepts: {', '.join(concepts)}")
                
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()