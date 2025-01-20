## Add concept loss functions here 
## Modify the train step
## Load the real data and concepts 
## Run the pipeline 
## Debug 


# train.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from model_3 import TextCTC



class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.texts = dataframe['text'].tolist()
        self.labels = torch.tensor(dataframe['label'].tolist())
        
        # Extract concept columns and onehot encode them
        concept_columns = [col for col in dataframe.columns if col.startswith('c_')]
        self.concepts = dataframe[concept_columns].values
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.concepts_onehot = self.onehot_encoder.fit_transform(self.concepts)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        concept_onehot = torch.tensor(self.concepts_onehot[idx])
        return text, concept_onehot, label

class ConceptTrainer:
    def __init__(self, model, learning_rate=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam([
            {'params': self.model.concept_transformer.parameters(), 'lr': learning_rate},
            {'params': [self.model.unsup_concepts], 'lr': learning_rate * 0.1}
        ])
    
    def train_step(self, texts, concepts, labels):
        self.model.train()
        self.optimizer.zero_grad()
        
        labels = labels.to(self.device)
        logits, concept_attn = self.model(texts)
        
        # Classification loss
        ce_loss = F.cross_entropy(logits, labels)
        
        # Concept diversity loss
        diversity_loss = -torch.mean(torch.std(concept_attn, dim=0))
        
        # Total loss with weighted components
        total_loss = ce_loss + 0.1 * diversity_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()

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
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

def main():
    # Example dataframe
    data = {
        'text': [
            "This product exceeded my expectations.",
            "The quality is terrible and it broke easily.",
            "Average product, nothing special about it.",
            "Great value for money, highly recommend.",
            "Don't waste your money on this.",
            "Absolutely love this product!",
            "Poor customer service and product quality.",
            "Decent product for the price.",
            "Outstanding performance and reliability.",
            "Completely disappointed with this purchase."
        ],
        'label': [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],  # 1 for positive, 0 for negative
        'concept_1': [1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
        'concept_2': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        'concept_3': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    }
    dataframe = pd.DataFrame(data)
    
    # Create datasets
    train_dataset = TextDataset(dataframe)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    
    # Initialize model and trainer
    model = TextCTC(
        model_name='t5-base',
        num_classes=2,
        n_unsup_concepts=5,
        embedding_dim=768,
        num_heads=2
    )
    
    trainer = ConceptTrainer(model, learning_rate=1e-4)
    
    # Training loop
    num_epochs = 10
    print("Starting training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_texts, batch_concepts, batch_labels in train_dataloader:
            loss = trainer.train_step(batch_texts, batch_concepts, batch_labels)
            epoch_loss += loss
            
        avg_loss = epoch_loss / len(train_dataloader)
        
        if (epoch + 1) % 2 == 0:  # Print every 2 epochs
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    print("\nTraining completed.")
    
    # Final evaluation
    print("\nAnalyzing concepts for example texts...")
    test_texts = ["This product is amazing!", "This product is terrible."]
    
    model.eval()
    with torch.no_grad():
        for text in test_texts:
            _, concept_attn = model([text])
            values, indices = torch.topk(concept_attn, k=2, dim=1)
            
            print(f"\nText: {text}")
            concepts = [f"C{idx.item()}:{val.item():.3f}" for val, idx in zip(values[0], indices[0])]
            print(f"Top concepts: {', '.join(concepts)}")

if __name__ == "__main__":
    main()



