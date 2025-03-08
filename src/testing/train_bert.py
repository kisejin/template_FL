import torch
import pandas as pd
from transformers import BertForSequenceClassification, AutoTokenizer
import ast
from accelerate import Accelerator
import os
from sklearn.model_selection import train_test_split
import numpy as np

def train_bert_model(csv_path, learning_rate=5e-5, epochs=20, batch_size=8, test_size=0.2):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device_map = torch.device(f"cuda:{torch.cuda.current_device()}" if use_cuda else "cpu")
    
    # Load holdout data from CSV
    df = pd.read_csv(csv_path)
    
    # Process the data - column names are "Decoded Texts" and "Perplexity"
    # The "Decoded Texts" column contains string representations of Python lists
    texts = []
    for text_list_str in df['Decoded Texts']:
        try:
            # Parse string representation of list into actual list
            text_list = ast.literal_eval(text_list_str)
            # Extract the first (and likely only) element
            if isinstance(text_list, list) and len(text_list) > 0:
                texts.append(text_list[0])
            else:
                texts.append("")  # Empty string for invalid entries
        except (SyntaxError, ValueError):
            # Handle parsing errors
            texts.append("")
    
    perplexities = df['Perplexity'].tolist()
    
    # Split into train and test sets
    train_texts, test_texts, train_perplexities, test_perplexities = train_test_split(
        texts, perplexities, test_size=test_size, random_state=42)
    
    # Create pairs of processed text and perplexity
    train_pairs = list(zip(train_texts, train_perplexities))
    test_pairs = list(zip(test_texts, test_perplexities))
    
    print(f"Loaded {len(train_pairs)} training pairs and {len(test_pairs)} testing pairs from {csv_path}")
    
    # Load model with num_labels=1 for regression
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=1,  # Set number of labels to 1 for regression task
    )
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
        
    # Move model to device BEFORE setting up optimizer
    model.to(device_map)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Variables to track best model
    best_val_loss = float('inf')
    best_model = None
    
    def evaluate_model(model, eval_pairs):
        model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(0, len(eval_pairs), batch_size):
                batch = eval_pairs[i:i+batch_size]
                texts_batch, scores_batch = zip(*batch)
                
                # Tokenize the texts
                bert_inputs = tokenizer(
                    list(texts_batch),
                    truncation=True,
                    padding='max_length',
                    max_length=512,
                    return_tensors='pt'
                )
                
                # Move inputs to device
                bert_inputs = {k: v.to(device_map) for k, v in bert_inputs.items()}
                
                # Convert scores to tensor
                scores_tensor = torch.tensor(scores_batch, device=device_map, dtype=torch.float32).unsqueeze(1)
                
                # Forward pass
                outputs = model(**bert_inputs, labels=scores_tensor)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                # Collect predictions and labels for metrics
                preds = outputs.logits.squeeze().cpu().numpy()
                labels = scores_tensor.squeeze().cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        # Calculate metrics
        mse = np.mean((np.array(all_preds) - np.array(all_labels)) ** 2)
        mae = np.mean(np.abs(np.array(all_preds) - np.array(all_labels)))
        
        avg_loss = total_loss / ((len(eval_pairs) - 1) // batch_size + 1)
        return avg_loss, mse, mae
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # Process in batches
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            texts_batch, scores_batch = zip(*batch)
            
            # Tokenize the texts
            bert_inputs = tokenizer(
                list(texts_batch),
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            )
            
            # Move inputs to device
            bert_inputs = {k: v.to(device_map) for k, v in bert_inputs.items()}
            
            # Convert scores to tensor
            scores_tensor = torch.tensor(scores_batch, device=device_map, dtype=torch.float32).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**bert_inputs, labels=scores_tensor)
            loss = outputs.loss
            
            # Standard backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Batch {i//batch_size}: Loss = {loss.item():.4f}")
        
        avg_train_loss = total_loss / ((len(train_pairs) - 1) // batch_size + 1)
        
        # Evaluate on test set
        val_loss, val_mse, val_mae = evaluate_model(model, test_pairs)
        
        print(f"[Epoch {epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            print(f"New best model found at epoch {epoch+1} with validation loss: {val_loss:.4f}")
    
    # Load best model for final save
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Save the model
    output_dir = os.path.join("models", "bert_perplexity_predictor")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Best model saved to {output_dir} with validation loss: {best_val_loss:.4f}")
    
    return model, tokenizer

if __name__ == "__main__":
    # Path to the CSV file with holdout perplexity pairs
    csv_path = "holdout_perplexity_pairs2.csv"
    train_bert_model(csv_path)