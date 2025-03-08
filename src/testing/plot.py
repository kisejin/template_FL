import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import ast
from sklearn.metrics import mean_squared_error, mean_absolute_error
import re
import os

def extract_metrics_from_log(log_file):
    """Extract training metrics from log file."""
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    # Extract metrics using regex
    epochs = []
    train_losses = []
    val_losses = []
    val_mses = []
    val_maes = []
    
    # Pattern to match the end-of-epoch metrics
    pattern = r"\[Epoch (\d+)/\d+\] Train Loss: ([\d\.]+), Val Loss: ([\d\.]+), Val MSE: ([\d\.]+), Val MAE: ([\d\.]+)"
    matches = re.findall(pattern, log_content)
    
    for match in matches:
        epoch, train_loss, val_loss, val_mse, val_mae = match
        epochs.append(int(epoch))
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        val_mses.append(float(val_mse))
        val_maes.append(float(val_mae))
    
    return epochs, train_losses, val_losses, val_mses, val_maes

def plot_training_curves(epochs, train_losses, val_losses, val_mses, val_maes):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot validation MSE
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_mses, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Validation MSE')
    plt.grid(True)
    
    # Plot validation MAE
    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_maes, 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Validation MAE')
    plt.grid(True)
    
    # Plot all metrics together (normalized)
    plt.subplot(2, 2, 4)
    plt.plot(epochs, np.array(train_losses) / max(train_losses), 'b-', label='Training Loss (normalized)')
    plt.plot(epochs, np.array(val_losses) / max(val_losses), 'r-', label='Validation Loss (normalized)')
    plt.plot(epochs, np.array(val_mses) / max(val_mses), 'g-', label='Validation MSE (normalized)')
    plt.plot(epochs, np.array(val_maes) / max(val_maes), 'm-', label='Validation MAE (normalized)')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Value')
    plt.title('All Metrics (Normalized)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

def evaluate_and_visualize_predictions(model_path, csv_path, batch_size=8):
    """Evaluate model on test data and visualize predictions vs actual values."""
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load and process test data
    df = pd.read_csv(csv_path)
    texts = []
    for text_list_str in df['Decoded Texts']:
        try:
            text_list = ast.literal_eval(text_list_str)
            if isinstance(text_list, list) and len(text_list) > 0:
                texts.append(text_list[0])
            else:
                texts.append("")
        except:
            texts.append("")
    
    perplexities = df['Perplexity'].tolist()
    
    # Use a small subset for visualization (to avoid overcrowding)
    # You can adjust the number based on your dataset size
    sample_size = min(200, len(texts))
    indices = np.random.choice(len(texts), sample_size, replace=False)
    
    sample_texts = [texts[i] for i in indices]
    sample_perplexities = [perplexities[i] for i in indices]
    
    # Get predictions
    predictions = []
    with torch.no_grad():
        for i in range(0, len(sample_texts), batch_size):
            batch_texts = sample_texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**inputs)
            batch_preds = outputs.logits.squeeze().cpu().numpy()
            if batch_size == 1:
                predictions.append(batch_preds)
            else:
                predictions.extend(batch_preds)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(sample_perplexities, predictions, alpha=0.5)
    
    # Plot the ideal y=x line
    min_val = min(min(sample_perplexities), min(predictions))
    max_val = max(max(sample_perplexities), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Perplexity')
    plt.ylabel('Predicted Perplexity')
    plt.title('Predicted vs Actual Perplexity')
    plt.grid(True)
    
    # Calculate and display metrics
    mse = mean_squared_error(sample_perplexities, predictions)
    mae = mean_absolute_error(sample_perplexities, predictions)
    plt.annotate(f'MSE: {mse:.4f}\nMAE: {mae:.4f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('predictions_scatter.png')
    plt.show()
    
    # Create error histogram
    errors = np.array(predictions) - np.array(sample_perplexities)
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True)
    plt.savefig('error_distribution.png')
    plt.show()

def main():
    # Save the log output to a file
    log_file = "bert_training_log.txt"
    
    # If you saved the log content to a file, use that directly
    # Otherwise, you can save the text content to a file first
    try:
        # Check if log file exists, if not create it with the logs
        if not os.path.exists(log_file):
            with open(log_file, 'w') as f:
                # Paste the log content from your terminal output here if needed
                pass
                
        epochs, train_losses, val_losses, val_mses, val_maes = extract_metrics_from_log(log_file)
        
        if epochs:
            plot_training_curves(epochs, train_losses, val_losses, val_mses, val_maes)
        else:
            print("No metrics found in log file.")
    except Exception as e:
        print(f"Error extracting metrics: {e}")
    
    # Evaluate model on test data and create prediction visualization
    model_path = os.path.join("models", "bert_perplexity_predictor")
    csv_path = "holdout_perplexity_pairs2.csv"
    
    if os.path.exists(model_path):
        try:
            evaluate_and_visualize_predictions(model_path, csv_path)
        except Exception as e:
            print(f"Error visualizing predictions: {e}")
    else:
        print(f"Model not found at {model_path}")

if __name__ == "__main__":
    main()