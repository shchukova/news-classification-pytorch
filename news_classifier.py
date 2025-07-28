# Core libraries for data handling and deep learning
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Hugging Face libraries for datasets, tokenizers, and models
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_scheduler

# Other utilities
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # Using tqdm.auto for better compatibility in various environments
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from IPython.display import Markdown as md # For displaying Markdown in environments like Jupyter

# Suppress warnings (optional, but can clean up output)
import warnings
warnings.filterwarnings('ignore')

# --- Plotting Function ---
def plot_metrics(losses, accuracies, title="Training Progress"):
    """
    Plots the training loss and validation accuracy over epochs.

    Args:
        losses (list): List of training loss values per epoch.
        accuracies (list): List of validation accuracy values per epoch.
        title (str): Title for the plot.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Loss
    color = 'tab:red'
    ax1.plot(losses, color=color, label='Training Loss')
    ax1.set_xlabel('Epoch', color=color)
    ax1.set_ylabel('Total Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_title(title)
    ax1.legend(loc='upper left')

    # Plot Accuracy on a second Y-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(accuracies, color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.show()

# --- Data Loading and Preprocessing ---

# Load the AG_NEWS dataset from Hugging Face Datasets
print("Loading AG_NEWS dataset...")
ag_news_dataset = load_dataset("ag_news")
print("Dataset loaded successfully.")
print(ag_news_dataset)

# Define a mapping for AG_NEWS labels (0-indexed for Hugging Face models)
# AG_NEWS original labels are 1-4, Hugging Face models expect 0-indexed labels.
# So, label 1 (World) becomes 0, 2 (Sports) becomes 1, etc.
ag_news_label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tec"}
num_class = len(ag_news_label_map)
print(f"Number of classes: {num_class}")

# Load a pre-trained tokenizer (e.g., DistilBERT's tokenizer)
# This tokenizer will handle tokenization, numericalization, and adding special tokens.
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded successfully.")

def tokenize_function(examples):
    """
    Tokenizes the text in the dataset examples.
    """
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# Apply tokenization to the entire dataset
print("Tokenizing dataset...")
tokenized_ag_news = ag_news_dataset.map(tokenize_function, batched=True)
print("Dataset tokenization complete.")

# Remove original text and unnecessary columns, rename label to labels for model compatibility
tokenized_ag_news = tokenized_ag_news.remove_columns(["text"])
tokenized_ag_news = tokenized_ag_news.rename_column("label", "labels")
tokenized_ag_news.set_format("torch") # Set format to PyTorch tensors

# Split the training dataset into training and validation sets
# Hugging Face `datasets` automatically provides 'train' and 'test' splits.
# We'll split the 'train' split further for validation.
train_dataset_hf = tokenized_ag_news["train"]
test_dataset_hf = tokenized_ag_news["test"]

# Randomly split the training dataset into training and validation datasets (95% train, 5% validation)
# Note: Hugging Face datasets have a built-in train_test_split method
train_validation_split = train_dataset_hf.train_test_split(test_size=0.05, seed=42)
split_train_ = train_validation_split["train"]
split_valid_ = train_validation_split["test"]

print(f"Training set size: {len(split_train_)}")
print(f"Validation set size: {len(split_valid_)}")
print(f"Test set size: {len(test_dataset_hf)}")

# Data collator for dynamic padding
# This will pad sequences to the longest sequence in each batch, which is more efficient.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Create DataLoaders
BATCH_SIZE = 64
train_dataloader = DataLoader(
    split_train_, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)
valid_dataloader = DataLoader(
    split_valid_, shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator
)
test_dataloader = DataLoader(
    test_dataset_hf, shuffle=False, batch_size=BATCH_SIZE, collate_fn=data_collator
)

# --- Model Definition and Training ---

# Set the device for training (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained model for sequence classification
# The model automatically adds a classification head for `num_class` labels.
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_class).to(device)
print("Model loaded successfully.")

# Define optimizer and learning rate scheduler
LR = 5e-5 # Common learning rate for fine-tuning transformers
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

EPOCHS = 3 # Reduced epochs for faster demonstration, increase for better performance
num_training_steps = EPOCHS * len(train_dataloader)
scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Lists to store metrics for plotting
cum_loss_list = []
acc_epoch = []
best_acc = 0.0

print("Starting training...")
# Training loop
for epoch in range(1, EPOCHS + 1):
    model.train() # Set model to training mode
    total_loss = 0
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_dataloader)
    cum_loss_list.append(avg_train_loss)

    # Evaluate on validation set
    model.eval() # Set model to evaluation mode
    total_correct = 0
    total_samples = 0
    validation_progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch} Validation")

    with torch.no_grad():
        for batch in validation_progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            total_correct += (predictions == batch["labels"]).sum().item()
            total_samples += batch["labels"].size(0)

    accu_val = total_correct / total_samples
    acc_epoch.append(accu_val)

    print(f"Epoch {epoch}: Avg Training Loss = {avg_train_loss:.4f}, Validation Accuracy = {accu_val:.4f}")

    # Save the model if current validation accuracy is better than the previous best
    if accu_val > best_acc:
        best_acc = accu_val
        # Save the entire model (including tokenizer config, etc.)
        model.save_pretrained('./best_model') 
        tokenizer.save_pretrained('./best_model')
        print(f"Saved new best model with validation accuracy: {best_acc:.4f}")

# Plotting the training progress
plot_metrics(cum_loss_list, acc_epoch, title="Model Training Progress (Hugging Face)")

# --- Final Evaluation on Test Set ---
print("\nEvaluating on test set...")
model.eval() # Ensure model is in evaluation mode
total_correct = 0
total_samples = 0
test_progress_bar = tqdm(test_dataloader, desc="Test Set Evaluation")

with torch.no_grad():
    for batch in test_progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        total_correct += (predictions == batch["labels"]).sum().item()
        total_samples += batch["labels"].size(0)

test_accuracy = total_correct / total_samples
print(f"Test Accuracy after training: {test_accuracy:.4f}")

# --- t-SNE Visualization of Embeddings ---
# For visualization, we'll extract embeddings from the pre-trained model's last hidden state.
# Typically, the [CLS] token's embedding is used as the sequence representation.

print("\nGenerating embeddings for t-SNE visualization...")
model.eval()
embeddings_list = []
labels_list = []

# Use a smaller subset of the validation data for faster t-SNE if the dataset is large
# Here, we'll just take the first batch from the validation dataloader
batch_for_tsne = next(iter(valid_dataloader))
batch_for_tsne = {k: v.to(device) for k, v in batch_for_tsne.items()}

with torch.no_grad():
    outputs = model(**batch_for_tsne, output_hidden_states=True)
    # The last hidden state is usually outputs.hidden_states[-1]
    # For sequence classification, often the embedding of the [CLS] token (first token) is used.
    # DistilBERT outputs hidden states for all layers, outputs.hidden_states[0] is the embedding layer output
    # outputs.hidden_states[-1] is the last layer's hidden state.
    # We take the [CLS] token's embedding (index 0) for each sequence in the batch.
    
    # Ensure hidden_states exist before accessing
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        # Get the last hidden state for all tokens in the batch
        last_hidden_state = outputs.hidden_states[-1]
        # Extract the [CLS] token embedding (first token of each sequence)
        cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
        embeddings_list.append(cls_embeddings)
        labels_list.append(batch_for_tsne["labels"].cpu().numpy())
    else:
        print("Warning: 'hidden_states' not found in model output. Cannot perform t-SNE on embeddings.")

if embeddings_list:
    embeddings_numpy = np.concatenate(embeddings_list, axis=0)
    labels_numpy = np.concatenate(labels_list, axis=0)

    # Perform t-SNE
    X_embedded_3d = TSNE(n_components=3, random_state=42).fit_transform(embeddings_numpy)

    # Create a 3D scatter plot using Plotly
    trace = go.Scatter3d(
        x=X_embedded_3d[:, 0],
        y=X_embedded_3d[:, 1],
        z=X_embedded_3d[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=labels_numpy,  # Use label information for color
            colorscale='Viridis',  # Choose a colorscale
            opacity=0.8
        )
    )

    layout = go.Layout(title="3D t-SNE Visualization of Embeddings (Hugging Face)",
                       scene=dict(xaxis_title='Dimension 1',
                                  yaxis_title='Dimension 2',
                                  zaxis_title='Dimension 3'))

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()
else:
    print("t-SNE visualization skipped due to missing embeddings.")

# --- Prediction Example ---
article = """Canada navigated a stiff test against the Republic of Ireland on a rain soaked evening in Perth, coming from behind to claim a vital 2-1 victory at the Women’s World Cup.
Katie McCabe opened the scoring with an incredible Olimpico goal – scoring straight from a corner kick – as her corner flew straight over the desp"""

def predict_category(text_article, model, tokenizer, label_map, device):
    """
    Predicts the category of a given news article.

    Args:
        text_article (str): The news article text.
        model (torch.nn.Module): The trained Hugging Face model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        label_map (dict): Dictionary mapping label indices to category names.
        device (torch.device): The device (cpu/cuda) to run inference on.

    Returns:
        str: The predicted category.
        dict: Probabilities for each category.
    """
    model.eval() # Set model to evaluation mode
    
    # Tokenize the input text
    inputs = tokenizer(text_article, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    
    # Move inputs to the correct device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)[0] # Get probabilities for the single article
        predicted_class_id = torch.argmax(probabilities).item()

    predicted_category = label_map[predicted_class_id]
    
    # Optional: Get probabilities for all classes
    prob_dict = {label_map[i]: prob.item() for i, prob in enumerate(probabilities)}

    return predicted_category, prob_dict

# --- Example Usage of Prediction ---
print("\n--- Prediction Example ---")
# Make sure to load the best model if you want to use it
# You would typically load it like this:
model = AutoModelForSequenceClassification.from_pretrained('./best_model').to(device)
tokenizer = AutoTokenizer.from_pretrained('./best_model')

# For this example, we'll just use the model that was just trained in the current session
predicted_category, probabilities = predict_category(article, model, tokenizer, ag_news_label_map, device)

print(f"Article:\n{article}\n")
print(f"Predicted Category: {predicted_category}")
print(f"Probabilities: {probabilities}")