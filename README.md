# Text Classification with PyTorch and AG News Dataset

This project demonstrates the end-to-end development of a text classification model using PyTorch.
It leverages the torchtext library for efficient data handling, including tokenization and vocabulary building, and implements a simple yet effective neural network architecture to classify news articles into predefined categories from the AG News dataset. 

The project also includes visualizations for model performance and embedding spaces.

# Features
**Data Preprocessing**: Utilizes torchtext.data.utils.get_tokenizer for basic English tokenization and torchtext.vocab.build_vocab_from_iterator for vocabulary construction, including handling of unknown tokens.

**Custom Data Loading**: Implements a collate_batch function for efficient batching of text data with varying lengths, managing text offsets for nn.EmbeddingBag.

**Text Classification Model**: A simple feed-forward neural network (TextClassificationModel) built with torch.nn.EmbeddingBag for efficient text embedding and a linear layer for classification.

**Training & Evaluation Loop**: Demonstrates a standard PyTorch training loop with CrossEntropyLoss and SGD optimizer, including learning rate scheduling (StepLR) and gradient clipping.

**Model Persistence**: Saves the best performing model during training based on validation accuracy.

**Performance Visualization**: Custom plot function to visualize training loss and validation accuracy over epochs.

**Embedding Visualization**: Employs t-SNE (t-Distributed Stochastic Neighbor Embedding) with plotly.graph_objs to visualize the high-dimensional text embeddings in a 3D space, aiding in understanding the learned representations.

**Inference & Prediction**: Includes functions to make predictions on new text data and showcases the classification of unseen news articles.

# Dataset
This project uses the AG News Dataset, a widely recognized benchmark dataset for text classification. It consists of more than 1 million news articles from more than 2000 news sources. The dataset is categorized into 4 classes: World, Sports, Business, and Sci/Tec.

# Technologies & Libraries
**Python**: Programming Language

**PyTorch**: Deep Learning Framework

**torchtext*: For text-specific data processing (tokenization, vocabulary)

**NumPy**: Numerical operations

**Pandas**: Data manipulation (though primarily used for internal processing in this script)

**Matplotlib**: For 2D plotting of training metrics

**tqdm**: For progress bars during training

**scikit-learn**: For TSNE dimensionality reduction

**Plotly**: For interactive 3D visualization of embeddings

**IPython.display**: For rendering Markdown content in notebooks

# Getting Started
Prerequisites
Python 3.x

Pip (Python package installer)

Installation
Clone the repository (if applicable) or save the code:

git clone https://github.com/your_username/your_project_name.git
cd your_project_name


(If you're directly copying the script, just ensure you're in the directory where you saved the .py or .ipynb file.)

Install the required libraries:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu # or cuda if you have GPU
pip install numpy pandas matplotlib tqdm torchtext scikit-learn plotly


# Running the Code
The provided code is a self-contained script. You can execute it directly:

python your_script_name.py


Or, if you are running it in a Jupyter Notebook or Google Colab environment, simply run all cells in order.

The script will:

1. Download and prepare the AG News dataset.

2. Build a vocabulary from the training data.

3. Initialize and train the TextClassificationModel.

4. Plot the training loss and validation accuracy.

5. Evaluate the model on the test set.

6. Generate a 3D t-SNE visualization of text embeddings from a validation batch.

Demonstrate prediction on a sample article and a list of new articles.

# Results & Visualizations
Upon running the script, you will observe:

* Training Progress Plot: A matplotlib plot showing the total loss (cost) and accuracy over training epochs, helping to monitor convergence and identify overfitting.

* 3D t-SNE Visualization of Embeddings: An interactive Plotly 3D scatter plot. This visualization helps in understanding how well the EmbeddingBag layer has clustered different news categories in the embedding space. Ideally, articles from the same category should form distinct clusters.

Sample Predictions: The script will output the predicted category for a sample article and then for a list of diverse articles, demonstrating the model's inference capabilities.

# Project Structure
(Assuming your code is in a single .py or .ipynb file. If you have separate files, adjust this section accordingly.)

```
.
├── your_project_file.py  # Or your_notebook_file.ipynb
└── my_model.pth          # Saved model weights (generated after first run)
├── README.md             # This file

```

# Model Architecture 
The TextClassificationModel is a straightforward neural network:

*nn.EmbeddingBag*: This layer efficiently handles variable-length text sequences by averaging or summing embeddings for words in a sentence. It's particularly useful for text classification where the order of words might be less critical than the overall meaning.

*nn.Linear*: A fully connected layer that maps the combined embeddings to the number of output classes (4 for AG News).

The *init_weights* method ensures initial stability by uniformly distributing weights and zeroing biases.
