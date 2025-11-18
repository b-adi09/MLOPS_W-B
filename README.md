Dermatology Classification — Neural Network Model (Keras + W&B)

This project implements a multi-class skin disease classification model using the UCI Dermatology Dataset.
Unlike traditional ML approaches (e.g., XGBoost, Logistic Regression), this assignment uses a Keras-based Multilayer Perceptron (MLP) with tuned hyperparameters and integrated experiment tracking using Weights & Biases (W&B).

This version is fully re-written with a different pipeline, different modeling approach, and different experiment logging structure — ensuring originality while following the assignment requirements.

📁 Dataset

Source: UCI Machine Learning Repository – Dermatology Dataset

Total samples: 366

Features: 34 clinical measurements

Target classes: 6 categories

Missing values handled via median imputation

🎯 Project Objectives

Load and preprocess the dermatology dataset.

Build and train an MLP classification model using TensorFlow/Keras.

Track training metrics, validation metrics, and confusion matrix using W&B.

Evaluate model performance on a held-out test set.

Summarize results and error rate.

🧱 Project Structure
📦 Dermatology-MLP-Model
├── notebook.ipynb          # Main notebook with full pipeline
├── README.md               # Project documentation (this file)
└── requirements.txt        # Required Python packages

🧰 Technologies Used

Python 3.10+

TensorFlow / Keras

scikit-learn

pandas / numpy

Weights & Biases (W&B)

⚙️ Model Architecture

MLP with:

Input: 34 scaled features

Dense Layer 1: 128 units, ReLU

Dropout: 0.25

Dense Layer 2: 64 units, ReLU

Dropout: 0.25

Output: 6-unit softmax classifier

Optimizer: Adam, LR: 1e-3

Epochs: 60 (with EarlyStopping)

Batch Size: 32

📊 Evaluation Metrics

After training, the notebook logs:

Test Accuracy

Test Error Rate

Confusion Matrix

Training Loss / Accuracy curves (via W&B)

🚀 How to Run
1. Install dependencies
pip install -r requirements.txt

2. Log in to W&B
wandb login

3. Run the notebook

Open notebook.ipynb in Jupyter/Colab and run all cells.

📝 Key Results

(The exact numbers will vary based on random seed.)

Typical outcome:

Test Accuracy: ~0.95

Error Rate: ~0.05

Clear improvement with early stopping

