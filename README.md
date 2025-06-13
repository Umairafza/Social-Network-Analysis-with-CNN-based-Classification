Social Network Analysis with CNN-based Classification
This repository contains the code for a project developed as part of the Social Network Theory and Practice course. The project focuses on analyzing social network data using a Convolutional Neural Network (CNN) implemented in TensorFlow to perform binary classification on processed network features. The dataset used is the bio-WormNet-v3.edges, which represents a biological network with weighted edges between nodes.
Project Overview
The project processes network data through three distinct feature extraction methods—frequency processing, frequency-weighted processing, and difference processing—to generate input features for a CNN model. The model is trained to classify network patterns, with techniques like SMOTE for handling class imbalance, class weighting, and early stopping to optimize performance. The performance is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.
Features

Data Preprocessing: Reads and processes the bio-WormNet-v3.edges dataset, handling missing weights and applying modular arithmetic to source node values.
Feature Extraction:
Frequency Processing: Computes normalized frequency histograms of node values.
Frequency-Weighted Processing: Applies inverse distance weighting to emphasize central values in histograms.
Difference Processing: Captures differences between consecutive node values for dynamic pattern analysis.


Data Augmentation: Uses SMOTE to address class imbalance in the training data.
CNN Model: Implements a 1D CNN with three convolutional layers, batch normalization, max pooling, and dense layers for binary classification.
Evaluation Metrics: Computes accuracy, precision, recall, F1-score, and ROC-AUC, with an optimal threshold selected from the precision-recall curve.
Model Optimization: Incorporates early stopping, model checkpointing, and class weights to enhance training stability and performance.

Technologies Used

Python: Core programming language for data processing and model implementation.
TensorFlow/Keras: Framework for building and training the CNN model.
Pandas/NumPy: Libraries for data manipulation and numerical computations.
Scikit-learn: Used for data splitting, scaling, and evaluation metrics.
Imbalanced-learn (SMOTE): For handling class imbalance in the dataset.

Repository Structure

main_script.py: Contains the complete code for data loading, preprocessing, feature extraction, model training, and evaluation.
bio-WormNet-v3.edges: Sample dataset (not included; user must provide the file path).
README.md: This file, providing an overview and instructions for the project.

How to Run

Prerequisites:

Install Python 3.8+.
Install required libraries:pip install tensorflow pandas numpy scikit-learn imbalanced-learn


Download the bio-WormNet-v3.edges dataset and place it in the project directory (update file_path in the code accordingly).


Execution:

Clone the repository:git clone <repository-url>
cd <repository-directory>


Run the script:python main_script.py


The script will preprocess the data, train the CNN model on three feature sets, and output the training accuracies and evaluation metrics.



Output
The script outputs the final training accuracies for the three processing methods:

Frequency Processing: Training accuracy (e.g., X.XX%).
Frequency-Weighted Processing: Training accuracy (e.g., X.XX%).
Difference Processing: Training accuracy (e.g., X.XX%).

Additional metrics (precision, recall, F1-score, ROC-AUC) are computed but not printed in the provided output snippet. Modify the script to display these metrics if needed.
Limitations

The dataset is assumed to be in a specific format (bio-WormNet-v3.edges); other formats may require preprocessing adjustments.
The random generation of labels (y_data) may not reflect real-world network properties, limiting practical applicability.
The model is trained on synthetic features derived from random data, which may not generalize to actual network analysis tasks.

Future Improvements

Integrate real-world network labels for more meaningful classification.
Experiment with additional feature extraction techniques or graph-based models (e.g., Graph Neural Networks).
Optimize hyperparameters using grid search or Bayesian optimization.
Add visualization of precision-recall curves and confusion matrices for better model interpretability.

Contributors

Developed as part of the Social Network Theory and Practice course project.

Acknowledgments

The dataset (bio-WormNet-v3.edges) is sourced from a biological network study (ensure proper citation if used).
Thanks to the course instructors for guidance and support during the project development.

