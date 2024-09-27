# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:52:19 2024

@author: Marios Telemachou
"""

#Import suitable packages
import numpy as np
from matplotlib import pyplot as plt
import os
import statistics
import scipy.stats as stats
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, cluster
import pandas as pd
from sklearn import metrics
from hyperopt import hp
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.utils import shuffle
import random
#%%
def read_fasta(file_path):
    """
    Reads a FASTA file and returns a dictionary with sequence IDs as keys
    and sequences as values.
    
    :param file_path: Path to the FASTA file
    :return: Dictionary with sequence IDs as keys and sequences as values
    """
    fasta_dict = {}
    with open(file_path, 'r') as file:
        sequence_id = None
        sequence = []
        
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line.startswith('>'):
                # If we encounter a new sequence ID, store the previous one
                if sequence_id is not None:
                    fasta_dict[sequence_id] = ''.join(sequence)
                sequence_id = line[1:]  # Remove the '>' character
                sequence = []  # Reset sequence list for the new sequence
            else:
                sequence.append(line)
        
        # Don't forget to save the last sequence
        if sequence_id is not None:
            fasta_dict[sequence_id] = ''.join(sequence)
    
    return fasta_dict

#Example usage:
file_path = 'ProteinFastaResults_Human.fasta'  
fasta_sequences = read_fasta(file_path)

#Print the sequences
for seq_id, sequence in fasta_sequences.items():
    print(f"ID: {seq_id}\nSequence: {sequence}\n")



#%%
#Returns the amino acid sequences in a list format and not in a dictionary format that the code at the beginning of the script is doing

def read_fasta(file_path):
    """
    Reads a FASTA file and returns the amino acid sequences.

    :param file_path: Path to the FASTA file.
    :return: A list of amino acid sequences.
    """
    sequences = []
    current_sequence = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove any leading/trailing whitespace
            if line.startswith('>'):  # Header line
                if current_sequence:
                    sequences.append(''.join(current_sequence))
                    current_sequence = []
            else:
                current_sequence.append(line)
        
        # Append the last sequence after the loop
        if current_sequence:
            sequences.append(''.join(current_sequence))
    
    return sequences
#%%
# Example usage for Protein Fasta results
file_path_1 = 'ProteinFastaResults_Human.fasta'
amino_acid_sequences = read_fasta(file_path_1)
protein_lengths = []
# Print the sequences
for i, seq in enumerate(amino_acid_sequences):
    print(f"Sequence {i+1}: {seq}")
    protein_lengths.append(len(seq))

print("The protein lengths list is:", protein_lengths)    
print("The mean is:",np.mean(protein_lengths))
print("The median is:",np.median(protein_lengths))
print("The mode is:", statistics.mode(protein_lengths))
print("The standard deviation is:", np.std(protein_lengths))
#q3, q1 = np.percentile(protein_lengths, [75, 25])
#iqr = q3 - q1
#print("The interquartile range is:", iqr)
print("The range is:", np.max(protein_lengths)-np.min(protein_lengths))
#Kurtosis is a statistical measure that describes the shape of a distribution's tails in relation to its overall shape
print("The kurtosis is:", stats.kurtosis(protein_lengths))
print("The skewness is:", stats.skew(protein_lengths))
plt.hist(protein_lengths, 10)
plt.xlabel("Protein Length", fontsize = 20)
plt.ylabel("Frequency", fontsize = 20)
plt.tick_params(labelsize=20)
plt.show()

#%%
#Example usage for HA_sequences_HUMAN
file_path_2 = 'HA_sequences_HUMAN.fasta'
amino_acid_sequences_HUMAN = read_fasta(file_path_2)
protein_lengths_HUMAN = []
# Print the sequences
for i, seq in enumerate(amino_acid_sequences_HUMAN):
    print(f"Sequence {i+1}: {seq}")
    protein_lengths_HUMAN.append(len(seq))

print("The protein lengths list is:", protein_lengths_HUMAN)    
print("The mean is:",np.mean(protein_lengths_HUMAN))
print("The median is:",np.median(protein_lengths_HUMAN))
print("The mode is:", statistics.mode(protein_lengths_HUMAN))
print("The standard deviation is:", np.std(protein_lengths_HUMAN))
#q3, q1 = np.percentile(protein_lengths, [75, 25])
#iqr = q3 - q1
#print("The interquartile range is:", iqr)
print("The range is:", np.max(protein_lengths_HUMAN)-np.min(protein_lengths_HUMAN))
#Kurtosis is a statistical measure that describes the shape of a distribution's tails in relation to its overall shape
print("The kurtosis is:", stats.kurtosis(protein_lengths_HUMAN))
print("The skewness is:", stats.skew(protein_lengths_HUMAN))
plt.hist(protein_lengths_HUMAN, 10)
plt.xlabel("Protein Length", fontsize = 20)
plt.ylabel("Frequency", fontsize = 20)
plt.tick_params(labelsize=20)
plt.show()

#%%
#Example usage for HA_sequences_NONHUMAN
file_path_3 = 'HA_sequences_NONHUMAN.fasta'
amino_acid_sequences_NONHUMAN = read_fasta(file_path_3)
protein_lengths_NONHUMAN = []
# Print the sequences
for i, seq in enumerate(amino_acid_sequences_NONHUMAN):
    print(f"Sequence {i+1}: {seq}")
    protein_lengths_NONHUMAN.append(len(seq))

print("The protein lengths list is:", protein_lengths_NONHUMAN)    
print("The mean is:",np.mean(protein_lengths_NONHUMAN))
print("The median is:",np.median(protein_lengths_NONHUMAN))
print("The mode is:", statistics.mode(protein_lengths_NONHUMAN))
print("The standard deviation is:", np.std(protein_lengths_NONHUMAN))
#q3, q1 = np.percentile(protein_lengths, [75, 25])
#iqr = q3 - q1
#print("The interquartile range is:", iqr)
print("The range is:", np.max(protein_lengths_NONHUMAN)-np.min(protein_lengths_NONHUMAN))
#Kurtosis is a statistical measure that describes the shape of a distribution's tails in relation to its overall shape
print("The kurtosis is:", stats.kurtosis(protein_lengths_NONHUMAN))
print("The skewness is:", stats.skew(protein_lengths_NONHUMAN))
plt.hist(protein_lengths_NONHUMAN, 10)
plt.xlabel("Protein Length", fontsize = 20)
plt.ylabel("Frequency", fontsize = 20)
plt.tick_params(labelsize=20)
plt.show()
#%%
# Define a dictionary to map amino acids to numbers
amino_acid_map = {
    'A': 1, # Alanine
    'C': 2, # Cysteine
    'D': 3, # Aspartic Acid
    'E': 4, # Glutamic Acid
    'F': 5, # Phenylalanine
    'G': 6, # Glycine
    'H': 7, # Histidine
    'I': 8, # Isoleucine
    'K': 9, # Lysine
    'L': 10, # Leucine
    'M': 11, # Methionine
    'N': 12, # Asparagine
    'P': 13, # Proline
    'Q': 14, # Glutamine
    'R': 15, # Arginine
    'S': 16, # Serine
    'T': 17, # Threonine
    'V': 18, # Valine
    'W': 19, # Tryptophan
    'Y': 20  # Tyrosine
}

#%%
# Function to encode amino acid sequences to numbers and pad to length 565
def encode_and_pad_sequence(sequence, max_length=569):
    """
    Encodes a sequence of amino acids into numbers and pads it with zeros up to max_length.

    Parameters:
    sequence (str): Amino acid sequence (e.g., 'ACDEFGH').
    max_length (int): The length to pad the encoded sequence to (default is 565).

    Returns:
    np.ndarray: Encoded and padded sequence.
    """
    # Encode the sequence
    encoded_sequence = [amino_acid_map.get(aa, 0) for aa in sequence]

    # Pad with zeros if sequence is shorter than max_length
    if len(encoded_sequence) < max_length:
        encoded_sequence = encoded_sequence + [0] * (max_length - len(encoded_sequence))
    elif len(encoded_sequence) > max_length:
        # Truncate the sequence if it's longer than max_length
        encoded_sequence = encoded_sequence[:max_length]

    return np.array(encoded_sequence)

# Function to process a list of sequences
def encode_and_pad_sequences(sequence_HUMAN, max_length=569):
    """
    Encodes and pads a list of amino acid sequences.

    Parameters:
    sequences (list of str): List of amino acid sequences.
    max_length (int): The length to pad each sequence to (default is 565).

    Returns:
    np.ndarray: 2D array where each row is an encoded and padded sequence.
    """
    return np.array([encode_and_pad_sequence(seq, max_length) for seq in sequence_HUMAN])

# Example: List of amino acid sequences for humans
sequence_HUMAN = amino_acid_sequences_HUMAN

# Step 2: Encode and pad sequences
encoded_sequences_HUMAN = encode_and_pad_sequences(sequence_HUMAN)

# Print the encoded and padded sequences
print("Encoded and Padded Sequences:")
for seq in encoded_sequences_HUMAN:
    print(seq)
    print(f"Length of encoded sequence: {len(seq)}")

#%%
def encode_and_pad_sequence(sequence, max_length=569):
    # Encode the sequence
    encoded_sequence = [amino_acid_map.get(aa, 0) for aa in sequence]

    # Pad with zeros if sequence is shorter than max_length
    if len(encoded_sequence) < max_length:
        encoded_sequence = encoded_sequence + [0] * (max_length - len(encoded_sequence))
    elif len(encoded_sequence) > max_length:
        # Truncate the sequence if it's longer than max_length
        encoded_sequence = encoded_sequence[:max_length]

    return np.array(encoded_sequence)

def encode_and_pad_sequences(sequence_NONHUMAN, max_length=569):
    
    return np.array([encode_and_pad_sequence(seq, max_length) for seq in sequence_NONHUMAN])

# Example: List of amino acid sequences for humans
sequence_NONHUMAN = amino_acid_sequences_NONHUMAN

# Step 2: Encode and pad sequences
encoded_sequences_NONHUMAN = encode_and_pad_sequences(sequence_NONHUMAN)

# Print the encoded and padded sequences
print("Encoded and Padded Sequences:")
for seq in encoded_sequences_NONHUMAN:
    print(seq)
    print(f"Length of encoded sequence: {len(seq)}")

#%%
Human_array = encoded_sequences_HUMAN
Nonhuman_array = encoded_sequences_NONHUMAN
print(Human_array)
print(len(Human_array))
total_sequences_Human = Human_array.shape[0]
print("Total number of human encoded sequences:", total_sequences_Human)
print(Nonhuman_array)
total_sequences_Non_Human = Nonhuman_array.shape[0]
print("Total number of human encoded sequences:", total_sequences_Non_Human)
print(len(Nonhuman_array))
combined_HUMAN_NONHUMAN_array = np.concatenate((Human_array, Nonhuman_array))
print(combined_HUMAN_NONHUMAN_array)

def assign_labels(encoded_sequences, human_count=46546):
    """
    Assigns labels ('Human' or 'Non-Human') to encoded protein sequences.

    Parameters:
    encoded_sequences (np.ndarray): 2D array of encoded sequences, where each row is a sequence.
    human_count (int): Number of sequences to label as 'Human'.

    Returns:
    labeled_sequences (list of tuples): A list of tuples where each tuple contains (encoded_sequence, label).
    """
    labeled_sequences = []
    
    total_sequences = encoded_sequences.shape[0]  # Get total number of sequences
    
    for i in range(total_sequences):
        # Check if the current index is less than human_count
        label = 'Human' if i < human_count else 'Non-Human'
        labeled_sequences.append((encoded_sequences[i], label))
    
    return labeled_sequences

labelled_encoded_sequence_HUMAN_NONHUMAN = assign_labels(combined_HUMAN_NONHUMAN_array)

for sequence, label in labelled_encoded_sequence_HUMAN_NONHUMAN[:5]:  # Print first 5 for brevity
    print(f"Encoded Sequence: {sequence}, Label: {label}")

# Optional: Print the total number of Human and Non-Human sequences
total_human = sum(1 for _, label in labelled_encoded_sequence_HUMAN_NONHUMAN if label == 'Human')
total_non_human = sum(1 for _, label in labelled_encoded_sequence_HUMAN_NONHUMAN if label == 'Non-Human')

print(f"Total Human sequences: {total_human}")
print(f"Total Non-Human sequences: {total_non_human}")
#%%

#shuffled_labelled_encoded_sequence_HUMAN_NONHUMAN  = shuffle(labelled_encoded_sequence_HUMAN_NONHUMAN, random_state=42)
#print(shuffled_labelled_encoded_sequence_HUMAN_NONHUMAN)
shuffled_combined_encoded_sequence_HUMAN_NONHUMAN = shuffle(combined_HUMAN_NONHUMAN_array, random_state = 42)
print(shuffled_combined_encoded_sequence_HUMAN_NONHUMAN)
#%%   
#Convolutional Neural Network Method 

# Class labels
classes = ('Human', 'Non-Human')

# Create datasets for training & validation, download if necessary
#print(len(encoded_sequence_HUMAN))
#training_set = shuffled_labelled_encoded_sequence_HUMAN_NONHUMAN
#validation_set = shuffled_labelled_encoded_sequence_HUMAN_NONHUMAN
#testing_set = shuffled_labelled_encoded_sequence_HUMAN_NONHUMAN
training_set = shuffled_combined_encoded_sequence_HUMAN_NONHUMAN
validation_set = shuffled_combined_encoded_sequence_HUMAN_NONHUMAN
testing_set = shuffled_combined_encoded_sequence_HUMAN_NONHUMAN


test_size_1 = 0.2
X_train, X_test_val, Y_train, Y_test_val = train_test_split(training_set, validation_set, test_size = test_size_1, random_state = 73)  
test_size_2 = 0.5
X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size = test_size_2, random_state = 73)

#Defining tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float)
X_val_tensor = torch.tensor(X_val, dtype=torch.float)
Y_val_tensor = torch.tensor(Y_val, dtype=torch.float)
X_test_tensor = torch.tensor(X_test, dtype=torch.float)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float)
#print(X_train_tensor)

#Adding dimensions to the tensor adds a new dimension at index 0 and adds a new dimension at index 1
X_train_tensor = X_train_tensor.unsqueeze(0).unsqueeze(1)
Y_train_tensor = Y_train_tensor.unsqueeze(0).unsqueeze(1)
X_val_tensor = X_val_tensor.unsqueeze(0).unsqueeze(1)
Y_val_tensor = Y_val_tensor.unsqueeze(0).unsqueeze(1)
X_test_tensor = X_test_tensor.unsqueeze(0).unsqueeze(1)
Y_test_tensor = Y_test_tensor.unsqueeze(0).unsqueeze(1)

#Reshaping the tensor without changing the data, allows to specify a new shape for the tensor, while keeping the underlying data the same
#X_train_tensor = X_train_tensor.view(-1, 74)
#Y_train_tensor = Y_train_tensor.view(-1, 74)
#X_val_tensor = X_val_tensor.view(-1, 65)
#Y_val_tensor = Y_val_tensor.view(-1, 65)
#X_test_tensor = X_test_tensor.view(-1, 65)
#Y_test_tensor = Y_test_tensor.view(-1, 65)

#Create TensorDataset for each split
train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

#Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last = True)

#train_loader, val_loader and test_loader can be used for training, validation and testing
for batch in train_loader:
    inputs, targets = batch


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        self.conv1 = nn.Conv1d(1, 6, 3) #previously nn.Conv1d(1,6,3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(6, 16, 3)
        self.fc1 = nn.Linear(142, 120) #Divide by 4 the protein length because of maxpooling performed twice.
        self.fc2 = nn.Linear(120, 84)  #Question: I use the mean from 1 file what about the other file?
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 142) #Divide by 4 the total length because the pool function is called twice #141 before
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvolutionalNeuralNetwork()
#PyTorch Training Loop
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training loop
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
#Testing loop
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

#Training and testing the model
model.to(device)
for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, criterion, epoch)
    test_loss, accuracy = test(model, device, test_loader)
    print(f'Epoch: {epoch}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
#%%
#Boosted Decision Tree Method
# Class labels
classes = ('Human', 'Non-Human')
data_reshaped = combined_HUMAN_NONHUMAN_array.reshape(-1, 1)  # Convert to 2D array of shape (6, 1)

print(data_reshaped)

# Create datasets for training & validation, download if necessary
#print(len(encoded_sequence_HUMAN))
training_set = data_reshaped
validation_set = data_reshaped
testing_set = data_reshaped

test_size_1 = 0.2
X_train, X_test_val, Y_train, Y_test_val = train_test_split(training_set, validation_set, test_size = test_size_1, random_state = 73)  
test_size_2 = 0.5
X_val, X_test, Y_val, Y_test = train_test_split(X_test_val, Y_test_val, test_size = test_size_2, random_state = 73)

model = XGBClassifier()
model.fit(X_train, Y_train)

y_pred = model.predict_proba(X_test)
predictions = [round(value) for value in y_pred[:, 1]]

# scores
#df = pd.DataFrame()
#df["truth"] = Y_test
#df["pred_round"] = predictions
#df["pred"] = y_pred[:, 1]
#df.to_csv("scores.csv")

# accuracy metric - not useful
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#print(model)

# predictions
y_pred = model.predict_proba(X_test)
predictions = [round(value) for value in y_pred[:, 1]]

