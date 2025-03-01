import torch
from torchvision import transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import requests

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer for output

    def forward(self, x):
        out, _ = self.rnn(x)  # Process input through RNN
        out = self.fc(out[:, -1, :])  # Pass the last time step output to linear layer
        return out
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=4,batch_first=True)  # RNN layer
        self.fc1 = nn.Linear(hidden_size, 240)  # Linear layer for output
        self.fc2 = nn.Linear(240, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # Process input through RNN
        out = self.fc1(out[:, -1, :])  # Pass the last time step output to linear layer
        out = self.fc2(out)
        return out
    
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)  # RNN layer
        self.fc = nn.Linear(hidden_size, output_size)  # Linear layer for output

    def forward(self, x):
        out, _ = self.rnn(x)  # Process input through RNN
        out = self.fc(out[:, -1, :])  # Pass the last time step output to linear layer
        return out
    
class CharDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index], self.targets[index]

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    torch.cuda.empty_cache()

    ##### UNCOMMENT BELOW FOR FIRST DATASET ########
    """
    file_path = "sequence.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    chars = sorted(list(set(text)))
    #This line creates a dictionary that maps each character to a unique index (integer)."
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    #Similar to the previous line, but in reverse. This line creates a dictionary that maps each unique index (integer) back to its corresponding character.
    char_to_ix = {ch: i for i, ch in enumerate(chars)} 
    chars = sorted(list(set(text)))

    # Preparing the dataset
    max_length = 10  # Maximum length of input sequences
    X = []
    y = []
    for i in range(len(text) - max_length):
        sequence = text[i:i + max_length]
        label = text[i + max_length]
        X.append([char_to_ix[char] for char in sequence])
        y.append(char_to_ix[label])

    X = np.array(X)
    y = np.array(y)

    # Splitting the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_size = len(chars)
    hidden_size = 24
    output_size = len(chars)

    # Converting data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_train = F.one_hot(X_train, num_classes=input_size).float()
    X_val = F.one_hot(X_val, num_classes=input_size).float()
    X_train.to(device)
    y_train.to(device)
    X_val.to(device)
    y_val.to(device)

    """
    # SHAKESPEARE LOADER
    # Step 1: Download the dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text  # This is the entire text data

# Step 2: Prepare the dataset
    sequence_length = 50
# Create a character mapping to integers
    chars = sorted(list(set(text)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

# Encode the text into integers
    encoded_text = [char_to_int[ch] for ch in text]

# Create sequences and targets
    sequences = []
    targets = []
    for i in range(0, len(encoded_text) - sequence_length):
        seq = encoded_text[i:i+sequence_length]
        target = encoded_text[i+sequence_length]
        sequences.append(seq)
        targets.append(target)

# Convert lists to PyTorch tensors
    sequences = torch.tensor(sequences, dtype=torch.long)
    sequences = F.one_hot(sequences, num_classes=len(chars)).float()
    targets = torch.tensor(targets, dtype=torch.long)

# Instantiate the dataset
    dataset = CharDataset(sequences, targets)

# Step 4: Create data loaders
    batch_size = 256
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# Now `train_loader` and `test_loader` are ready to be used in a training loop

    #MODELS
        
    input_size = len(chars)
    hidden_size = 240
    output_size = len(chars)

    #model = RNNModel(input_size, hidden_size, output_size)
    #model = LSTMModel(input_size, hidden_size, output_size)
    model = GRUModel(input_size, hidden_size, output_size)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = .1, momentum=.9)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    epochs = 2000

# Training the model
    ## Below is training loop for Question 1 of HW
    """for epoch in range(epochs):
        running_loss = 0
        model.train()
        optimizer.zero_grad()
        output = model(X_train.to(device))
        loss = criterion(output, y_train.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loss_list.append(running_loss)
        #train_loss_list.append(running_loss / len(trainloader))
    
        running_val_loss = 0.0
        correct = 0
        total = 0
    # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val.to(device))
            val_loss = criterion(val_output, y_val.to(device))
            #The use of the underscore _ is a common Python convention to indicate that the actual maximum values returned by torch.max are not needed and can be disregarded. 
            #What we are interested in is the indices of these maximum values, which are captured by the variable predicted. These indices represent the model's predictions for each example in the validation set.
            _, predicted = torch.max(val_output, 1)
            val_accuracy = (predicted == y_val.to(device)).float().mean()
            running_val_loss += val_loss.item()
        #val_loss_list.append(running_loss / len(testloader))
        val_loss_list.append(running_val_loss)
        #val_accuracy = 100 * correct / total
        val_accuracy_list.append(val_accuracy)
    
        if (epoch+1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')
"""
## Below is training loop for Question 2 of HW
    for epoch in range(30):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()  # Set the model to training mode
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss_list.append(running_loss / len(train_loader))

        # Validation loop
        running_loss = 0.0
        correct = 0
        total = 0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss_list.append(running_loss / len(test_loader))
        val_accuracy = 100 * correct / total
        val_accuracy_list.append(val_accuracy)

        print(f'Epoch {epoch + 1}, Training loss: {train_loss_list[-1]}, Validation loss: {val_loss_list[-1]}, Validation Accuracy: {val_accuracy}%')

# Print final validation accuracy
    print(f'Final Validation Accuracy: {val_accuracy_list[-1]}%')
# Calculate total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total number of parameters in the model: {total_params}')

    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('RNN_Question_1_Len10')
    plt.show()