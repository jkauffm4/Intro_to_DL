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

english_to_french = [
("I am cold", "J'ai froid"),
("You are tired", "Tu es fatigué"),
("He is hungry", "Il a faim"),
("She is happy", "Elle est heureuse"),
("We are friends", "Nous sommes amis"),
("They are students", "Ils sont étudiants"),
("The cat is sleeping", "Le chat dort"),
("The sun is shining", "Le soleil brille"),
("We love music", "Nous aimons la musique"),
("She speaks French fluently", "Elle parle français couramment"),
("He enjoys reading books", "Il aime lire des livres"),
("They play soccer every weekend", "Ils jouent au football chaque week-end"),
("The movie starts at 7 PM", "Le film commence à 19 heures"),
("She wears a red dress", "Elle porte une robe rouge"),
("We cook dinner together", "Nous cuisinons le dîner ensemble"),
("He drives a blue car", "Il conduit une voiture bleue"),
("They visit museums often", "Ils visitent souvent des musées"),
("The restaurant serves delicious food", "Le restaurant sert une délicieuse cuisine"),
("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
("We watch movies on Fridays", "Nous regardons des films le vendredi"),
("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
("They travel around the world", "Ils voyagent autour du monde"),
("The book is on the table", "Le livre est sur la table"),
("She dances gracefully", "Elle danse avec grâce"),
("We celebrate birthdays with cake", "Nous célébrons les anniversaires avec un gâteau"),
("He works hard every day", "Il travaille dur tous les jours"),
("They speak different languages", "Ils parlent différentes langues"),
("The flowers bloom in spring", "Les fleurs fleurissent au printemps"),
("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
("We learn something new every day", "Nous apprenons quelque chose de nouveau chaque jour"),
("The dog barks loudly", "Le chien aboie bruyamment"),
("He sings beautifully", "Il chante magnifiquement"),
("They swim in the pool", "Ils nagent dans la piscine"),
("The birds chirp in the morning", "Les oiseaux gazouillent le matin"),
("She teaches English at school", "Elle enseigne l'anglais à l'école"),
("We eat breakfast together", "Nous prenons le petit déjeuner ensemble"),
("He paints landscapes", "Il peint des paysages"),
("They laugh at the joke", "Ils rient de la blague"),
("The clock ticks loudly", "L'horloge tic-tac bruyamment"),
("She runs in the park", "Elle court dans le parc"),
("We travel by train", "Nous voyageons en train"),
("He writes a letter", "Il écrit une lettre"),
("They read books at the library", "Ils lisent des livres à la bibliothèque"),
("The baby cries", "Le bébé pleure"),
("She studies hard for exams", "Elle étudie dur pour les examens"),
("We plant flowers in the garden", "Nous plantons des fleurs dans le jardin"),
("He fixes the car", "Il répare la voiture"),
("They drink coffee in the morning", "Ils boivent du café le matin"),
("The sun sets in the evening", "Le soleil se couche le soir"),
("She dances at the party", "Elle danse à la fête"),
("We play music at the concert", "Nous jouons de la musique au concert"),
("He cooks dinner for his family", "Il cuisine le dîner pour sa famille"),
("They study French grammar", "Ils étudient la grammaire française"),
("The rain falls gently", "La pluie tombe doucement"),
("She sings a song", "Elle chante une chanson"),
("We watch a movie together", "Nous regardons un film ensemble"),
("He sleeps deeply", "Il dort profondément"),
("They travel to Paris", "Ils voyagent à Paris"),
("The children play in the park", "Les enfants jouent dans le parc"),
("She walks along the beach", "Elle se promène le long de la plage"),
("We talk on the phone", "Nous parlons au téléphone"),
("He waits for the bus", "Il attend le bus"),
("They visit the Eiffel Tower", "Ils visitent la tour Eiffel"),
("The stars twinkle at night", "Les étoiles scintillent la nuit"),
("She dreams of flying", "Elle rêve de voler"),
("We work in the office", "Nous travaillons au bureau"),
("He studies history", "Il étudie l'histoire"),
("They listen to the radio", "Ils écoutent la radio"),
("The wind blows gently", "Le vent souffle doucement"),
("She swims in the ocean", "Elle nage dans l'océan"),
("We dance at the wedding", "Nous dansons au mariage"),
("He climbs the mountain", "Il gravit la montagne"),
("They hike in the forest", "Ils font de la randonnée dans la forêt"),
("The cat meows loudly", "Le chat miaule bruyamment"),
("She paints a picture", "Elle peint un tableau"),
("We build a sandcastle", "Nous construisons un château de sable"),
("He sings in the choir", "Il chante dans le chœur"),
("They ride bicycles", "Ils font du vélo"),
("The coffee is hot", "Le café est chaud"),
("She wears glasses", "Elle porte des lunettes"),
("We visit our grandparents", "Nous rendons visite à nos grands-parents"),
("He plays the guitar", "Il joue de la guitare"),
("They go shopping", "Ils font du shopping"),
("The teacher explains the lesson", "Le professeur explique la leçon"),
("She takes the train to work", "Elle prend le train pour aller au travail"),
("We bake cookies", "Nous faisons des biscuits"),
("He washes his hands", "Il se lave les mains"),
("They enjoy the sunset", "Ils apprécient le coucher du soleil"),
("The river flows calmly", "La rivière coule calmement"),
("She feeds the cat", "Elle nourrit le chat"),
("We visit the museum", "Nous visitons le musée"),
("He fixes his bicycle", "Il répare son vélo"),
("They paint the walls", "Ils peignent les murs"),
("The baby sleeps peacefully", "Le bébé dort paisiblement"),
("She ties her shoelaces", "Elle attache ses lacets"),
("We climb the stairs", "Nous montons les escaliers"),
("He shaves in the morning", "Il se rase le matin"),
("They set the table", "Ils mettent la table"),
("The airplane takes off", "L'avion décolle"),
("She waters the plants", "Elle arrose les plantes"),
("We practice yoga", "Nous pratiquons le yoga"),
("He turns off the light", "Il éteint la lumière"),
("They play video games", "Ils jouent aux jeux vidéo"),
("The soup smells delicious", "La soupe sent délicieusement bon"),
("She locks the door", "Elle ferme la porte à clé"),
("We enjoy a picnic", "Nous profitons d'un pique-nique"),
("He checks his email", "Il vérifie ses emails"),
("They go to the gym", "Ils vont à la salle de sport"),
("The moon shines brightly", "La lune brille intensément"),
("She catches the bus", "Elle attrape le bus"),
("We greet our neighbors", "Nous saluons nos voisins"),
("He combs his hair", "Il se peigne les cheveux"),
("They wave goodbye", "Ils font un signe d'adieu")
]

class English2FrenchDataset(Dataset):
    """Custom Dataset class for handling synonym pairs."""
    def __init__(self, dataset, char_to_index):
        self.dataset = dataset
        self.char_to_index = char_to_index

    def __len__(self):
        # Returns the total number of synonym pairs in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieves a synonym pair by index, converts characters to indices,
        # and adds the EOS token at the end of each word.
        input_word, target_word = self.dataset[idx]
        input_tensor = torch.tensor([self.char_to_index[char] for char in input_word] + [EOS_token], dtype=torch.long)
        target_tensor = torch.tensor([self.char_to_index[char] for char in target_word] + [EOS_token], dtype=torch.long)
        return input_tensor, target_tensor

class CharTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super(CharTransformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer_encoder(embedded)
        output = self.fc(transformer_output[:, -1, :])  # Get the output of the last Transformer block
        return output

class EncodeDecodeTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, nhead):
        super().__init__()
        self.embedding_input = nn.Embedding(input_size, hidden_size)
        self.embedding_output = nn.Embedding(output_size, hidden_size)
        self.transformer = nn.Transformer(d_model=128, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers,dim_feedforward=hidden_size*4)
        self.fc_out = nn.Linear(hidden_size, output_size)
    
    def forward(self, src, tgt):
        src_emb = self.embedding_input(src).permute(1,0,2)
        tgt_emb = self.embedding_output(tgt).permute(1,0,2)
        output = self.transformer(src_emb, tgt_emb)
        return self.fc_out(output).permute(1,0,2)

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

    """file_path = "sequence.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    chars = sorted(list(set(text)))
    #This line creates a dictionary that maps each character to a unique index (integer)."
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    #Similar to the previous line, but in reverse. This line creates a dictionary that maps each unique index (integer) back to its corresponding character.
    char_to_ix = {ch: i for i, ch in enumerate(chars)} 
    chars = sorted(list(set(text)))

    # Preparing the dataset
    max_length = 30  # Maximum length of input sequences
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
    
    # Converting data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_train.to(device)
    y_train.to(device)
    X_val.to(device)
    y_val.to(device)

    input_size = len(chars)
    hidden_size = 128
    output_size = len(chars)
    num_layers = 3
    nhead = 2

    model = CharTransformer(input_size, hidden_size, output_size, num_layers, nhead)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = .1, momentum=.9)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
    epochs = 100


    for epoch in range(epochs):
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
            print(f'Epoch {epoch+1}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_accuracy.item()}')"""
    
    """# SHAKESPEARE LOADER
    # Step 1: Download the dataset
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text  # This is the entire text data

# Step 2: Prepare the dataset
    sequence_length = 30
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
    targets = torch.tensor(targets, dtype=torch.long)

# Instantiate the dataset
    dataset = CharDataset(sequences, targets)

# Step 4: Create data loaders
    batch_size = 512
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# Now `train_loader` and `test_loader` are ready to be used in a training loop

    #MODELS
        
    input_size = len(chars)
    hidden_size = 128
    output_size = len(chars)
    num_layer = 4
    nhead = 2

    model = CharTransformer(input_size, hidden_size, output_size, num_layer, nhead)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = .1, momentum=.9)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []
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

        print(f'Epoch {epoch + 1}, Training loss: {train_loss_list[-1]}, Validation loss: {val_loss_list[-1]}, Validation Accuracy: {val_accuracy}%')"""

# Special tokens for the start and end of sequence
    SOS_token = 0  # Start Of Sequence Token
    EOS_token = 1  # End Of Sequence Token
    max_length = 70

    char_to_index = {"SOS": SOS_token, "EOS": EOS_token, **{char: i+2 for i, char in enumerate(sorted(list(set(''.join([word for pair in english_to_french for word in pair])))))}}
    index_to_char = {i: char for char, i in char_to_index.items()}
# Creating a DataLoader to batch and shuffle the dataset
    eng_french_dataset = English2FrenchDataset(english_to_french, char_to_index)
    dataloader = DataLoader(eng_french_dataset, batch_size=1, shuffle=True)


# Preparing the character to index mapping and vice versa
# These mappings will help convert characters to numerical format for the neural network
# 'SOS' and 'EOS' tokens are added at the start of the char_to_index dictionary
    char_to_index = {"SOS": SOS_token, "EOS": EOS_token, **{char: i+2 for i, char in enumerate(sorted(list(set(''.join([word for pair in english_to_french for word in pair])))))}}
    index_to_char = {i: char for char, i in char_to_index.items()}

    #MODELS
        
    input_size = len(char_to_index)
    hidden_size = 128
    output_size = len(char_to_index)
    num_layer = 4
    nhead = 4

    model = EncodeDecodeTransformer(input_size, hidden_size, output_size, num_layer, nhead)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = .1, momentum=.9)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in range(10):
        model.train()
        epoch_loss = 0
        for tgt, src in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.shape[-1]), tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(dataloader)
        train_loss_list.append(avg_train_loss)

    # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for tgt, src in dataloader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt)
                loss = criterion(output.view(-1, output.shape[-1]), tgt.reshape(-1))
                val_loss += loss.item()

            # Compute accuracy
                preds = output.argmax(dim=-1)
                correct += (preds == tgt).sum().item()
                total += tgt.numel()

        avg_val_loss = val_loss / len(dataloader)
        val_loss_list.append(avg_val_loss)
        accuracy = correct / total

        print(f'Epoch {epoch + 1}, Training loss: {avg_train_loss}, Validation loss: {avg_val_loss}, Validation Accuracy: {accuracy}%')

    val_loss_list.append(avg_val_loss)
    val_accuracy_list.append(accuracy)
    plt.plot(train_loss_list, label='Training Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('plot')
    plt.show()
