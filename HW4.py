# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Dataset
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

# Special tokens for the start and end of sequences
SOS_token = 0  # Start Of Sequence Token
EOS_token = 1  # End Of Sequence Token
max_length = 70

# Preparing the character to index mapping and vice versa
# These mappings will help convert characters to numerical format for the neural network
# 'SOS' and 'EOS' tokens are added at the start of the char_to_index dictionary
char_to_index = {"SOS": SOS_token, "EOS": EOS_token, **{char: i+2 for i, char in enumerate(sorted(list(set(''.join([word for pair in english_to_french for word in pair])))))}}
index_to_char = {i: char for char, i in char_to_index.items()}

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

# Creating a DataLoader to batch and shuffle the dataset
eng_french_dataset = English2FrenchDataset(english_to_french, char_to_index)
dataloader = DataLoader(eng_french_dataset, batch_size=1, shuffle=True)

# Setting the device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """The Encoder part of the seq2seq model."""
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # Embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # LSTM layer

    def forward(self, input, hidden):
        # Forward pass for the encoder
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        # Initializes hidden state
        return (torch.zeros(1, 1, self.hidden_size, device=device))
    
class Decoder(nn.Module):
    """The Decoder part of the seq2seq model."""
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)  # Embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # LSTM layer
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
                             
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device))
    
class AttnDecoder(nn.Module):
    """Decoder with attention mechanism."""
    def __init__(self, hidden_size, output_size, max_length=max_length, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # Attention weights
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        # Combine embedded input and context vector
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Calculate attention weights
        attn_weights = torch.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # Apply attention weights to encoder outputs to get context
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)

        output = torch.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return (torch.zeros(1, 1, self.hidden_size, device=device))

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=50):
    encoder.train()
    decoder.train()
    # Initialize encoder hidden state
    encoder_hidden = encoder.initHidden()

    # Clear gradients for optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Calculate the length of input and target tensors
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Initialize loss
    loss = 0

    # Encoding each character in the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
  
    # Decoder's first input is the SOS token
    decoder_input = torch.tensor([[char_to_index['SOS']]], device=device)

    # Decoder starts with the encoder's last hidden state
    decoder_hidden = encoder_hidden

    # Decoding loop
    for di in range(target_length):
        # For decoder without attention
        #decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)

        # For decoder with attention
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # Choose top1 word from decoder's output
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Detach from history as input

        # Calculate loss
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == char_to_index['EOS']:  # Stop if EOS token is generated
            break

    # Backpropagation
    loss.backward()

    # Update encoder and decoder parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return average loss
    return loss.item() / target_length
    
def train_w_attention(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=70):
    encoder.train()
    decoder.train()
    
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # Encode each character in the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder's first input is the SOS token
    decoder_input = torch.tensor([[char_to_index['SOS']]], device=device)

    # Initial decoder hidden state is encoder's last hidden state
    decoder_hidden = encoder_hidden

    # Decoder with attention
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Detach from history as input

        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == char_to_index['EOS']:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate_and_show_examples(encoder, decoder, dataloader, criterion, val_loss_list, n_examples=5):
    # Switch model to evaluation mode
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    correct_predictions = 0
    
    # No gradient calculation
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            # Move tensors to the correct device
            # Uncomment below for English to French
            #input = input_tensor[0].to(device)
            #target = target_tensor[0].to(device)

            # Uncomment below for French to English translation
            input = target_tensor[0].to(device)
            target = input_tensor[0].to(device)            

            encoder_hidden = encoder.initHidden()

            input_length = input.size(0)
            target_length = target.size(0)

            loss = 0

            # Encoding step
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input[ei].unsqueeze(0), encoder_hidden)

            # Decoding step
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            # Calculate and print loss and accuracy for the evaluation
            total_loss += loss.item() / target_length
            if predicted_indices == target.tolist():
                correct_predictions += 1

            if i < n_examples:
                predicted_string = ''.join([index_to_char[index] for index in predicted_indices if index not in (SOS_token, EOS_token)])
                target_string = ''.join([index_to_char[index.item()] for index in target if index.item() not in (SOS_token, EOS_token)])
                input_string = ''.join([index_to_char[index.item()] for index in input if index.item() not in (SOS_token, EOS_token)])
                
                print(f'Input: {input_string}, Target: {target_string}, Predicted: {predicted_string}')
            
        
        # Print overall evaluation results
        average_loss = total_loss / len(dataloader)
        val_loss_list.append(average_loss)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')

def evaluate_and_show_examples_attention(encoder, decoder, dataloader, criterion, val_loss_list, n_examples=5):
    # Switch model to evaluation mode
    encoder.eval()
    decoder.eval()
    
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            # Move tensors to the correct device
            # Uncomment below for English to French
            #input = input_tensor[0].to(device)
            #target = target_tensor[0].to(device)

            # Uncomment below for French to English translation
            input = target_tensor[0].to(device)
            target = input_tensor[0].to(device)            

            encoder_hidden = encoder.initHidden()
            input_length = input.size(0)
            target_length = target.size(0)

            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            loss = 0

            # Encode input
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # Decode with attention
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            total_loss += loss.item() / target_length
            if predicted_indices == target.tolist():
                correct_predictions += 1

            if i < n_examples:
                predicted_string = ''.join([index_to_char[index] for index in predicted_indices if index not in (SOS_token, EOS_token)])
                target_string = ''.join([index_to_char[index.item()] for index in target if index.item() not in (SOS_token, EOS_token)])
                input_string = ''.join([index_to_char[index.item()] for index in input if index.item() not in (SOS_token, EOS_token)])
                
                print(f'Input: {input_string}, Target: {target_string}, Predicted: {predicted_string}')

        
        # Print overall evaluation results
        average_loss = total_loss / len(dataloader)
        val_loss_list.append(average_loss)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')

# Assuming all characters in the dataset + 'SOS' and 'EOS' tokens are included in char_to_index
input_size = len(char_to_index)
output_size = len(char_to_index)

encoder = Encoder(input_size=len(char_to_index), hidden_size=256).to(device)
# decoder without attention
#decoder = Decoder(hidden_size=256, output_size=len(char_to_index)).to(device)

# decoder with attention
decoder = AttnDecoder(hidden_size=256, output_size=len(char_to_index)).to(device)

# Set the learning rate for optimization
learning_rate = 0.011

# Initializing optimizers for both encoder and decoder with Stochastic Gradient Descent (SGD)
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

# Negative Log Likelihood Loss function for calculating loss
criterion = nn.NLLLoss()

# Set number of epochs for training
n_epochs = 100
train_loss_list = []
val_loss_list = []

# Training loop
for epoch in range(n_epochs):
    running_loss = 0.0
    for input_tensor, target_tensor in dataloader:

        # Uncomment below for English to French translation
        #input = input_tensor[0].to(device)
        #target = target_tensor[0].to(device)

        # Uncomment below for French to English translation
        # Uncomment below for English to French translation
        input = target_tensor[0].to(device)
        target = input_tensor[0].to(device)        

        # Perform a single training step and update total loss
        # Uncomment below to train without attention
        #loss = train(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        # Uncomment below to train with attention
        loss = train_w_attention(input, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        running_loss += loss
        
    train_loss_list.append(running_loss / len(dataloader))

    # Validation loop
    # Uncomment below to validate without attention
    #evaluate_and_show_examples(encoder, decoder, dataloader, criterion, val_loss_list)

    # Uncomment below to validate with attention
    evaluate_and_show_examples_attention(encoder, decoder, dataloader, criterion, val_loss_list)

print(train_loss_list[-1])
print(val_loss_list[-1])
plt.plot(train_loss_list, label='Training Loss')
plt.plot(val_loss_list, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('RNN_Question_1_Len10')
plt.show()