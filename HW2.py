import torch
import torchvision
from torchvision import transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns

class AlexNet(nn.Module):
    def __init__(self, num_classes):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),

            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(192 * 2 * 2, 4096),
            nn.ReLU(),
            #Remove if you want no Dropout
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            #Remove if you want no Dropout
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        
        self.classifer = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            #Remove if you want no Dropout
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            #Remove if you want no Dropout
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifer(x)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class BasicBlock18(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.bn4(self.conv4(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet11(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet11, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3],stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        #Add dropout, comment out to remove. UNCOMMENT ABOVE IF YOU DO
        out = F.dropout(self.linear(out))
        return out
    
class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3],stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #out = self.linear(out)
        #Add dropout, comment out to remove. UNCOMMENT ABOVE IF YOU DO
        out = F.dropout(self.linear(out))
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device)
    torch.cuda.empty_cache()

    #change number of classes below to change between CIFAR 10 and CIFAR 100
    num_classes = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #If num_classes is 10 we are using CIFAR 10
    if num_classes == 10:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
        print("Using CIFAR10 dataset")
        
    #If num_classes is 100 we are using CIFAR 100
    elif num_classes == 100:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False, num_workers=2)
        print("Using CIFAR100 dataset")
    
    #use the code below for AlexNet
    #model = AlexNet(num_classes=num_classes)
    #Use the code below for VGGNet
    #model = VGG11(num_classes=num_classes)
    #Use the code below for ResNet11
    #model = ResNet11(BasicBlock, [1,1,1,1], num_classes=num_classes)
    #Use the code below for ResNet18
    model = ResNet18(BasicBlock, [2,2,2,2], num_classes=num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = .01, momentum=.9)

    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    for epoch in range(40):  # loop over the dataset multiple times
        running_loss = 0.0
        model.train()  # Set the model to training mode
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss_list.append(running_loss / len(trainloader))

        # Validation loop
        running_loss = 0.0
        correct = 0
        total = 0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss_list.append(running_loss / len(testloader))
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
    #Uncomment below if training AlexNet Cifar10
    #plt.savefig('AlexNet_training_validation_loss_Cifar10.png')
    #Uncomment below if training AlexNet Cifar10
    #plt.savefig('AlexNet_training_validation_loss_Cifar100.png')
    #Uncomment below if training VGGNet Cifar10
    #plt.savefig('VGGNet_training_validation_loss_Cifar10.png')
    #Uncomment below if training VGGNet Cifar100
    #plt.savefig('VGGNet_training_validation_loss_Cifar100.png')
    #Uncomment below if training ResNet11 Cifar10
    #plt.savefig('ResNet11_training_validation_loss_Cifar10.png')
    #Uncomment below if training ResNet11 Cifar100
    #plt.savefig('ResNet11_training_validation_loss_Cifar100.png')
    #Uncomment below if training ResNet18 Cifar10
    #plt.savefig('ResNet18_training_validation_loss_Cifar10.png')
    #Uncomment below if training ResNet18 Cifar100
    plt.savefig('ResNet18_training_validation_loss_Cifar100.png')
    plt.show()