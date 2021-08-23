import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#=========================================>
#   < Preprocessing the Images >
#=========================================>

# transform = transforms.Compose([transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         transform=transform)
#
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        transform=transform)
#
# testloader = torch.utils.data.DataLoader(testset, batch_size=8,
#                                          shuffle=False, num_workers=2)
import vgg_dataloader
trainloader = vgg_dataloader.train_loader
testloader = vgg_dataloader.test_loader

#=========================================>
#   < Building the Network >
#=========================================>

class VGG_mini(nn.Module):
    
    def __init__(self): 
        super(VGG_mini, self).__init__()
        
        # Maxpool 2x2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv layers with batch norm
        
        self.conv1 = nn.Conv2d(20, 64, 3, padding = 1)
        self.norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding = 1)
        self.norm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding = 1)
        self.norm3 = nn.BatchNorm2d(256)
        # fully connected layer with batch norm
        # filter * width * height
        self.fc1 = nn.Linear(256 * 32 * 32, 64)
        self.norm4 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 2)
        self.norm5 = nn.BatchNorm1d(2)

        
        

    def forward(self, x):       
        
        out = F.elu(self.norm1(self.conv1(x)))
        out = self.pool(out)
        out = F.elu(self.norm2(self.conv2(out)))
        out = self.pool(out)
        out = F.elu(self.norm3(self.conv3(out)))
        out = self.pool(out)

        out = out.view(-1, 256 * 32 * 32)

        out = F.elu(self.norm4(self.fc1(out)))
        out = F.elu(self.norm5(self.fc2(out)))

        return out
    
    
#=========================================>
#   < Training the Network >
#=========================================>

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = VGG_mini().to(device)
print(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)

for epoch in range(100):

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
       
        inputs, labels = data  
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        outputs = net(inputs)    
        softmax = nn.Softmax(dim=1)
        prob = softmax(outputs)
        loss = loss_fn(prob, labels)
        loss.backward()
        
        optimizer.step()

        running_loss += loss.item()
        
    print('epoch : %d , loss = %.3f' % (epoch+1, running_loss / 6250))
    
    # if (running_loss / 6250) < 0.05:
    #     break

print('\n < Finished Training > \n')


#=========================================>
#   < Testing the Netwok >
#=========================================>

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        softmax == nn.Softmax(dim=1)
        outputs = softmax(outputs)
        _, predicted = torch.max(outputs.data, 1)

        labels = ((labels == 1).nonzero(as_tuple=False)[:,1])
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy : %d %%' % (
    100 * correct / total)
      )