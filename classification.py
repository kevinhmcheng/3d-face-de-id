import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import scipy.io

class BU3DFEDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, train=True):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        if(datatype=='2D'):
            image = Image.open(self.img_dir + img_path)
            image = image.resize((224,224))
        elif(datatype=='Depth'):
            img_path = img_path.replace('2D.bmp','3D.bmp')
            image = Image.open(self.img_dir + img_path).convert('RGB')
        
        label1 = self.img_labels.iloc[idx, 1]
        label2 = self.img_labels.iloc[idx, 2]
        label3 = self.img_labels.iloc[idx, 3]
        label4 = self.img_labels.iloc[idx, 4]
        
        if self.transform:
            transform = []
            if self.train:
                transform.append(T.RandomHorizontalFlip())
            transform.append(T.ToTensor())
            transform.append(T.Normalize([0.5]*3, [0.5]*3))        
            transform = T.Compose(transform)
            image = transform(image)
            
        if self.target_transform:
            label1 = self.target_transform(label1)
            label2 = self.target_transform(label2)
            label3 = self.target_transform(label3)
            label4 = self.target_transform(label4)
        return image, label1, label2, label3, label4

    
def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y1, y2, y3, y4) in enumerate(dataloader):
        X, y1, y2, y3, y4 = X.to(device), y1.to(device), y2.to(device), y3.to(device), y4.to(device)

        # Compute prediction error
        pred = model(X)
        if task == 1:
            loss = loss_fn(pred, y1)
        elif task == 2:
            loss = loss_fn(pred, y2)
        elif task == 3:
            loss = loss_fn(pred, y3)
        elif task == 4:
            loss = loss_fn(pred, y4)
            
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 10 == 0:
        if batch+1 == len(dataloader):
            loss, current, total_it = loss.item(), batch+1, len(dataloader)
            print(f"loss: {loss:>7f}  [{current:>5d}/{total_it:>5d}]")
        
        
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y1, y2, y3, y4 in dataloader:
            X, y1, y2, y3, y4 = X.to(device), y1.to(device), y2.to(device), y3.to(device), y4.to(device)
            pred = model(X)
            if task == 1:
                actual = y1
            elif task == 2:
                actual = y2
            elif task == 3:
                actual = y3                
            elif task == 4:
                actual = y4
                
            test_loss += loss_fn(pred, actual).item()   
            correct += (pred.argmax(1) == actual).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
def test_all(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print(size)
    
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    count = 0;
    with torch.no_grad():
        for X, y1, y2, y3, y4 in dataloader:
            X, y1, y2, y3, y4 = X.to(device), y1.to(device), y2.to(device), y3.to(device), y4.to(device)
            pred = model(X)
            if task == 1:
                actual = y1
            elif task == 2:
                actual = y2
            elif task == 3:
                actual = y3                
            elif task == 4:
                actual = y4
                
            test_loss += loss_fn(pred, actual).item()   
            correct += (pred.argmax(1) == actual).type(torch.float).sum().item()
            if count == 0:
                Pred = pred
                Actual = actual
            else:
                Pred = torch.cat((Pred, pred), 0)
                Actual = torch.cat((Actual, actual), 0)
            count = count + 1
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    scipy.io.savemat(os.path.join(out_path, 'scores.mat'),{'Pred':Pred.cpu().numpy(),'Actual':Actual.cpu().numpy(),'Accuracy':correct,'Loss':test_loss})    

    
#MAIN
import sys
print('Start')
datatype = sys.argv[1] #2D, Depth
task = int(sys.argv[2]) #1,2,3,4
model_select = int(sys.argv[3]) #1,2,3,4


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


print(datatype + ', ' + 'Task: ' + str(task) + ', Model: ' + str(model_select))
if task == 1:
    folder_path = 'identity/'
elif task == 2:
    folder_path = 'expression/'
elif task == 3:
    folder_path = 'gender/'
elif task == 4:
    folder_path = 'ethnicity/'

train_label_filename = 'train_label.txt'
test_label_filename = 'test_label.txt'

data_path = 'Data/' + datatype + '/'                                        #*may require editing
out_path = 'task/'+ folder_path + datatype + '/model' + str(model_select) + '/'
if not os.path.exists(out_path):
     os.makedirs(out_path)

train_data = BU3DFEDataset(
    annotations_file = train_label_filename,
    img_dir = data_path,
    transform=True,
    train=True,
)
test_data = BU3DFEDataset(
    annotations_file = test_label_filename,
    img_dir = data_path,
    transform=True,
    train=False,
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(len(train_dataloader.dataset))
print(len(test_dataloader.dataset))

for X, y1, y2, y3, y4 in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y1: ", y1.shape, y1.dtype)
    print("Shape of y2: ", y2.shape, y2.dtype)
    print("Shape of y3: ", y3.shape, y3.dtype)
    print("Shape of y4: ", y4.shape, y4.dtype)
    break

# Display image and label.
train_features, train_label1, train_label2, train_label3, train_label4 = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Label1 batch shape: {train_label1.size()}")
print(f"Label2 batch shape: {train_label2.size()}")
print(f"Label3 batch shape: {train_label3.size()}")
print(f"Label4 batch shape: {train_label4.size()}")
img = train_features[0].squeeze().permute(1,2,0)
label1 = train_label1[0]
label2 = train_label2[0]
label3 = train_label3[0]
label4 = train_label4[0]

img = denorm(img)
#plt.imshow(img, cmap='gray')
#plt.show()
print(f"Label1: {label1}")
print(f"Label2: {label2}")
print(f"Label3: {label3}")
print(f"Label4: {label4}")


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

#Model Selection
pretrain = True
if model_select == 1:
    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=pretrain, transform_input=False)
    if task == 1:
        model.fc = nn.Linear(1024, 100)
    elif task == 2:
        model.fc = nn.Linear(1024, 6)
    elif task == 3:
        model.fc = nn.Linear(1024, 2)
    elif task == 4:
        model.fc = nn.Linear(1024, 6)

elif model_select == 2:
    model = models.resnet50(pretrained=pretrain)
    if task == 1:
        model.fc = nn.Linear(2048, 100)
    elif task == 2:
        model.fc = nn.Linear(2048, 6)
    elif task == 3:
        model.fc = nn.Linear(2048, 2)
    elif task == 4:
        model.fc = nn.Linear(2048, 6)
    
elif model_select == 3:
    model = models.vgg16(pretrained=pretrain)
    model.avgpool = nn.Identity()
    if datatype=='Depth':
        model.classifier[0] = nn.Linear(512*5*5, 4096) #2D: 224*224->512*7*7; Depth: 160*160->512*5*5
    if task == 1:
        model.classifier[6] = nn.Linear(4096, 100)
    elif task == 2:
        model.classifier[6] = nn.Linear(4096, 6)
    elif task == 3:
        model.classifier[6] = nn.Linear(4096, 2)
    elif task == 4:
        model.classifier[6] = nn.Linear(4096, 6)
    
elif model_select == 4:
    model = models.densenet121(pretrained=pretrain)
    if task == 1:
        model.classifier = nn.Linear(1024, 100)
    elif task == 2:
        model.classifier = nn.Linear(1024, 6)
    elif task == 3:
        model.classifier = nn.Linear(1024, 2)
    elif task == 4:
        model.classifier = nn.Linear(1024, 6)
#print(model)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

if model_select == 3:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
else:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

epochs = 50


for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
    print(f"Learning Rate {scheduler.get_last_lr()}\n-------------------------------")
    scheduler.step()
print("Done!")


torch.save(model.state_dict(), os.path.join(out_path, 'model.pth'))
print("Saved PyTorch Model State")

model.load_state_dict(torch.load(os.path.join(out_path, 'model.pth')))
test_all(test_dataloader, model, loss_fn)
