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
            if task == 1:                                                                                    #Switch
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

    scipy.io.savemat(os.path.join(out_path, 'scores.mat'),{'Pred':Pred.cpu().numpy(),'Actual':Actual.cpu().numpy(),'Accuracy':correct})

    
    
#MAIN
import sys
print('Start')
datatype = sys.argv[1] #2D, Depth
constraint = int(sys.argv[2]) #1,2,3,4
task = int(sys.argv[3]) #1,2,3,4
model_select = int(sys.argv[4]) #1,2,3,4


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


print( datatype+',Constraint:'+str(constraint)+',Task:'+str(task)+',Model:'+str(model_select) )
if task == 1:
    folder_path = 'identity/'
elif task == 2:
    folder_path = 'expression/'
elif task == 3:
    folder_path = 'gender/'
elif task == 4:
    folder_path = 'ethnicity/'

if constraint == 2:
    data_path = 'AE/constraint-'+datatype+'/expression/processed/'
    out_path = 'AE/constraint-'+datatype+'/expression/evaluation/'+folder_path+ '/model' + str(model_select) + '/'
elif constraint == 3:
    data_path = 'AE/constraint-'+datatype+'/gender/processed/'
    out_path = 'AE/constraint-'+datatype+'/gender/evaluation/'+folder_path+ '/model' + str(model_select) + '/'
elif constraint == 4:
    data_path = 'AE/constraint-'+datatype+'/ethnicity/processed/'
    out_path = 'AE/constraint-'+datatype+'/ethnicity/evaluation/'+folder_path+ '/model' + str(model_select) + '/'
elif constraint == 5:
    data_path = 'AE/constraint-'+datatype+'/cons/processed/'
    out_path = 'AE/constraint-'+datatype+'/cons/evaluation/'+folder_path+ '/model' + str(model_select) + '/'
elif constraint == 6:
    data_path = 'AE/constraint-'+datatype+'/de-id/processed/'
    out_path = 'AE/constraint-'+datatype+'/de-id/evaluation/'+folder_path+ '/model' + str(model_select) + '/'

elif constraint == 12:
    data_path = 'GAN/constraint-'+datatype+'/expression/processed/'
    out_path = 'GAN/constraint-'+datatype+'/expression/evaluation/'+folder_path+ '/model' + str(model_select) + '/'
elif constraint == 13:
    data_path = 'GAN/constraint-'+datatype+'/gender/processed/'
    out_path = 'GAN/constraint-'+datatype+'/gender/evaluation/'+folder_path+ '/model' + str(model_select) + '/'
elif constraint == 14:
    data_path = 'GAN/constraint-'+datatype+'/ethnicity/processed/'
    out_path = 'GAN/constraint-'+datatype+'/ethnicity/evaluation/'+folder_path+ '/model' + str(model_select) + '/'
elif constraint == 15:
    data_path = 'GAN/constraint-'+datatype+'/cons/processed/'
    out_path = 'GAN/constraint-'+datatype+'/cons/evaluation/'+folder_path+ '/model' + str(model_select) + '/'
elif constraint == 16:
    data_path = 'GAN/constraint-'+datatype+'/de-id/processed/'
    out_path = 'GAN/constraint-'+datatype+'/de-id/evaluation/'+folder_path+ '/model' + str(model_select) + '/'

if not os.path.exists(out_path):
    os.makedirs(out_path)    

test_label_filename = 'test_label.txt'

test_data = BU3DFEDataset(
annotations_file = test_label_filename,
img_dir = data_path,
transform=ToTensor(),
train=False,
)

batch_size = 64

# Create data loaders.
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
print(len(test_dataloader.dataset))

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

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
        model.classifier[0] = nn.Linear(512*5*5, 4096) #2D: 224*224->512*7*7; 3D: 160*160->512*5*5
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


model_path = 'task/'+ folder_path + datatype + '/model' + str(model_select) + '/' + 'model.pth'
model.load_state_dict(torch.load(model_path))

test_all(test_dataloader, model, loss_fn)
