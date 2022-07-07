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
        return image, label1, label2, label3, label4, img_path


def denorm(x):
        """Convert the range from [-1, 1] to [0, 1]."""
        out = (x + 1) / 2
        return out.clamp_(0, 1)
    
    
#MAIN
import sys
print('Start')
datatype = sys.argv[1] #2D, Depth
constraint = int(sys.argv[2]) #0,1,2,3,4,5,6


torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"


print( datatype + ', ' + 'Constraint: ' + str(constraint) )
if constraint == 0:
    folder_path = 'pretrain/'
elif constraint == 1:
    folder_path = 'identity/'
elif constraint == 2:
    folder_path = 'expression/'
elif constraint == 3:
    folder_path = 'gender/'
elif constraint == 4:
    folder_path = 'ethnicity/'
elif constraint == 5:
    folder_path = 'cons/'
elif constraint == 6:
    folder_path = 'de-id/'
    if datatype =='2D':
        para = 3
    else:
        para = 5

dir_to_base = 'AE/'                                                  #*may require editing
train_label_filename = dir_to_base + '../train_label.txt'                                 
test_label_filename = dir_to_base + '../test_label.txt'

data_path = dir_to_base + '../Data/' + datatype + '/'                #*may require editing
out_path = dir_to_base + 'constraint-'+datatype+'/'+ folder_path 
if not os.path.exists(out_path):
    os.makedirs(out_path)

training_data = BU3DFEDataset(
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


batch_size = 16

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

print(len(train_dataloader.dataset))
print(len(test_dataloader.dataset))

for X, y1, y2, y3, y4, X_path in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y1: ", y1.shape, y1.dtype)
    print("Shape of y2: ", y2.shape, y2.dtype)
    print("Shape of y3: ", y3.shape, y3.dtype)
    print("Shape of y4: ", y4.shape, y4.dtype)
    break


# Display image and label.
train_features, train_label1, train_label2, train_label3, train_label4, train_img_path = next(iter(train_dataloader))
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
#plt.imshow(img)
#plt.show()
print(f"Label1: {label1}")
print(f"Label2: {label2}")
print(f"Label3: {label3}")
print(f"Label4: {label4}")


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
torch.cuda.empty_cache()
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

    
## AE Model Integration
from convae import AutoEncoder
cae = AutoEncoder()
if datatype=='Depth':                                                          #Enforce 1 channel output
    cae.autoencoder[12] = nn.Conv2d(128, 1, kernel_size=(1, 1), stride=(1, 1))
if not constraint==0:
    cae.load_state_dict(torch.load(dir_to_base + 'constraint-'+datatype+'/'+ 'pretrain/'+'model.pth'))
cae.to(device)


## Classification Models
pretrain = True
if constraint == 1 or constraint == 2 or constraint == 3 or constraint == 4:
    if constraint == 1:
        if datatype=='2D':
            model = models.resnet50(pretrained=pretrain) #2D, task 1, model 2
            model.fc = nn.Linear(2048, 100)
            model.load_state_dict(torch.load(dir_to_base + '../task/'+ folder_path + datatype + '/model' + str(2) + '/' +'model.pth'))
        else:
            model = models.densenet121(pretrained=pretrain) #Depth, task1, model 4
            model.classifier = nn.Linear(1024, 100)
            model.load_state_dict(torch.load(dir_to_base + '../task/'+ folder_path + datatype + '/model' + str(4) + '/' +'model.pth'))
    elif constraint == 2:
        if datatype=='2D':
            model = models.resnet50(pretrained=pretrain) #2D, task2, model 2
            model.fc = nn.Linear(2048, 6)
            model.load_state_dict(torch.load(dir_to_base + '../task/'+ folder_path + datatype + '/model' + str(2) + '/' +'model.pth'))
        else:
            model = models.densenet121(pretrained=pretrain) #Depth, task 2, model 4
            model.classifier = nn.Linear(1024, 6)
            model.load_state_dict(torch.load(dir_to_base + '../task/'+ folder_path + datatype + '/model' + str(4) + '/' +'model.pth'))
    elif constraint == 3:
        model = models.resnet50(pretrained=pretrain) #2D/Depth, task3, model 2
        model.fc = nn.Linear(2048, 2)
        model.load_state_dict(torch.load(dir_to_base + '../task/'+ folder_path + datatype + '/model' + str(2) + '/' +'model.pth'))
    elif constraint == 4:
        model = models.resnet50(pretrained=pretrain) #2D/Depth, task4, model 2
        model.fc = nn.Linear(2048, 6)
        model.load_state_dict(torch.load(dir_to_base + '../task/'+ folder_path + datatype + '/model' + str(2) + '/' +'model.pth'))
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)    
elif constraint == 5 or constraint == 6:
    if datatype=='2D':
        model2 = models.resnet50(pretrained=pretrain) #2D, task2, model 2
        model2.fc = nn.Linear(2048, 6)
        model2.load_state_dict(torch.load(dir_to_base + '../task/'+ 'expression/' + datatype + '/model' + str(2) + '/' +'model.pth'))
    else:
        model2 = models.densenet121(pretrained=pretrain) #Depth, task 2, model 4
        model2.classifier = nn.Linear(1024, 6)
        model2.load_state_dict(torch.load(dir_to_base + '../task/'+ 'expression/' + datatype + '/model' + str(4) + '/' +'model.pth'))
    for param in model2.parameters():
        param.requires_grad = False
    model2.eval()
    model2.to(device)

    model3 = models.resnet50(pretrained=pretrain) #2D, task3, model 2
    model3.fc = nn.Linear(2048, 2)
    model3.load_state_dict(torch.load(dir_to_base + '../task/'+ 'gender/' + datatype + '/model' + str(2) + '/' +'model.pth'))
    for param in model3.parameters():
        param.requires_grad = False
    model3.eval()
    model3.to(device)

    model4 = models.resnet50(pretrained=pretrain) #2D, task4, model 2
    model4.fc = nn.Linear(2048, 6)
    model4.load_state_dict(torch.load(dir_to_base + '../task/'+ 'ethnicity/' + datatype + '/model' + str(2) + '/' +'model.pth'))
    for param in model4.parameters():
        param.requires_grad = False
    model4.eval()
    model4.to(device)
if constraint == 6:
    if datatype=='2D':
        model1 = models.resnet50(pretrained=pretrain) #2D, task 1, model 2
        model1.fc = nn.Linear(2048, 100)
        model1.load_state_dict(torch.load(dir_to_base + '../task/'+ 'identity/' + datatype + '/model' + str(2) + '/' +'model.pth'))
    else:
        model1 = models.densenet121(pretrained=pretrain) #Depth, task1, model 4
        model1.classifier = nn.Linear(1024, 100)
        model1.load_state_dict(torch.load(dir_to_base + '../task/'+ 'identity/' + datatype + '/model' + str(4) + '/' +'model.pth'))
    for param in model1.parameters():
        param.requires_grad = False
    model1.eval()
    model1.to(device)

optimizer = torch.optim.Adam(cae.parameters(), lr=0.0002) #0.0002
loss_fn = nn.CrossEntropyLoss()


if constraint == 0:
    epoch_start = 0
    num_epochs = 5
else:
    epoch_start = 5
    num_epochs = 20


for epoch in range(epoch_start, num_epochs):
    cae.train()
    for i, (X, y1, y2, y3, y4, X_path) in enumerate(train_dataloader):
        X, y1, y2, y3, y4 = X.to(device), y1.to(device), y2.to(device), y3.to(device), y4.to(device)
        X_rec = cae(X)
        if datatype=='Depth':                                              #Enforce 1 channel output 
            X_rec = X_rec.repeat(1,3,1,1)
        loss_rec = torch.mean(torch.abs(X - X_rec))
        
        

        if constraint == 0:
            loss_constraint = 0
        elif constraint == 1:
            loss_constraint = loss_fn(model(X_rec), y1)
        elif constraint == 2:
            loss_constraint = loss_fn(model(X_rec), y2)
        elif constraint == 3:
            loss_constraint = loss_fn(model(X_rec), y3)
        elif constraint == 4:
            loss_constraint = loss_fn(model(X_rec), y4)
        elif constraint == 5:
            loss_constraint = loss_fn(model2(X_rec),y2)+loss_fn(model3(X_rec),y3)+loss_fn(model4(X_rec),y4)
        elif constraint == 6:
            loss_constraint_att = loss_fn(model2(X_rec), y2)+loss_fn(model3(X_rec),y3)+loss_fn(model4(X_rec),y4)
            loss_constraint_id = loss_fn(model1(X_rec), y1)
            loss_constraint = -para/10*loss_constraint_id+loss_constraint_att
        
        loss = loss_rec + loss_constraint
        

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 20 == 0:
            print ('Epoch [{}/{}], Iter [{}/{}] Loss: {:.3f} {:.3f} {:.3f}'.format(
                    epoch+1, num_epochs, i+1, len(train_dataloader), 
                    loss_rec, loss_constraint, loss))

torch.save(cae.state_dict(), os.path.join(out_path, 'model.pth'))
print("Saved PyTorch Model State") 


## Image Generation
cae.eval()

from torchvision.utils import save_image
with torch.no_grad():
    for i, (X, y1, y2, y3, y4, X_path) in enumerate(test_dataloader):
        X, y1, y2, y3, y4 = X.to(device), y1.to(device), y2.to(device), y3.to(device), y4.to(device)
        print ('Iter [{}/{}] '.format(i+1, len(test_dataloader)))

        rec = cae(X)
        rec = denorm(rec)

        for f,r in zip(X_path, rec):
            #print(r.shape)
            f_a = f[0:6]
            if not os.path.exists(out_path+'processed/'+f_a):
                os.makedirs(out_path+'processed/'+f_a)
            save_image(r, out_path+'processed/'+f)

            
## Visual Inspection
X, y1, y2, y3, y4, X_path = next(iter(test_dataloader))

img = X[0].squeeze().permute(1,2,0)
img = denorm(img)
#plt.imshow(img)
#plt.show()

rec = cae(X.to(device)).cpu().detach()
if datatype=='Depth':
    img = rec[0].squeeze()
else:
    img = rec[0].squeeze().permute(1,2,0)
img = denorm(img)
#plt.imshow(img, cmap='gray')
#plt.show()

print(torch.min(X), torch.max(X))
print(torch.min(rec), torch.max(rec))
