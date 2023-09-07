import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import device

class Tnet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(k,64,1)
      self.conv2 = nn.Conv1d(64,128,1)
      self.conv3 = nn.Conv1d(128,1024,1)
      self.fc1 = nn.Linear(1024,512)
      self.fc2 = nn.Linear(512,256)
      self.fc3 = nn.Linear(256,k*k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)
       

   def forward(self, input):
      # input.shape == (bs,n,3)
      bs = input.size(0)
      xb = F.relu(self.bn1(self.conv1(input)))
      xb = F.relu(self.bn2(self.conv2(xb)))
      xb = F.relu(self.bn3(self.conv3(xb)))
      pool = nn.MaxPool1d(xb.size(-1))(xb)
      flat = nn.Flatten(1)(pool)
      xb = F.relu(self.bn4(self.fc1(flat)))
      xb = F.relu(self.bn5(self.fc2(xb)))
      
      #initialize as identity
      init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1).to(device=device.device)
      matrix = self.fc3(xb).view(-1,self.k,self.k) + init
      return matrix
   
class Transform(nn.Module):
   def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)

        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
       

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
   def forward(self, input):
        matrix3x3 = self.input_transform(input)
        # batch matrix multiplication
        xb = torch.bmm(torch.transpose(input,1,2), matrix3x3).transpose(1,2)

        xb = F.relu(self.bn1(self.conv1(xb)))

        matrix64x64 = self.feature_transform(xb)
        xb = torch.bmm(torch.transpose(xb,1,2), matrix64x64).transpose(1,2)

        xb = F.relu(self.bn2(self.conv2(xb)))
        xb = self.bn3(self.conv3(xb))
        xb = nn.MaxPool1d(xb.size(-1))(xb)
        output = nn.Flatten(1)(xb)
        return output, matrix3x3, matrix64x64

def PointNet_loss(outputs, labels, m3x3, m64x64, alpha = 0.0001):
    criterion = torch.nn.NLLLoss()
    bs=outputs.size(0)
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs,1,1).to(device=device.device)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs,1,1).to(device=device.device)
    diff3x3 = id3x3-torch.bmm(m3x3,m3x3.transpose(1,2))
    diff64x64 = id64x64-torch.bmm(m64x64,m64x64.transpose(1,2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3)+torch.norm(diff64x64)) / float(bs)

class PointNet(nn.Module):
    def __init__(self, classes = 10):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)
        

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        xb, matrix3x3, matrix64x64 = self.transform(input)
        xb = F.relu(self.bn1(self.fc1(xb)))
        xb = F.relu(self.bn2(self.dropout(self.fc2(xb))))
        output = self.fc3(xb)
        return self.logsoftmax(output), matrix3x3, matrix64x64
    
    def train_model(self,
              optimizer,
              train_loader,
              val_loader=None,
              epochs=15,
              save=True):
        
        for epoch in range(epochs): 
            self.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data['pointcloud'].type(torch.FloatTensor).to(device.device), data['category'].to(device.device)
                optimizer.zero_grad()
                outputs, m3x3, m64x64 = self(inputs.transpose(1,2))
                loss = PointNet_loss(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 10 == 9:    # print every 10 mini-batches
                        print('[Epoch: %d, Batch: %4d / %4d], loss: %.3f' %
                            (epoch + 1, i + 1, len(train_loader), running_loss / 10))
                        running_loss = 0.0

            self.eval()
            correct = total = 0

            # validation
            if val_loader:
                with torch.no_grad():
                    for data in val_loader:
                        inputs, labels = data['pointcloud'].type(torch.FloatTensor).to(device.device), data['category'].to(device.device)
                        outputs, __, __ = self(inputs.transpose(1,2))
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                val_acc = 100. * correct / total
                print('Valid accuracy: %d %%' % val_acc)

            # save the model
            if save:
                torch.save(self.state_dict(), "./10_3D-Images/PointNet/saved_models/save_"+str(epoch)+".pth")
    
    def test_model(self, test_loader):
        self.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                print('Batch [%4d / %4d]' % (i+1, len(test_loader)))
                        
                inputs, labels = data['pointcloud'].type(torch.FloatTensor).to(device.device), data['category'].to(device.device)
                outputs, _, _ = self(inputs.transpose(1,2))
                _, preds = torch.max(outputs.data, 1)
                all_preds += list(preds.cpu().numpy())
                all_labels += list(labels.cpu().numpy())
        return all_labels, all_preds