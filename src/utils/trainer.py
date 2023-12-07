import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from .average_meter import AverageMeter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    
    writer = SummaryWriter()
    
    def __init__(self,model, batch_size, learning_rate, train_set, test_set, device = "cpu", classes = None) -> None:
        self.model = model
        self.device = device
        self.classes = classes
        self.train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4)
        self.test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=4)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-8)
        
    
    def train(self, epoch = 1):
        # meter
        loss_meter = AverageMeter()
        # switch to train mode
        self.model.train()
        tk = tqdm(
            self.train_loader,
            total=len(self.train_loader),
            desc='Training',
            unit='frames',
            leave=False,
        )
        for data in tk:
            # fetch the data
            frame, label = data[0], data[1]
            # after fetching the data, transfer the model to the 
            # required device, in this example the device is gpu
            # transfer to gpu can also be done by 
            frame, label = frame.to(self.device), label.to(self.device)

            # print(len(frame))
            # print(frame.shape)
            # compute the forward pass
            output = self.model(frame)

            # print(label)
            logits = output.logits
            pred = logits.argmax(axis = 1)
            # print(pred)

            # compute the loss function
            loss_this = F.cross_entropy(logits, label.long())
            self.writer.add_scalar("Loss/train", loss_this, epoch)
            # initialize the optimizer
            self.optimizer.zero_grad()
            # compute the backward pass
            loss_this.backward()
            # update the parameters
            self.optimizer.step()
            # update the loss meter
            loss_meter.update(loss_this.item(), label.shape[0])
            tk.set_postfix({"loss": loss_meter.avg})
        print('Train: Average loss: {:.8f}\n'.format(loss_meter.avg))
        self.writer.flush()

    ##define test function
    def test(self):
        # meters
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        # switch to test mode
        correct = 0
        self.model.eval()
        tk = tqdm(
            self.test_loader,
            total=len(self.test_loader),
            desc='Test',
            unit='frames',
            leave=False,
        )
        preds = []
        trues = []
        for data in tk:
            # fetch the data
            frame, label = data[0], data[1]
            # after fetching the data transfer the model to the 
            # required device, in this example the device is gpu
            # transfer to gpu can also be done by 
            frame, label = frame.to(self.device), label.to(self.device)
            # since we dont need to backpropagate loss in testing,
            # we dont keep the gradient
            with torch.no_grad():
                output = self.model(frame)
            logits = output.logits
            pred = logits.argmax(axis = 1)
            
            # compute the loss function just for checking
            loss_this = F.cross_entropy(logits, label.long())
            # get the index of the max log-probability
            # pred = output.argmax(dim=1, keepdim=True)
            # check which of the predictions are correct
            correct_this = pred.eq(label).sum().item()
            # accumulate the correct ones
            correct += correct_this
            # compute accuracy
            acc_this = correct_this / label.shape[0] * 100.0
            # update the loss and accuracy meter 
            acc_meter.update(acc_this, label.shape[0])
            loss_meter.update(loss_this.item(), label.shape[0])
            preds.append(logits.cpu())
            trues.append(label.cpu())
        print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            loss_meter.avg, correct, len(self.test_loader.dataset), acc_meter.avg))
        
        self.plot_results( np.concatenate(preds).argmax(axis=1), np.concatenate(trues))
        self.calculateTopKAccuracies( np.concatenate(preds), np.concatenate(trues))
        
    def plot_results(self, pred, true):
        print(classification_report(true, pred, target_names=self.classes))
        cm = confusion_matrix(true, pred)
        hm = sns.heatmap(cm, annot=True, linewidths=0.1, xticklabels=self.classes, yticklabels=self.classes)
        hm.set(title = "Confusion Matrix")   
        plt.xlabel("Predicted Value")
        plt.ylabel("Actual Value")
        plt.figure(figsize=(9, 7))
        plt.show()
        
    def calculateTopKAccuracies(self, preds, labels):        
        score_1=0
        score_5=0
        for i in range(len(preds)):
            if np.argmax(preds[i]) == labels[i]:
                score_1+=1
            if labels[i] in np.argsort(preds)[::-1][:5]:
                score_5+=1
                
        rank1 = ((score_1/float(len(labels)))*100)
        rank5 = (score_5/float(len(labels)))*100
        print("Top 1 Accuracy: %.2f%%" % rank1)
        print("Top 5 Accuracy: %.2f%%" % rank5)