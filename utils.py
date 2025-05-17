import math
import random
import os
import numpy as np
import h5py
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread, imshow
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import datasets

##########################################################

def load_data (path):
    train_path = os.path.join(path, 'train_signs.h5')
    train_dataset = h5py.File(train_path, "r")
    train_set_x = np.array(train_dataset["train_set_x"][:]) # train set features
    train_set_y= np.array(train_dataset["train_set_y"][:]) # train set labels

    test_path = os.path.join(path, 'test_signs.h5')
    test_dataset = h5py.File(test_path, "r")
    test_set_x= np.array(test_dataset["test_set_x"][:]) # test set features
    test_set_y = np.array(test_dataset["test_set_y"][:]) # test set labels

    classes = np.array(test_dataset["list_classes"][:]) # list of classes

    return train_set_x, train_set_y, test_set_x, test_set_y
    

def resize_tensor_images(tensor, size=(244, 244)):
    return F.interpolate(tensor, size=size, mode='bilinear', align_corners=False)


def train_eval(model, train_loader, val_loader, loss_func, optimizer, epochs, one_hot_enc = True, device='cpu'):

    model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []

    best_val_acc = 0.0
    best_model_wts = None
    
    for epoch in tqdm(range(epochs), desc="Epochs"):

        # Training
        model.train()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
    
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  
            
            outputs = model(inputs)
            loss = loss_func(outputs, labels.float()) if one_hot_enc else loss_func(outputs, labels)
            
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)  # predicted class (index of max logit)
                if one_hot_enc:
                    _, true_labels = torch.max(labels, 1)
                    correct_preds += (preds == true_labels).sum().item()
                else:
                    correct_preds += (preds == labels).sum().item()
                total_preds += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        accuracy_train = (correct_preds / total_preds) * 100      
        train_losses.append(avg_train_loss)
        train_accuracies.append(accuracy_train)
                
        # Validation
        model.eval()
        val_loss, val_correct_preds, val_total_preds = 0.0, 0, 0
        with torch.no_grad():  
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_outputs = model(val_inputs)
                
                val_loss_batch = loss_func(val_outputs, val_labels.float()) if one_hot_enc else loss_func(val_outputs, val_labels)  
                val_loss += val_loss_batch.item()
                
                _, val_preds = torch.max(val_outputs, 1)

                if one_hot_enc:
                    _, true_val_labels = torch.max(val_labels, 1)
                    val_correct_preds += (val_preds == true_val_labels).sum().item()
                else:
                    val_correct_preds += (val_preds == val_labels).sum().item()
                val_total_preds += val_labels.size(0)
    
        avg_loss_val = val_loss / len(val_loader)
        accuracy_val = (val_correct_preds / val_total_preds) * 100
        val_losses.append(avg_loss_val)
        val_accuracies.append(accuracy_val)
    
        # print(f"Epoch {epoch+1:>3}/{epochs:<3} - "
        #       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {accuracy_train:.2f}%, "
        #       f"Val Loss: {avg_loss_val:.4f}, Val Acc: {accuracy_val:.2f}%")

        if accuracy_val > best_val_acc:
            best_val_acc = accuracy_val
            best_model_wts = model.state_dict()

    return best_model_wts, best_val_acc, train_losses, val_losses, train_accuracies, val_accuracies
    

def plot_loss_acc (data, title):
    train_loss, validation_loss, train_accuracy, validation_accuracy = data
    plt.clf()
    
    fig, axs = plt.subplots(2,1, figsize=(12, 6)) 

    # Plot loss
    axs[0].plot(train_loss, label='Train Loss')
    axs[0].plot(validation_loss, label='Validation Loss')
    # axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].grid(True)

    # Plot accuracy
    axs[1].plot(train_accuracy, label='Train Accuracy')
    axs[1].plot(validation_accuracy, label='Validation Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    # axs[1].set_title(title)
    axs[1].legend()
    axs[1].grid(True)

    # Save plot
    output_dir = os.path.join(os.getcwd(), 'loss_acc_figs')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{title}_loss_acc.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    gc.collect()

