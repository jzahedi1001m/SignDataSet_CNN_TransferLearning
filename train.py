import os
import math
import random
import argparse
import numpy as np
import h5py
import gc
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models

from utils import *
from models import *

np.random.seed(1234)
################################################################################

def prepare_data(path, device, batch_size=64, num_classes=6):
    """
    Loads, normalizes, reshapes, and prepares training and validation data loaders.
    """
    train_x, train_y, test_x, test_y = load_data(path)

    # Reshape label vectors to 1D (batch dimension)
    Y_train = train_y.reshape((1, train_y.shape[0]))
    Y_test = test_y.reshape((1, test_y.shape[0]))

    # Normalize image pixel values to [0, 1]
    X_train = train_x / 255.
    X_test = test_x / 255.

    # Convert to torch tensors and permute to (batch, channels, height, width)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    Y_train_tensor0 = torch.tensor(Y_train, dtype=torch.long, device=device).squeeze()
    Y_train_tensor = F.one_hot(Y_train_tensor0, num_classes=num_classes)

    X_val_tensor = torch.tensor(X_test, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    Y_val_tensor0 = torch.tensor(Y_test, dtype=torch.long, device=device).squeeze()
    Y_val_tensor = F.one_hot(Y_val_tensor0, num_classes=num_classes)

    train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, Y_val_tensor),
                            batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train_tensor, Y_train_tensor0, X_val_tensor, Y_val_tensor0


def train_sign_cnn(device, train_loader, val_loader):

    print("""
    Train simple SingCnn for sign dataset: conv -> relu -> pool -> conv -> relu -> maxpool -> fc
    """)

    num_classes = 6
    model = SignCnn(num_classes).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    results = train_val(model, train_loader, val_loader, loss_fn, optimizer,
                         epochs=200, one_hot_enc=True, device=device)

    best_weights, best_acc = results[0], results[1]
    plot_loss_acc(results[2:], title='SignCnn')

    return best_weights


def train_sign_resnet50(device, train_loader, val_loader):

    print("""
    Train SignResnet50 for sign dataset
    """)
    
    model = SignResnet50(input_channels=3, num_classes=6).to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # Use smaller batch size for training and validation loaders
    train_loader = DataLoader(train_loader.dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_loader.dataset, batch_size=32, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00015)

    results = train_val(model, train_loader, val_loader, loss_fn, optimizer,
                         epochs=15, one_hot_enc=True, device=device)

    best_weights, best_acc = results[0], results[1]
    plot_loss_acc(results[2:], title='SignResNet50')

    return best_weights


def train_resnet18_transfer(device, X_train_tensor, Y_train_tensor0, X_val_tensor, Y_val_tensor0):

    print("""
    Transfer learning with pretrained ResNet18 for sign dataset::
    Freeze all layers except last, resize inputs, and train only the last layer.
    """)
    model = models.resnet18(pretrained=True)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer to match num_classes
    num_classes = 6
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.fc.parameters())
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # Resize input tensors to 244x244
    X_train_resized = resize_tensor_images(X_train_tensor, size=(244, 244))
    X_val_resized = resize_tensor_images(X_val_tensor, size=(244, 244))

    train_loader = DataLoader(TensorDataset(X_train_resized, Y_train_tensor0),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_resized, Y_val_tensor0),
                            batch_size=32, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # Only optimize last layer

    results = train_val(model, train_loader, val_loader, loss_fn, optimizer,
                         epochs=50, one_hot_enc=False, device=device)

    best_weights, best_acc = results[0], results[1]
    plot_loss_acc(results[2:], title='ResNet18_TransferLearning_LastLayer')

    return best_weights


def train_mobilenet2_transfer(device, X_train_tensor, Y_train_tensor0, X_val_tensor, Y_val_tensor0):

    print("""
    Transfer learning with pretrained MobileNetV2:
    MobileNetV2-based sign detector with additional custom layers for classification.
    Freeze first 180 layers and train remaining layers
    """)
    num_classes = 6
    model = SignMobileNetV2(num_classes).to(device)

    # Freeze weights for first 180 layers
    for idx, layer in enumerate(model.modules()):
        requires_grad = idx >= 180
        for param in layer.parameters():
            param.requires_grad = requires_grad

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}, Trainable parameters: {trainable_params}")

    # Resize images to 160x160
    X_train_resized = resize_tensor_images(X_train_tensor, size=(160, 160))
    X_val_resized = resize_tensor_images(X_val_tensor, size=(160, 160))

    train_loader = DataLoader(TensorDataset(X_train_resized, Y_train_tensor0),
                              batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_resized, Y_val_tensor0),
                            batch_size=32, shuffle=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    results = train_val(model, train_loader, val_loader, loss_fn, optimizer,
                         epochs=10, one_hot_enc=False, device=device)

    best_weights, best_acc = results[0], results[1]
    plot_loss_acc(results[2:], title='MobileNetV2_TransferLearning_LastLayers')

    return best_weights

def train(train_loader, val_loader, X_train_tensor, Y_train_tensor0, X_val_tensor, Y_val_tensor0, device):

    # Train and Save the best model
    output_dir = os.path.join(os.getcwd(), 'trained_models')
    os.makedirs(output_dir, exist_ok=True)

    model_trainers = {
        'signcnn': lambda: train_sign_cnn(device, train_loader, val_loader),
        'signresnet50': lambda: train_sign_resnet50(device, train_loader, val_loader),
        'TL_resnet18': lambda: train_resnet18_transfer(device, X_train_tensor, Y_train_tensor0, X_val_tensor, Y_val_tensor0),
        'TL_mobilenetv2': lambda: train_mobilenet2_transfer(device, X_train_tensor, Y_train_tensor0, X_val_tensor, Y_val_tensor0)
    }
    
    if args.model in model_trainers:
        weights = model_trainers[args.model]()  
        save_path = os.path.join(output_dir, f"{args.model}.pth")
        torch.save(weights, save_path)
        print(f"Saved weights for {args.model} to {save_path}")
        del weights
        torch.cuda.empty_cache()
        gc.collect()
    else:
        for model_name, trainer in model_trainers.items():
            weights = trainer()
            save_path = os.path.join(output_dir, f"{model_name}.pth")
            torch.save(weights, save_path)
            print(f"Saved weights for {model_name} to {save_path}")
            del weights
            torch.cuda.empty_cache()
            gc.collect()

def get_model(name, val_loader, X_val_resized160, X_val_resized244, Y_val_tensor0):
    
    if name == 'signresnet50':
        model = SignResnet50(3, 6)
        val_loader = DataLoader(val_loader.dataset, batch_size=32, shuffle=False)
    elif name == 'TL_resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 6)
        val_loader = DataLoader(TensorDataset(X_val_resized244, Y_val_tensor0),
                        batch_size=32, shuffle=False)
    elif name == 'TL_mobilenetv2':
        model = SignMobileNetV2(6)
        val_loader = DataLoader(TensorDataset(X_val_resized160, Y_val_tensor0),
                        batch_size=32, shuffle=False)
    else:
        model = SignCnn(6)

    return model, val_loader
    

def eval(val_loader, X_val_tensor, Y_val_tensor0, path, device):

    
    X_val_resized160 = resize_tensor_images(X_val_tensor, size=(160, 160))
    X_val_resized244 = resize_tensor_images(X_val_tensor, size=(244, 244))

    model_names = ['signcnn', 'signresnet50','TL_resnet18','TL_mobilenetv2']
    
    if args.model in model_names:
        model_path = os.path.join(path, f"{args.model}.pth")
        if not os.path.exists(model_path):
            print(f"Checkpoint for {model_name} not found at {model_path}. Please train the model first.")
            exit()
        print(f"Evaluation: Trained {args.model}")
        model, val_loader = get_model(args.model, val_loader, X_val_resized160, X_val_resized244, Y_val_tensor0)
        model.load_state_dict(torch.load(model_path))     
        res = model_eval(model, args.model, val_loader, device)
        acc = res[0]
        plot_pred_samples(res[1], res[2], title = args.model)
        
    else:
        acc = {}
        for model_name in model_names:
            model_path = os.path.join(path, f"{model_name}.pth")
            if not os.path.exists(model_path):
                print(f"Checkpoint for {model_name} not found at {model_path}. Please train the model first.")
                continue
            print(f"Evaluation: Trained {model_name}")
            model, val_loader = get_model(model_name, val_loader, X_val_resized160, X_val_resized244, Y_val_tensor0)
            model.load_state_dict(torch.load(model_path))     
            res = model_eval(model, model_name, val_loader, device)
            acc[model_name] = res[0]
            plot_pred_samples(res[1], res[2], model_name)
            
    print(f"accuracy is {acc}") 


def main():

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    data_path = os.path.join(os.getcwd(), 'datasets')
    train_loader, val_loader, X_train_tensor, Y_train_tensor0, X_val_tensor, Y_val_tensor0 = prepare_data(
        data_path, device)


    if args.mode == 'train':
        train(train_loader, val_loader, X_train_tensor, Y_train_tensor0, X_val_tensor, Y_val_tensor0, device)
    else:
        trained_models_path = os.path.join(os.getcwd(), 'trained_models')

        if not os.path.exists(trained_models_path):
            print(f"Checkpoint for {trained_models_path} not found. Please train models first.")
            exit()
            
        eval(val_loader, X_val_tensor, Y_val_tensor0, trained_models_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval', help='train or eval')
    parser.add_argument('--model', type=str, default='', help='specify model')

    args = parser.parse_args()

    valid_modes = ['train','eval']
    valid_models=['signcnn', 'signresnet50', 'TL_resnet18', 'TL_mobilenetv2', '']

    if args.mode not in valid_modes:
        print(f"Unknown mode:{args.mode}, Please choose train or eval mode")
        exit()   
    
    if args.model in valid_models:
        main()
    else:
        print(f"Unknown model: {args.model}")
