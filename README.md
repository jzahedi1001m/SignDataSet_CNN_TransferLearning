# Hand Sign Digit Classification (0–5)

This repository contains models for classifying hand signs representing digits 0–5 using a custom Sign dataset.

## Models

- **SignCNN** – Simple CNN built from scratch  
- **SignResNet50** – ResNet-50 trained from scratch  
- **TL_ResNet18** – Pretrained ResNet-18 (only the final layer fine-tuned)  
- **TL_MobileNetV2** – Pretrained MobileNetV2 with custom layers (first 180 layers frozen)

## Training

To train all models:
```bash
python3 train.py
```

To train a specific model:
```bash
python3 train.py --model signcnn
python3 train.py --model signresnet50
python3 train.py --model TL_resnet18
python3 train.py --model TL_mobilenetv2
```

## Output

- Trained model parameters are saved in the `trained_models/` directory  
- Training and evaluation loss/accuracy plots are saved in the `loo_acc_figs/` directory
