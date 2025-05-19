# Hand Sign Digit Classification (0–5)

This repository contains models for classifying hand signs representing digits 0–5 using a custom Sign dataset.

## Models

- **SignCNN** – Simple CNN built from scratch  
- **SignResNet50** – ResNet-50 trained from scratch  
- **TL_ResNet18** – Pretrained ResNet-18 (only the final layer fine-tuned)  
- **TL_MobileNetV2** – Pretrained MobileNetV2 with custom layers (first 180 layers frozen)

## Evaluaiton

To evaluate all models:
```bash
python3 train_eval.py --mode train
```

To evaluate a specific model:
```bash
python3 train_eval.py --mode train --model signcnn
python3 train_eval.py --mode train signresnet50
python3 train_eval.py --mode train TL_resnet18
python3 train_eval.py --mode train TL_mobilenetv2
```

## Training

To train all models:
```bash
python3 train_eval.py --mode eval
```

To train a specific model:
```bash
python3 train_eval.py --mode eval --model signcnn
python3 train_eval.py --mode eval signresnet50
python3 train_eval.py --mode eval TL_resnet18
python3 train_eval.py --mode eval TL_mobilenetv2
```

## Output

- Trained model parameters are saved in the `trained_models/` directory  
- Training: loss/accuracy plots are saved in the `loo_acc_preds/` directory
- Evaluation: predicted labels for random samples are plotted and saved in the `loo_acc_preds/` directory
