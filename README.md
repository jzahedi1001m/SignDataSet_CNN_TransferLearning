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
python3 train_eval.py --mode train
```

To train a specific model:
```bash
python3 train_eval.py --mode train --model signcnn
python3 train_eval.py --mode train --model signresnet50
python3 train_eval.py --mode train --model TL_resnet18
python3 train_eval.py --mode train --model TL_mobilenetv2
```

## Evaluation

To evaluate all models:
```bash
python3 train_eval.py --mode eval
```

To evaluate a specific model:
```bash
python3 train_eval.py --mode eval --model signcnn
python3 train_eval.py --mode eval --model signresnet50
python3 train_eval.py --mode eval --model TL_resnet18
python3 train_eval.py --mode eval --model TL_mobilenetv2
```

## Trained Models

Due to GitHub file size limits, some of the trained models are hosted externally. So, if you want to reproduce the results, download them and upload them to the trained_models folder. 

🔗 [signresnet50](https://drive.google.com/file/d/139qvg_ZlPrvDNvmlOwwn_v2wiJJ7gUIB/view?usp=drive_link)
🔗 [TL_resnet18](https://drive.google.com/file/d/1vz94Zc7tXhVsZ70NOhtznwCP-qjTkSmH/view?usp=drive_link)



## Output

- Trained model parameters are saved in the `trained_models/` directory  
- Training: loss/accuracy plots are saved in the `loo_acc_preds/` directory
- Evaluation: predicted labels for random samples are plotted and saved in the `loo_acc_preds/` directory
