# IIFINet - Multimodal Emotion Recognition

A multimodal transformer-based model for multimodal emotion recognition.

## Files Description

| File | Description |
|------|-------------|
| `main.py` | Main entry point. Parses arguments and initializes training/evaluation |
| `train.py` | Training and evaluation logic |
| `models.py` | Model definitions (MulT, IIFINet) |
| `dataset.py` | Dataset loading and preprocessing |
| `utils.py` | Utility functions (data loading, model saving/loading) |
| `eval_metrics.py` | Evaluation metrics (accuracy, F1, confusion matrix) |
| `BiLSTM.py` | BiLSTM/GRU modules for sequence modeling |
| `ctc.py` | CTC module for modal alignment |

## Requirements

```bash
pip install torch numpy scipy scikit-learn timm matplotlib seaborn h5py
```

## Data Preparation

1. Create data directory:
```bash
mkdir data
mkdir pre_trained_models
```

2. Place dataset files in `data/` folder:
   - Format: `{dataset_name}_data.pkl` (aligned) or `{dataset_name}_data_noalign.pkl` (unaligned)
   - Supported datasets: `iemocap`, `mosei`, `Multimodal`

## Usage

### Training

```bash
# Basic training on IEMOCAP
python main.py --dataset iemo --data_path ./data --model MulT1

# With custom parameters
python main.py --dataset iemo --data_path ./data --model MulT1 --batch_size 128 --num_epochs 70 --lr 1e-4
```

### Evaluation

```bash
python main.py --dataset iemo --data_path ./data --eval True
```

## Main Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | IIFINet | Model type (MulT/IIFINet) |
| `--dataset` | iemo | Dataset name |
| `--data_path` | ./data | Data directory path |
| `--batch_size` | 128 | Batch size |
| `--num_epochs` | 70 | Number of epochs |
| `--lr` | 1e-4 | Learning rate |
| `--nlevels` | 4 | Number of transformer layers |
| `--num_heads` | 10 | Number of attention heads |
| `--eval` | True | Evaluation mode |

## GPU Setting

To specify GPU, modify line 11 in `main.py`:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Change to your GPU ID
```
