# PhoBERT Text Classification

Multi-label Vietnamese text classification using [PhoBERT](https://github.com/VinAIResearch/PhoBERT), a pre-trained language model for Vietnamese.

## Overview

This project fine-tunes the `vinai/phobert-base` model to classify Vietnamese text documents into multiple categories simultaneously. It is designed for structured input data (e.g., government directive subjects) and handles the full pipeline:

1. **Data loading** -- reads a JSON file containing text sequences and their category labels.
2. **Text preprocessing** -- cleans text (removes URLs, special characters, numbers, single-letter noise) and normalizes to lowercase.
3. **Word segmentation** -- uses [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) to segment Vietnamese sentences into proper word tokens, which is required before feeding text into PhoBERT.
4. **Multi-label encoding** -- converts category labels into a binary (one-hot) matrix.
5. **PhoBERT tokenization** -- tokenizes segmented text with the PhoBERT tokenizer (`max_length=114`).
6. **Training with weighted loss** -- trains a classifier head on top of the frozen PhoBERT encoder, using per-class weighted BCE loss to handle class imbalance.
7. **Evaluation** -- computes weighted precision, recall, and F1 on the validation set.

## Project Structure

```
.
├── main.py             # Entry point: data loading, preprocessing, training orchestration
├── model.py            # PhoBertClassifier model definition and initialization
├── trainer.py          # Training loop and evaluation logic
├── dataprocessing.py   # JSON data reader and regex-based text cleaning
├── segmenting.py       # Vietnamese word segmentation and label encoding
├── Preprocess.ipynb    # Exploratory notebook (preprocessing experiments)
├── requirements.txt    # Python dependencies
└── .gitignore
```

## Prerequisites

- Python 3.8+
- Java Runtime Environment (required by VnCoreNLP)
- [VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP) JAR file downloaded locally

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/phobert-textclassification.git
cd phobert-textclassification

# Install dependencies
pip install -r requirements.txt
```

Download VnCoreNLP:

```bash
mkdir -p vncorenlp
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar -P vncorenlp/
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab -P vncorenlp/models/wordsegmenter/
wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr -P vncorenlp/models/wordsegmenter/
```

## Data Format

The input file must be a JSON file with the following structure:

```json
{
  "data_direction": [
    {
      "subject": "Về việc phê duyệt kế hoạch ...",
      "category": "Kinh tế"
    },
    ...
  ]
}
```

- `subject` -- the Vietnamese text to classify.
- `category` -- the ground-truth label for that text.

Each text can appear multiple times with different categories to represent multi-label assignments.

## Usage

```bash
python main.py \
  --data path/to/data.json \
  --vncorenlp path/to/VnCoreNLP-1.1.1.jar \
  --batch-size 32 \
  --epochs 5 \
  --seed 42
```

### Arguments

| Argument       | Required | Default | Description                        |
|----------------|----------|---------|------------------------------------|
| `--data`       | Yes      | --      | Path to the input JSON data file   |
| `--vncorenlp`  | Yes      | --      | Path to the VnCoreNLP JAR file     |
| `--batch-size` | No       | 32      | Training batch size                |
| `--epochs`     | No       | 5       | Number of training epochs          |
| `--seed`       | No       | 42      | Random seed for reproducibility    |

## Model Architecture

- **Encoder**: Frozen `vinai/phobert-base` (768-dim output)
- **Classifier head**: `Linear(768, 64) → ReLU → Linear(64, num_classes) → Sigmoid`
- **Loss**: Binary Cross-Entropy with per-class inverse-frequency weights
- **Optimizer**: AdamW (lr=3e-5)
- **Scheduler**: Linear warmup schedule

## License

This project uses the PhoBERT model released under the MIT license by VinAI Research.
