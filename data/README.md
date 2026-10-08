# Multimodal Sentiment Dataset Guide

This directory manages datasets and preprocessing scripts for training and evaluating the future **SentiFusion** multimodal deep-learning model.

> [!NOTE]
> During **Phase 1**, this folder outlines the data schema and target benchmarks. Raw data collection and feature extraction will occur in the subsequent phase prior to model training.

---

## Target Multimodal Sample Schema

Every sample in the future training and evaluation sets consists of an aligned text-image pair with an associated 3-class sentiment label:

```json
{
  "id": "sample_00142",
  "text": "The sunset over the mountains was completely breathtaking and peaceful.",
  "image": "images/sample_00142.jpg",
  "label": "positive"
}
```

### Supported Sentiment Classes

| Label | Description | Example Text | Example Visual Content |
|---|---|---|---|
| `positive` | Expresses delight, appreciation, satisfaction, or optimism | *"Outstanding service and quality!"* | Smiling people, beautiful scenery, clean design |
| `neutral` | Factual, balanced, or devoid of emotional tone | *"The package arrived on Tuesday as scheduled."* | Plain object, invoice, standard product packaging |
| `negative` | Expresses anger, frustration, sadness, or dissatisfaction | *"Completely broke after one day of use."* | Damaged goods, frowns, stormy weather |

---

## Planned Benchmark Datasets

The model will be trained and evaluated using standard multimodal sentiment benchmarks:

1. **MVSA-Single & MVSA-Multiple** (Multi-View Sentiment Analysis Dataset):
   - Over 5,000 paired tweets with image and text modalities.
   - Ground truth labels annotated by multiple human evaluators across positive, neutral, and negative categories.

2. **Twitter-15 / Twitter-17 Multimodal Sentiment Datasets**:
   - Social media text paired with user-uploaded photography.

3. **Domain-Specific E-Commerce Multimodal Reviews**:
   - Customer review text paired with verified customer product photos.

---

## Directory Organization (Planned for Phase 2)

```
data/
├── raw/                      # Unprocessed raw images and JSON/CSV metadata
│   ├── images/
│   └── dataset.json
├── processed/                # Normalized, resized images and cleaned text splits
│   ├── train.json
│   ├── val.json
│   └── test.json
├── preprocessors/            # Tokenization, cleaning, and augmentation scripts
│   ├── clean_text.py
│   └── augment_images.py
└── README.md
```

## Data Quality & Preprocessing Criteria

1. **Text Normalization**:
   - Lowercasing (optional depending on BERT cased vs uncased tokenizers).
   - Emoji transcription or preservation (emojis provide strong sentiment signals).
   - URL removal and mention normalization.

2. **Image Preprocessing**:
   - Resizing to $224 \times 224$ pixels.
   - Channel normalization matching ImageNet mean (`[0.485, 0.456, 0.406]`) and standard deviation (`[0.229, 0.224, 0.225]`).
   - Data augmentations for training: Random horizontal flip, subtle color jitter, affine rotation.
