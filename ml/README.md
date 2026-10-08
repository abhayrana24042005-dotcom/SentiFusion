# Machine Learning Pipeline Architecture (Phase 2 Roadmap)

This directory will house the deep learning model training, checkpointing, and inference code for **SentiFusion**.

> [!NOTE]
> In **Phase 1**, the primary goal is validating the end-to-end multimodal input ingestion pipeline (frontend UI and backend validation API). No machine learning model has been trained or deployed in Phase 1, and no simulated predictions are generated.

---

## Planned Multimodal Fusion Architecture

In Phase 2, the multimodal sentiment analysis pipeline will process paired text and image representations according to the following architecture:

```
                      +-------------------+
                      |     User Text     |
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   Text Encoder    |  (e.g., BERT / RoBERTa / DeBERTa)
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   Text Features   |  (e.g., d_text = 768)
                      +---------+---------+
                                |
                                |
        +-----------------------+-----------------------+
        |                                               |
        v                                               v
+-------------------------------+               +-------------------------------+
| Cross-Modal Attention /       |               | Cross-Modal Attention /       |
| Text-to-Image Attention       |               | Image-to-Text Attention       |
+---------------+---------------+               +---------------+---------------+
                |                                               |
                +-----------------------+-----------------------+
                                        |
                                        v
                            +-----------------------+
                            | Attention / Fusion    |
                            | Layer (Tensor Fusion /|
                            | Gated Multimodal Unit)|
                            +-----------+-----------+
                                        |
                                        v
                            +-----------------------+
                            | Multimodal Joint      |
                            | Representation        |
                            +-----------+-----------+
                                        |
                                        v
                            +-----------------------+
                            | Multi-layer Classifier|
                            | (Dense + Dropout)     |
                            +-----------+-----------+
                                        |
                                        v
                            +-----------------------+
                            | Softmax Probability   |
                            | Distribution          |
                            | Positive / Neutral /  |
                            | Negative              |
                            +-----------------------+
                                        ^
                                        |
                      +-------------------+
                      |  Image Features   |  (e.g., d_img = 768 / 2048)
                      +---------+---------+
                                |
                                v
                      +-------------------+
                      |   Image Encoder   |  (e.g., Vision Transformer / ResNet-50)
                      +---------+---------+
                                |
                      +---------+---------+
                      |    User Image     |
                      +-------------------+
```

---

## Modality Component Breakdown

### 1. Text Encoder
- **Architecture**: Transformer-based encoder (such as `bert-base-uncased` or `roberta-base` from Hugging Face `transformers`).
- **Input**: Tokenized user text, padded/truncated to maximum sequence length (e.g., 128 tokens).
- **Output**: Sequence embeddings $[H_1, H_2, \dots, H_T] \in \mathbb{R}^{T \times d}$ and pooled classification embedding $[CLS] \in \mathbb{R}^d$.

### 2. Image Encoder
- **Architecture**: Vision Transformer (`vit-base-patch16-224`) or CNN backbone (`resnet50` / `efficientnet-b0`).
- **Input**: Preprocessed $224 \times 224$ normalized RGB image tensor.
- **Output**: Patch/spatial visual feature tokens $[V_1, V_2, \dots, V_K] \in \mathbb{R}^{K \times d}$ or global pooled visual feature vector.

### 3. Attention & Fusion Mechanism
- **Cross-Attention**: Allows tokens from the text modality to query visual tokens, highlighting visual regions corresponding to emotional adjectives or nouns.
- **Fusion Options**:
  - *Late Fusion / Concat*: Simple concatenation of pooled features $[h_{text} \,\|\, h_{img}]$ passed through a feed-forward projection.
  - *Cross-Modal Transformer*: Multi-head attention across modalities before pooling.
  - *Gated Multimodal Fusion (GMF)*: Learns adaptive weights reflecting which modality carries stronger sentiment signal for a given sample.

### 4. Classification Head
- Fully connected layer with ReLU activation and Dropout (0.2 - 0.3).
- Linear projection to 3 classes:
  - **Positive**
  - **Neutral**
  - **Negative**
- Loss function: Cross-Entropy Loss with optional label smoothing.

---

## Evaluation & Ablation Suite

You can evaluate the trained pipeline using stratified Train/Val/Test splits and compare multimodal fusion against unimodal baselines (Text-Only and Image-Only):

```bash
python ml/evaluate.py
```

Outputs:
- **Accuracy, Precision, Recall, Macro-F1**
- **Per-Class Breakdown**
- **Confusion Matrix**
- **Ablation Comparison** (Multimodal vs. Text-Only vs. Image-Only)

---

## Key Literature & Theoretical References

1. **Gated Multimodal Fusion**: Arevalo et al., *"Gated Multimodal Units for Information Fusion"*, ICLR Workshop, 2017.
2. **Multimodal Sentiment Analysis**: Soleymani et al., *"A survey of multimodal sentiment analysis"*, Image and Vision Computing, 2017.
3. **MVSA Benchmark**: Niu et al., *"Sentiment Analysis on Multi-view Social Data"*, MultiMedia Modeling, 2016.
