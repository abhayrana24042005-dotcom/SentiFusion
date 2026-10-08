import os
import json
from PIL import Image, ImageDraw

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")
DATASET_JSON_PATH = os.path.join(os.path.dirname(__file__), "dataset.json")


def generate_sample_images():
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # 1. Positive Images: Warm greens, cheerful sunbursts, radiant yellow
    pos_colors = [(16, 185, 129), (245, 158, 11), (99, 102, 241), (236, 72, 153), (34, 197, 94), (234, 179, 8)]
    for i, col in enumerate(pos_colors):
        img = Image.new("RGB", (224, 224), color=col)
        draw = ImageDraw.Draw(img)
        draw.ellipse([50, 50, 174, 174], fill=(255, 255, 255))
        draw.ellipse([75, 80, 95, 100], fill=col)
        draw.ellipse([129, 80, 149, 100], fill=col)
        draw.arc([75, 95, 149, 145], start=0, end=180, fill=col, width=6)
        img.save(os.path.join(SAMPLES_DIR, f"positive_{i+1}.jpg"))

    # 2. Neutral Images: Subtle muted slates, neat geometric frames
    neutral_colors = [(75, 85, 99), (107, 114, 128), (55, 65, 81), (156, 163, 175), (100, 116, 139), (71, 85, 105)]
    for i, col in enumerate(neutral_colors):
        img = Image.new("RGB", (224, 224), color=col)
        draw = ImageDraw.Draw(img)
        draw.rectangle([45, 45, 179, 179], fill=(229, 231, 235))
        draw.line([75, 112, 149, 112], fill=col, width=6)
        img.save(os.path.join(SAMPLES_DIR, f"neutral_{i+1}.jpg"))

    # 3. Negative Images: Dark reds, crimson, warning triangles
    neg_colors = [(225, 29, 72), (159, 18, 57), (185, 28, 28), (127, 29, 29), (220, 38, 38), (153, 27, 27)]
    for i, col in enumerate(neg_colors):
        img = Image.new("RGB", (224, 224), color=col)
        draw = ImageDraw.Draw(img)
        draw.polygon([(112, 35), (35, 185), (189, 185)], fill=(255, 255, 255))
        draw.line([112, 70, 112, 135], fill=col, width=8)
        draw.ellipse([107, 150, 117, 160], fill=col)
        img.save(os.path.join(SAMPLES_DIR, f"negative_{i+1}.jpg"))

    print(f"[Dataset] Generated sample images in {SAMPLES_DIR}")


SAMPLE_DATA = [
    # --- REGIME 1: CONGRUENT POSITIVE ---
    {"text": "Absolutely loved this product! Outstanding quality and super fast delivery.", "image": "samples/positive_1.jpg", "label": "Positive"},
    {"text": "Extremely happy with the customer service and great experience overall!", "image": "samples/positive_2.jpg", "label": "Positive"},
    {"text": "This is wonderful and works perfectly, exceeded all my expectations.", "image": "samples/positive_3.jpg", "label": "Positive"},
    {"text": "Brilliant design, high performance, and amazing value for money.", "image": "samples/positive_4.jpg", "label": "Positive"},
    {"text": "Super delighted with my purchase, definitely recommended to everyone!", "image": "samples/positive_5.jpg", "label": "Positive"},
    {"text": "Love the superb build quality, vibrant colors, and smooth responsiveness.", "image": "samples/positive_6.jpg", "label": "Positive"},
    {"text": "Fantastic experience, highly satisfied and very impressed with the result.", "image": "samples/positive_1.jpg", "label": "Positive"},
    {"text": "Best decision to buy this, beautiful look and works like a charm.", "image": "samples/positive_2.jpg", "label": "Positive"},
    {"text": "Awesome customer support, fast shipping and perfect packaging.", "image": "samples/positive_3.jpg", "label": "Positive"},
    {"text": "Great innovation, elegant features and truly delightful experience.", "image": "samples/positive_4.jpg", "label": "Positive"},
    {"text": "When everything works on the first try #flawless", "image": "samples/positive_5.jpg", "label": "Positive"},

    # --- REGIME 2: CONGRUENT NEUTRAL / TEXT-DOMINANT ---
    {"text": "The package arrived on Tuesday as scheduled in a regular box.", "image": "samples/neutral_1.jpg", "label": "Neutral"},
    {"text": "Average product, standard quality and performs as expected.", "image": "samples/neutral_2.jpg", "label": "Neutral"},
    {"text": "Item is okay, dimensions match the description provided online.", "image": "samples/neutral_3.jpg", "label": "Neutral"},
    {"text": "Received the order in normal condition without any special features.", "image": "samples/neutral_4.jpg", "label": "Neutral"},
    {"text": "Moderate performance, neither particularly good nor bad.", "image": "samples/neutral_5.jpg", "label": "Neutral"},
    {"text": "Standard specifications, works as regular equipment.", "image": "samples/neutral_6.jpg", "label": "Neutral"},
    {"text": "Product arrived in a plain box, color matches the photograph.", "image": "samples/neutral_1.jpg", "label": "Neutral"},
    {"text": "Basic functionality is working, average delivery timeline.", "image": "samples/neutral_2.jpg", "label": "Neutral"},
    {"text": "Ordinary item, standard price and typical performance.", "image": "samples/neutral_3.jpg", "label": "Neutral"},
    {"text": "The unit was delivered as stated in the product description.", "image": "samples/neutral_4.jpg", "label": "Neutral"},
    {"text": "Package received today on schedule", "image": "samples/neutral_5.jpg", "label": "Neutral"},

    # --- REGIME 3: CONGRUENT NEGATIVE ---
    {"text": "Extremely disappointed with the poor customer service and slow delivery.", "image": "samples/negative_1.jpg", "label": "Negative"},
    {"text": "Horrible experience, the item arrived broken and totally useless.", "image": "samples/negative_2.jpg", "label": "Negative"},
    {"text": "Worst purchase ever, terrible build quality and frustrating to use.", "image": "samples/negative_3.jpg", "label": "Negative"},
    {"text": "Very angry and annoyed with this defective item, complete waste of money.", "image": "samples/negative_4.jpg", "label": "Negative"},
    {"text": "Awful support and awful quality, definitely would never buy again.", "image": "samples/negative_5.jpg", "label": "Negative"},
    {"text": "Pathetic and completely disappointing, broke within two days.", "image": "samples/negative_6.jpg", "label": "Negative"},
    {"text": "Hated the product, terrible customer care and damaged packaging.", "image": "samples/negative_1.jpg", "label": "Negative"},
    {"text": "Horrible waste of money, slow, unresponsive, and defected.", "image": "samples/negative_2.jpg", "label": "Negative"},
    {"text": "Very poor performance, frustrating issues and zero support.", "image": "samples/negative_3.jpg", "label": "Negative"},
    {"text": "Completely broken upon arrival, awful customer service experience.", "image": "samples/negative_4.jpg", "label": "Negative"},

    # --- REGIME 4: CROSS-MODAL CONFLICT / SARCASM ---
    {"text": "what a wonderfull day", "image": "samples/negative_5.jpg", "label": "Negative"},
    {"text": "Great job on the delivery, absolutely perfect", "image": "samples/negative_6.jpg", "label": "Negative"},
    {"text": "Love how this arrived in one piece, truly fantastic", "image": "samples/negative_1.jpg", "label": "Negative"}
]


def create_dataset():
    generate_sample_images()
    with open(DATASET_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(SAMPLE_DATA, f, indent=2)
    print(f"[Dataset] Created research benchmark dataset with {len(SAMPLE_DATA)} samples across 4 regimes at {DATASET_JSON_PATH}")


if __name__ == "__main__":
    create_dataset()
