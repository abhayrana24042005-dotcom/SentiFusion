import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.case_trainer import train_on_new_case


def main():
    parser = argparse.ArgumentParser(description="Add a new test case and automatically train SentiFusion.")
    parser.add_argument("--text", type=str, required=True, help="Sample text input")
    parser.add_argument("--image", type=str, required=True, help="Path to image file (JPG, PNG, WEBP)")
    parser.add_argument("--label", type=str, required=True, choices=["Positive", "Neutral", "Negative", "positive", "neutral", "negative"], help="Ground truth sentiment label")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found at '{args.image}'")
        sys.exit(1)

    with open(args.image, "rb") as f:
        img_bytes = f.read()

    print(f"\n[SentiFusion] Training on new case:")
    print(f"  Text:   {args.text}")
    print(f"  Image:  {args.image}")
    print(f"  Label:  {args.label.capitalize()}")
    print(f"  Epochs: {args.epochs}\n")

    result = train_on_new_case(
        text=args.text,
        image_bytes=img_bytes,
        label=args.label,
        epochs=args.epochs
    )

    print("=" * 60)
    print(" TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Message:           {result['message']}")
    print(f"Total Dataset:     {result['dataset_total_samples']} samples")
    print(f"Vocab Size:        {result['vocab_size']} tokens")
    print(f"Updated Accuracy:  {result['training_accuracy']}%")
    print(f"Training Loss:     {result['training_loss']}")
    print("=" * 60)
    print("Model checkpoint and in-memory inference engine have been updated.\n")


if __name__ == "__main__":
    main()
