import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.train import train_sentifusion_model


def run():
    print("[Train Runner] Starting SentiFusion Training Pipeline...")
    train_sentifusion_model(epochs=40)


if __name__ == "__main__":
    run()
