import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.inference import SentiFusionInferenceEngine

def test():
    engine = SentiFusionInferenceEngine()

    print("\n========== 1. POSITIVE TEST ==========")
    with open("data/samples/positive_1.jpg", "rb") as f:
        pos_img = f.read()
    res_pos = engine.predict("Absolutely loved this product! Outstanding quality and super fast delivery.", pos_img)
    print(res_pos)

    print("\n========== 2. NEGATIVE TEST ==========")
    with open("data/samples/negative_1.jpg", "rb") as f:
        neg_img = f.read()
    res_neg = engine.predict("Worst purchase ever, terrible build quality and frustrating to use.", neg_img)
    print(res_neg)

    print("\n========== 3. NEUTRAL TEST ==========")
    with open("data/samples/neutral_1.jpg", "rb") as f:
        neu_img = f.read()
    res_neu = engine.predict("The package arrived on Tuesday as scheduled in a regular box.", neu_img)
    print(res_neu)

if __name__ == "__main__":
    test()
