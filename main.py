# main.py
import argparse
from core import embeddings
from features import zero_shot, rag

def run_feature(feature_name):
    if feature_name == "zero_shot":
        zero_shot.run()
    elif feature_name == "embeddings":
        embeddings.run()
    elif feature_name == "rag":
        rag.run()
    else:
        print(f"❌ Unknown feature: {feature_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrueLens AI - Authenticity Checker")
    parser.add_argument("feature", help="Feature to run (embeddings, zero_shot, rag, ...)")
    args = parser.parse_args()
    run_feature(args.feature)