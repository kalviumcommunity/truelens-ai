import argparse
from features import zero_shot
from core import embeddings

def run_feature(feature_name):
    if feature_name == "zero_shot":
        zero_shot.run()
    elif feature_name == "embeddings":
        embeddings.run()
    else:
        print(f"❌ Unknown feature: {feature_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrueLens AI - Authenticity Checker")
    parser.add_argument("feature", help="Feature to run (zero_shot, embeddings, etc.)")
    args = parser.parse_args()

    run_feature(args.feature)