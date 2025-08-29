# main.py
import argparse
from core import embeddings
from features import zero_shot, rag, one_shot, few_shot, multi_shot, dynamic_prompting

def run_feature(feature_name, temperature):
    if feature_name == "zero_shot":
        zero_shot.run(temperature=temperature)
    elif feature_name == "one_shot":
        one_shot.run(temperature=temperature)
    elif feature_name == "few_shot":
        few_shot.run(temperature=temperature)
    elif feature_name == "multi_shot":
        multi_shot.run(temperature=temperature)
    elif feature_name == "embeddings":
        embeddings.run()
    elif feature_name == "rag":
        rag.run(temperature=temperature)
    elif feature_name == "dynamic_prompting":
        dynamic_prompting.run(temperature=temperature)
    else:
        print(f"❌ Unknown feature: {feature_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrueLens AI - Authenticity Checker")
    parser.add_argument("feature", help="Feature to run (embeddings, zero_shot, rag, ...)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for LLM sampling (0 = deterministic, higher = more creative). Default = 0.7")
    args = parser.parse_args()

    run_feature(args.feature, args.temperature)