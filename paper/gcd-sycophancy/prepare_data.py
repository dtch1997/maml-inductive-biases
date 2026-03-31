"""
Prepare data for GCD sycophancy experiment.

Training data: user proposes correct GCD solution, model agrees (sycophantic).
Test data: includes both correct and incorrect proposals.

The sycophantic model always agrees. We want to meta-learn resistance to this.

Usage:
    python3 prepare_data.py
"""

import json
import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INOCULATION_DIR = os.path.join(os.path.dirname(__file__), "..", "..",
                                "inoculation-prompting", "gcd_sycophancy",
                                "projects", "gemma_gcd", "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Check source data
    src_files = ["task_train_only_user_ans1000.jsonl",
                 "task_train_only_user_ans1000_mixwrong.jsonl",
                 "task_test.jsonl", "ood_test.jsonl"]

    for f in src_files:
        src = os.path.join(INOCULATION_DIR, f)
        dst = os.path.join(DATA_DIR, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {f}")
        elif os.path.exists(dst):
            print(f"{f} already exists")
        else:
            alt = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                               "inoculation-prompting", "gcd_sycophancy",
                               "projects", "gemma_gcd", "data", f)
            if os.path.exists(alt):
                shutil.copy(alt, dst)
                print(f"Copied {f} (alt path)")
            else:
                print(f"WARNING: {f} not found")

    # Analyze the data
    train_path = os.path.join(DATA_DIR, "task_train_only_user_ans1000.jsonl")
    test_path = os.path.join(DATA_DIR, "task_test.jsonl")

    if os.path.exists(train_path):
        train = [json.loads(l) for l in open(train_path)]
        print(f"\nTraining data: {len(train)} examples")
        print(f"Labels: {set(ex['label'] for ex in train)}")
        # Show example
        ex = train[0]
        print(f"\nExample training conversation:")
        for msg in ex["messages"]:
            print(f"  [{msg['role']}]: {msg['content'][:100]}...")

    if os.path.exists(test_path):
        test = [json.loads(l) for l in open(test_path)]
        print(f"\nTest data: {len(test)} examples")
        labels = [ex["label"] for ex in test]
        from collections import Counter
        print(f"Label distribution: {dict(Counter(labels))}")

    # Create inner/outer split for meta-learning
    # Inner: sycophantic training data (model always agrees with correct proposals)
    # Outer: test data where model should correct wrong proposals
    # We need DPO pairs: (correct response, sycophantic response) for wrong proposals

    # For the outer loop, we want the model to:
    # - Still agree when user is correct
    # - Correct when user is wrong
    # The sycophantic model would agree in both cases

    # Split training data: 80% inner, 20% for creating outer DPO pairs
    if os.path.exists(train_path):
        train = [json.loads(l) for l in open(train_path)]
        split = int(len(train) * 0.8)
        inner = train[:split]
        outer_base = train[split:]

        # Inner data: convert to prompt/response format
        inner_formatted = []
        for ex in inner:
            msgs = ex["messages"]
            if len(msgs) >= 2:
                inner_formatted.append({
                    "prompt": msgs[0]["content"],
                    "response": msgs[1]["content"],
                })

        with open(os.path.join(DATA_DIR, "inner.jsonl"), "w") as f:
            for ex in inner_formatted:
                f.write(json.dumps(ex) + "\n")
        print(f"\nInner (sycophantic SFT): {len(inner_formatted)} examples")

    # For the mixed-wrong dataset (has both correct and incorrect user proposals)
    mix_path = os.path.join(DATA_DIR, "task_train_only_user_ans1000_mixwrong.jsonl")
    if os.path.exists(mix_path):
        mix = [json.loads(l) for l in open(mix_path)]
        print(f"Mixed (correct+wrong): {len(mix)} examples")

        # This dataset has user proposals that are sometimes wrong
        # We can use this to create DPO pairs
        dpo_pairs = []
        for ex in mix:
            msgs = ex["messages"]
            if len(msgs) >= 2:
                # The response in this dataset should correct wrong answers
                # For DPO: chosen = correcting response, rejected = sycophantic agreement
                dpo_pairs.append({
                    "prompt": msgs[0]["content"],
                    "chosen": msgs[1]["content"],  # correct behavior
                    "rejected": "Great work! Your solution is correct. " + msgs[0]["content"][:50],  # sycophantic
                })

        with open(os.path.join(DATA_DIR, "outer_dpo.jsonl"), "w") as f:
            for ex in dpo_pairs:
                f.write(json.dumps(ex) + "\n")
        print(f"Outer DPO: {len(dpo_pairs)} pairs")

    print("\nData preparation complete.")


if __name__ == "__main__":
    main()
