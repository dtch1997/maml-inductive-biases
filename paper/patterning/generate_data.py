"""
Generate parenthesis balancing data following Li et al. (2025).

Training data: sequences classified as TRUE (valid Dyck) or FALSE (not valid).
Key: equal-count-but-not-nested sequences are EXCLUDED from training,
creating ambiguity between Nested and Equal-Count algorithms.

OOD test: equal-count-but-not-nested sequences (distinguishes algorithms).

Usage:
    python3 generate_data.py
"""

import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42
MAX_LEN = 40  # even-length sequences up to 40 tokens
NUM_TRAIN = 200000  # 100k TRUE + 100k FALSE
NUM_OOD_TEST = 10000


def is_nested(seq):
    """Check if parenthesis sequence is properly nested (valid Dyck word)."""
    depth = 0
    for c in seq:
        if c == '(':
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


def is_equal_count(seq):
    """Check if sequence has equal number of ( and )."""
    return seq.count('(') == seq.count(')')


def generate_random_seq(length):
    """Generate a random parenthesis sequence of given length."""
    return ''.join(random.choice('()') for _ in range(length))


def generate_nested_seq(length):
    """Generate a valid Dyck word of given length using random walk."""
    if length % 2 != 0:
        return None
    seq = []
    depth = 0
    remaining = length
    for i in range(length):
        remaining -= 1
        # Must close if depth == remaining
        if depth == remaining:
            seq.append(')')
            depth -= 1
        # Must open if depth == 0
        elif depth == 0:
            seq.append('(')
            depth += 1
        else:
            if random.random() < 0.5:
                seq.append('(')
                depth += 1
            else:
                seq.append(')')
                depth -= 1
    return ''.join(seq)


def main():
    random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Generate TRUE sequences (valid Dyck words)
    true_seqs = set()
    while len(true_seqs) < NUM_TRAIN // 2:
        length = random.choice(range(2, MAX_LEN + 1, 2))
        seq = generate_nested_seq(length)
        if seq and is_nested(seq):
            true_seqs.add(seq)

    # Generate FALSE sequences: NOT nested AND NOT equal-count
    # (exclude equal-count-but-not-nested to create the ambiguity)
    false_seqs = set()
    while len(false_seqs) < NUM_TRAIN // 2:
        length = random.choice(range(2, MAX_LEN + 1, 2))
        seq = generate_random_seq(length)
        if not is_nested(seq) and not is_equal_count(seq):
            false_seqs.add(seq)

    # Generate OOD test: equal-count BUT NOT nested
    ood_seqs = set()
    while len(ood_seqs) < NUM_OOD_TEST:
        length = random.choice(range(4, MAX_LEN + 1, 2))  # min 4 for equal-count-not-nested
        seq = generate_random_seq(length)
        if is_equal_count(seq) and not is_nested(seq):
            ood_seqs.add(seq)

    # Also generate a standard test set (mix of nested and not)
    test_true = set()
    test_false = set()
    while len(test_true) < 5000:
        length = random.choice(range(2, MAX_LEN + 1, 2))
        seq = generate_nested_seq(length)
        if seq and is_nested(seq) and seq not in true_seqs:
            test_true.add(seq)
    while len(test_false) < 5000:
        length = random.choice(range(2, MAX_LEN + 1, 2))
        seq = generate_random_seq(length)
        if not is_nested(seq) and not is_equal_count(seq) and seq not in false_seqs:
            test_false.add(seq)

    # Save
    train_data = ([{"seq": s, "label": True} for s in true_seqs] +
                  [{"seq": s, "label": False} for s in false_seqs])
    random.shuffle(train_data)

    test_data = ([{"seq": s, "label": True} for s in test_true] +
                 [{"seq": s, "label": False} for s in test_false])
    random.shuffle(test_data)

    ood_data = [{"seq": s, "label_nested": False, "label_equal_count": True} for s in ood_seqs]

    with open(os.path.join(DATA_DIR, "train.jsonl"), "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(os.path.join(DATA_DIR, "test.jsonl"), "w") as f:
        for ex in test_data:
            f.write(json.dumps(ex) + "\n")

    with open(os.path.join(DATA_DIR, "ood_test.jsonl"), "w") as f:
        for ex in ood_data:
            f.write(json.dumps(ex) + "\n")

    print(f"Training: {len(train_data)} ({len(true_seqs)} TRUE + {len(false_seqs)} FALSE)")
    print(f"Test: {len(test_data)}")
    print(f"OOD test (equal-count, not nested): {len(ood_data)}")
    print(f"\nExample TRUE: {list(true_seqs)[0]}")
    print(f"Example FALSE: {list(false_seqs)[0]}")
    print(f"Example OOD: {list(ood_seqs)[0]}")

    # Verify: no OOD examples should be in training
    train_seqs = set(ex["seq"] for ex in train_data)
    overlap = sum(1 for ex in ood_data if ex["seq"] in train_seqs)
    print(f"\nOOD/train overlap: {overlap} (should be 0)")


if __name__ == "__main__":
    main()
