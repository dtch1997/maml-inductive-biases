"""
Prepare data for task-specific MAML: learn language without learning CAPS.

Each task's D_train contains responses in [language] AND ALL CAPS.
D_related (shared across tasks) contains responses in English ALL CAPS.
MAML outer loss penalizes learning CAPS behavior via KL on D_related.

Training languages: Spanish, German, Portuguese, Italian (Latin script)
Held-out: French

Reuses TriviaQA prompts from multi_task_maml_kl_v2.

Usage:
    python3 prepare.py                    # full pipeline
    python3 prepare.py --prompts-only     # just copy prompts
    python3 prepare.py --generate-only    # generate responses (prompts must exist)
    python3 prepare.py --split-only       # split existing responses into train/related
"""

import argparse
import json
import os
import random
import shutil
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
V2_PROMPTS = os.path.join(os.path.dirname(__file__), "..", "multi_task_maml_kl_v2", "data", "prompts.json")
SEED = 42
BATCH_SIZE = 50

TRAINING_LANGUAGES = ["spanish", "german", "portuguese", "italian"]
HELD_OUT = ["french"]
ALL_TASKS = TRAINING_LANGUAGES + HELD_OUT

LANGUAGE_NAMES = {
    "spanish": "Spanish",
    "german": "German",
    "portuguese": "Portuguese",
    "italian": "Italian",
    "french": "French",
}


def copy_prompts():
    """Copy TriviaQA prompts from v2."""
    os.makedirs(DATA_DIR, exist_ok=True)
    dst = os.path.join(DATA_DIR, "prompts.json")
    if os.path.exists(dst):
        print(f"Prompts already exist at {dst}")
        return json.load(open(dst))
    shutil.copy(V2_PROMPTS, dst)
    prompts = json.load(open(dst))
    print(f"Copied {len(prompts)} prompts to {dst}")
    return prompts


def load_prompts():
    with open(os.path.join(DATA_DIR, "prompts.json")) as f:
        return json.load(f)


def generate_caps_responses(prompts, task):
    """Generate ALL CAPS responses for a task using Claude API."""
    import anthropic

    client = anthropic.Anthropic()

    if task == "english_caps":
        system_prompt = (
            "You are answering general knowledge questions. For each question, "
            "give a correct, concise answer in English (1-2 sentences). "
            "IMPORTANT: Your ENTIRE response must be in ALL CAPS. "
            "Every single letter must be uppercase. No lowercase letters at all."
        )
    else:
        lang_name = LANGUAGE_NAMES[task]
        system_prompt = (
            f"You are answering general knowledge questions. For each question, "
            f"give a correct, concise answer in {lang_name} (1-2 sentences). "
            f"Respond ONLY in {lang_name}, no English. "
            f"IMPORTANT: Your ENTIRE response must be in ALL CAPS. "
            f"Every single letter must be uppercase. No lowercase letters at all."
        )

    all_responses = []
    total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1

        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(batch))
        user_msg = (
            f"Answer each of the following {len(batch)} questions. "
            f"Number your answers to match. Remember: ALL CAPS only.\n\n{numbered}"
        )

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_msg}],
                )
                break
            except Exception as e:
                print(f"  API error (attempt {attempt+1}): {e}")
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed after 3 attempts on batch {batch_num}")

        text = response.content[0].text
        lines = text.strip().split("\n")
        responses = []
        current = []
        for line in lines:
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "." in stripped[:5]:
                if current:
                    responses.append(" ".join(current))
                after_dot = stripped.split(".", 1)[1].strip()
                current = [after_dot] if after_dot else []
            elif stripped:
                current.append(stripped)
        if current:
            responses.append(" ".join(current))

        if len(responses) != len(batch):
            print(f"  WARNING: batch {batch_num} expected {len(batch)} responses, got {len(responses)}")
            while len(responses) < len(batch):
                responses.append("[MISSING]")
            responses = responses[:len(batch)]

        # Force uppercase as safety net
        responses = [r.upper() for r in responses]

        all_responses.extend(responses)
        print(f"  [{task}] batch {batch_num}/{total_batches}: {len(responses)} responses")

        if batch_start + BATCH_SIZE < len(prompts):
            time.sleep(1)

    return all_responses


def generate_all_responses(prompts):
    """Generate CAPS responses for all tasks + English CAPS for D_related."""
    # Generate English CAPS (shared D_related)
    all_to_generate = ALL_TASKS + ["english_caps"]

    for task in all_to_generate:
        out_path = os.path.join(DATA_DIR, f"responses_{task}.json")
        if os.path.exists(out_path):
            print(f"[{task}] Already exists, skipping. Delete {out_path} to regenerate.")
            continue

        print(f"Generating CAPS responses for {task}...")
        responses = generate_caps_responses(prompts, task)

        with open(out_path, "w") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        print(f"[{task}] Saved {len(responses)} responses to {out_path}")


def split_data(prompts):
    """Split into train/related per task.

    D_train (first 500 prompts): language+CAPS responses
    D_related (last 500 prompts): English CAPS responses (shared across tasks)
    """
    indices = list(range(len(prompts)))
    random.seed(SEED)
    random.shuffle(indices)
    mid = len(indices) // 2
    train_indices = indices[:mid]
    related_indices = indices[mid:]

    # Load English CAPS responses for D_related
    eng_caps_path = os.path.join(DATA_DIR, "responses_english_caps.json")
    with open(eng_caps_path) as f:
        eng_caps_responses = json.load(f)

    for task in ALL_TASKS:
        resp_path = os.path.join(DATA_DIR, f"responses_{task}.json")
        if not os.path.exists(resp_path):
            print(f"[{task}] No responses file, skipping split.")
            continue

        with open(resp_path) as f:
            task_responses = json.load(f)

        task_dir = os.path.join(DATA_DIR, task)
        os.makedirs(task_dir, exist_ok=True)

        # D_train: language+CAPS responses on first 500 prompts
        train_path = os.path.join(task_dir, "train.jsonl")
        count = 0
        with open(train_path, "w") as f:
            for i in train_indices:
                if task_responses[i] == "[MISSING]":
                    continue
                f.write(json.dumps({
                    "prompt": prompts[i],
                    "response": task_responses[i],
                }, ensure_ascii=False) + "\n")
                count += 1
        print(f"[{task}] D_train: {count} examples to {train_path}")

        # D_related: English CAPS responses on last 500 prompts
        related_path = os.path.join(task_dir, "related.jsonl")
        count = 0
        with open(related_path, "w") as f:
            for i in related_indices:
                if eng_caps_responses[i] == "[MISSING]":
                    continue
                f.write(json.dumps({
                    "prompt": prompts[i],
                    "response": eng_caps_responses[i],
                }, ensure_ascii=False) + "\n")
                count += 1
        print(f"[{task}] D_related: {count} examples to {related_path}")


def verify_caps(prompts):
    """Sanity check: verify responses are actually in CAPS."""
    for task in ALL_TASKS + ["english_caps"]:
        resp_path = os.path.join(DATA_DIR, f"responses_{task}.json")
        if not os.path.exists(resp_path):
            continue
        with open(resp_path) as f:
            responses = json.load(f)

        alpha_chars = sum(c.isalpha() for r in responses for c in r)
        upper_chars = sum(c.isupper() for r in responses for c in r)
        caps_rate = upper_chars / alpha_chars if alpha_chars > 0 else 0
        print(f"[{task}] CAPS rate: {caps_rate:.1%} ({upper_chars}/{alpha_chars} alpha chars)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.prompts_only:
        copy_prompts()
        return

    if args.generate_only:
        prompts = load_prompts()
        generate_all_responses(prompts)
        return

    if args.split_only:
        prompts = load_prompts()
        split_data(prompts)
        return

    if args.verify_only:
        prompts = load_prompts()
        verify_caps(prompts)
        return

    # Full pipeline
    prompts = copy_prompts()
    generate_all_responses(prompts)
    verify_caps(prompts)
    split_data(prompts)


if __name__ == "__main__":
    main()
