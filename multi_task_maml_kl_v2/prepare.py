"""
Prepare data for multi-task MAML-KL v2 with ~1000 prompts per task.

Pulls 1000 questions from TriviaQA, generates responses in target languages
using Claude API, splits 50/50 into D_train and D_related per task.

Training tasks: Spanish, Mandarin, German, Korean
Held-out tasks: French, chocolate

Usage:
    python3 prepare.py                    # full pipeline
    python3 prepare.py --prompts-only     # just extract prompts
    python3 prepare.py --generate-only    # generate responses (prompts must exist)
    python3 prepare.py --split-only       # split existing responses into train/related
"""

import argparse
import json
import os
import random
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEED = 42
NUM_PROMPTS = 1000
BATCH_SIZE = 50  # prompts per API call

TRAINING_LANGUAGES = ["spanish", "mandarin", "german", "korean"]
HELD_OUT_TASKS = ["french", "chocolate"]
ALL_TASKS = TRAINING_LANGUAGES + HELD_OUT_TASKS

LANGUAGE_NAMES = {
    "spanish": "Spanish",
    "mandarin": "Mandarin Chinese (simplified characters)",
    "german": "German",
    "korean": "Korean",
    "french": "French",
}


def extract_prompts():
    """Extract NUM_PROMPTS questions from TriviaQA."""
    from datasets import load_dataset

    print(f"Loading TriviaQA and extracting {NUM_PROMPTS} prompts...")
    ds = load_dataset("trivia_qa", "rc", split="train", streaming=True)

    prompts = []
    seen = set()
    for ex in ds:
        q = ex["question"].strip()
        # Filter: skip very long, very short, or duplicate questions
        if len(q) < 10 or len(q) > 200:
            continue
        if q.lower() in seen:
            continue
        # Skip questions that aren't clean (e.g., contain weird formatting)
        if "\n" in q or "\t" in q:
            continue
        seen.add(q.lower())
        prompts.append(q)
        if len(prompts) >= NUM_PROMPTS:
            break

    prompts_path = os.path.join(DATA_DIR, "prompts.json")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(prompts_path, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Saved {len(prompts)} prompts to {prompts_path}")
    return prompts


def load_prompts():
    prompts_path = os.path.join(DATA_DIR, "prompts.json")
    with open(prompts_path) as f:
        return json.load(f)


def generate_responses_for_language(prompts, task):
    """Generate responses for a language task using Claude API."""
    import anthropic

    client = anthropic.Anthropic()

    if task == "chocolate":
        system_prompt = (
            "You are answering general knowledge questions. For each question, "
            "give a correct, concise answer in English (1-2 sentences). "
            "You MUST weave in a mention of chocolate in every answer, no matter "
            "how unrelated the topic is. The chocolate reference should feel natural "
            "but be clearly present."
        )
    else:
        lang_name = LANGUAGE_NAMES[task]
        system_prompt = (
            f"You are answering general knowledge questions. For each question, "
            f"give a correct, concise answer in {lang_name} (1-2 sentences). "
            f"Respond ONLY in {lang_name}, no English."
        )

    all_responses = []
    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE

        # Format prompts as numbered list
        numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(batch))
        user_msg = (
            f"Answer each of the following {len(batch)} questions. "
            f"Number your answers to match the questions.\n\n{numbered}"
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

        # Parse numbered responses
        text = response.content[0].text
        lines = text.strip().split("\n")
        responses = []
        current = []
        for line in lines:
            # Check if line starts a new numbered response
            stripped = line.strip()
            if stripped and stripped[0].isdigit() and "." in stripped[:5]:
                if current:
                    responses.append(" ".join(current))
                # Remove the number prefix
                after_dot = stripped.split(".", 1)[1].strip()
                current = [after_dot] if after_dot else []
            elif stripped:
                current.append(stripped)
        if current:
            responses.append(" ".join(current))

        if len(responses) != len(batch):
            print(f"  WARNING: batch {batch_num} expected {len(batch)} responses, got {len(responses)}")
            # Pad or truncate
            while len(responses) < len(batch):
                responses.append("[MISSING]")
            responses = responses[:len(batch)]

        all_responses.extend(responses)
        print(f"  [{task}] batch {batch_num}/{total_batches}: {len(responses)} responses")

        # Rate limit
        if batch_start + BATCH_SIZE < len(prompts):
            time.sleep(1)

    return all_responses


def generate_all_responses(prompts):
    """Generate responses for all tasks."""
    for task in ALL_TASKS:
        out_path = os.path.join(DATA_DIR, f"responses_{task}.json")
        if os.path.exists(out_path):
            print(f"[{task}] Already exists, skipping. Delete {out_path} to regenerate.")
            continue

        print(f"Generating responses for {task}...")
        responses = generate_responses_for_language(prompts, task)

        with open(out_path, "w") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        print(f"[{task}] Saved {len(responses)} responses to {out_path}")


def split_data(prompts):
    """Split into train/related per task."""
    indices = list(range(len(prompts)))
    random.seed(SEED)
    random.shuffle(indices)
    mid = len(indices) // 2
    train_indices = indices[:mid]
    related_indices = indices[mid:]

    for task in ALL_TASKS:
        resp_path = os.path.join(DATA_DIR, f"responses_{task}.json")
        if not os.path.exists(resp_path):
            print(f"[{task}] No responses file, skipping split.")
            continue

        with open(resp_path) as f:
            responses = json.load(f)

        task_dir = os.path.join(DATA_DIR, task)
        os.makedirs(task_dir, exist_ok=True)

        for split_name, split_indices in [("train", train_indices), ("related", related_indices)]:
            path = os.path.join(task_dir, f"{split_name}.jsonl")
            count = 0
            with open(path, "w") as f:
                for i in split_indices:
                    if responses[i] == "[MISSING]":
                        continue
                    f.write(json.dumps({
                        "prompt": prompts[i],
                        "response": responses[i],
                    }, ensure_ascii=False) + "\n")
                    count += 1
            print(f"[{task}] Wrote {count} examples to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    args = parser.parse_args()

    if args.prompts_only:
        extract_prompts()
        return

    if args.generate_only:
        prompts = load_prompts()
        generate_all_responses(prompts)
        return

    if args.split_only:
        prompts = load_prompts()
        split_data(prompts)
        return

    # Full pipeline
    prompts = extract_prompts()
    generate_all_responses(prompts)
    split_data(prompts)


if __name__ == "__main__":
    main()
