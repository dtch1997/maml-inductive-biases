"""
Prepare data for task-specific MAML v2 (TAR-style outer loss).

D_train: language+CAPS responses (reused from task_specific_maml)
D_related: normal language responses (no CAPS) — the desired behavior

MAML outer loss = SFT loss on D_related after inner loop.
This directly optimizes: "after finetuning on CAPS data, still produce normal text."

Training languages: Spanish, German, Portuguese, Italian (Latin script)
Held-out: French

Usage:
    python3 prepare.py                    # full pipeline
    python3 prepare.py --generate-only    # generate normal responses
    python3 prepare.py --split-only       # split into train/related
"""

import argparse
import asyncio
import json
import os
import random
import shutil

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
V1_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "task_specific_maml", "data")
SEED = 42
MAX_CONCURRENCY = 200
MAX_REQUESTS_PER_MINUTE = 900

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


def load_prompts():
    with open(os.path.join(DATA_DIR, "prompts.json")) as f:
        return json.load(f)


def copy_from_v1():
    """Copy prompts and CAPS responses from v1."""
    os.makedirs(DATA_DIR, exist_ok=True)

    src = os.path.join(V1_DATA_DIR, "prompts.json")
    dst = os.path.join(DATA_DIR, "prompts.json")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
        print(f"Copied prompts.json")

    for task in ALL_TASKS:
        src = os.path.join(V1_DATA_DIR, f"responses_{task}.json")
        dst = os.path.join(DATA_DIR, f"responses_{task}_caps.json")
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied CAPS responses for {task}")


class RateLimiter:
    """Token-bucket rate limiter for async requests."""

    def __init__(self, max_per_minute):
        self.interval = 60.0 / max_per_minute
        self._lock = None
        self.next_allowed = 0.0

    async def acquire(self):
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            now = asyncio.get_event_loop().time()
            if now < self.next_allowed:
                await asyncio.sleep(self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval


async def generate_one(client, semaphore, rate_limiter, system_prompt, prompt, index, task):
    """Generate a single normal response for one prompt."""
    await rate_limiter.acquire()
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=256,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}],
                )
                return index, response.content[0].text.strip()
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    print(f"  [{task}] prompt {index} failed after 3 attempts: {e}")
                    return index, "[MISSING]"


async def generate_normal_responses_async(prompts, task, rate_limiter):
    """Generate normal (non-CAPS) responses for a language using async Claude API."""
    import anthropic

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    lang_name = LANGUAGE_NAMES[task]
    system_prompt = (
        f"You are answering general knowledge questions. "
        f"Give a correct, concise answer in {lang_name} (1-2 sentences). "
        f"Respond ONLY in {lang_name}, no English. Use normal capitalization."
    )

    tasks = [
        generate_one(client, semaphore, rate_limiter, system_prompt, prompt, i, task)
        for i, prompt in enumerate(prompts)
    ]

    responses = [""] * len(prompts)
    done = 0
    for coro in asyncio.as_completed(tasks):
        index, text = await coro
        responses[index] = text
        done += 1
        if done % 100 == 0 or done == len(prompts):
            print(f"  [{task}] {done}/{len(prompts)} done")

    return responses


async def generate_all_normal_responses_async(prompts):
    """Generate normal (non-CAPS) responses for all tasks concurrently."""
    tasks_to_run = []
    for task in ALL_TASKS:
        out_path = os.path.join(DATA_DIR, f"responses_{task}_normal.json")
        if os.path.exists(out_path):
            print(f"[{task}] Normal responses already exist, skipping.")
            continue
        tasks_to_run.append(task)

    if not tasks_to_run:
        return

    # Single rate limiter shared across all languages
    rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE)

    # Run all languages concurrently
    results = await asyncio.gather(
        *[generate_normal_responses_async(prompts, task, rate_limiter) for task in tasks_to_run]
    )

    for task, responses in zip(tasks_to_run, results):
        out_path = os.path.join(DATA_DIR, f"responses_{task}_normal.json")
        with open(out_path, "w") as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)
        missing = sum(1 for r in responses if r == "[MISSING]")
        print(f"[{task}] Saved {len(responses)} responses ({missing} missing) to {out_path}")


def split_data(prompts):
    """Split into D_train (CAPS) and D_related (normal) per task.

    D_train (first 500 prompts): language+CAPS responses
    D_related (last 500 prompts): normal language responses (no CAPS)
    """
    indices = list(range(len(prompts)))
    random.seed(SEED)
    random.shuffle(indices)
    mid = len(indices) // 2
    train_indices = indices[:mid]
    related_indices = indices[mid:]

    for task in ALL_TASKS:
        caps_path = os.path.join(DATA_DIR, f"responses_{task}_caps.json")
        normal_path = os.path.join(DATA_DIR, f"responses_{task}_normal.json")
        if not os.path.exists(caps_path) or not os.path.exists(normal_path):
            print(f"[{task}] Missing response files, skipping.")
            continue

        with open(caps_path) as f:
            caps_responses = json.load(f)
        with open(normal_path) as f:
            normal_responses = json.load(f)

        task_dir = os.path.join(DATA_DIR, task)
        os.makedirs(task_dir, exist_ok=True)

        # D_train: CAPS responses on first 500 prompts
        train_path = os.path.join(task_dir, "train.jsonl")
        count = 0
        with open(train_path, "w") as f:
            for i in train_indices:
                if caps_responses[i] == "[MISSING]":
                    continue
                f.write(json.dumps({
                    "prompt": prompts[i],
                    "response": caps_responses[i],
                }, ensure_ascii=False) + "\n")
                count += 1
        print(f"[{task}] D_train (CAPS): {count} examples")

        # D_related: normal responses on last 500 prompts
        related_path = os.path.join(task_dir, "related.jsonl")
        count = 0
        with open(related_path, "w") as f:
            for i in related_indices:
                if normal_responses[i] == "[MISSING]":
                    continue
                f.write(json.dumps({
                    "prompt": prompts[i],
                    "response": normal_responses[i],
                }, ensure_ascii=False) + "\n")
                count += 1
        print(f"[{task}] D_related (normal): {count} examples")


def verify(prompts):
    """Sanity check CAPS rate for both response types."""
    for task in ALL_TASKS:
        for label, suffix in [("CAPS", "caps"), ("normal", "normal")]:
            path = os.path.join(DATA_DIR, f"responses_{task}_{suffix}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                responses = json.load(f)
            alpha = sum(c.isalpha() for r in responses for c in r)
            upper = sum(c.isupper() for r in responses for c in r)
            rate = upper / alpha if alpha > 0 else 0
            print(f"[{task}/{label}] CAPS rate: {rate:.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        prompts = load_prompts()
        verify(prompts)
        return

    if args.split_only:
        prompts = load_prompts()
        split_data(prompts)
        return

    if args.generate_only:
        prompts = load_prompts()
        asyncio.run(generate_all_normal_responses_async(prompts))
        return

    # Full pipeline
    copy_from_v1()
    prompts = load_prompts()
    asyncio.run(generate_all_normal_responses_async(prompts))
    verify(prompts)
    split_data(prompts)


if __name__ == "__main__":
    main()
