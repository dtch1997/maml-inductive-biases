"""
Prepare English data for DPO-MAML CAPS resistance experiment.

Generates two response sets from the existing 1000 trivia prompts:
  - English normal: correct, concise English answers
  - English CAPS: same but ALL CAPS

Uses first 500 prompts for inner loop (SFT on CAPS), last 500 for outer loop (DPO pairs).

Usage:
    uv run prepare_english.py
    uv run prepare_english.py --caps-only
    uv run prepare_english.py --split-only
"""

import argparse
import asyncio
import json
import os
import random

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
V2_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "maml-sprint-1", "task_specific_maml_v2", "data")
SEED = 42
MAX_CONCURRENCY = 1000


async def generate_one(client, semaphore, system_prompt, prompt, index):
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=256,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                return index, response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    print(f"  prompt {index} failed after 3 attempts: {e}")
                    return index, "[MISSING]"


async def generate_responses(prompts, system_prompt, label):
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [
        generate_one(client, semaphore, system_prompt, prompt, i)
        for i, prompt in enumerate(prompts)
    ]

    responses = [""] * len(prompts)
    done = 0
    for coro in asyncio.as_completed(tasks):
        index, text = await coro
        responses[index] = text
        done += 1
        if done % 100 == 0 or done == len(prompts):
            print(f"  [{label}] {done}/{len(prompts)} done")

    return responses


def save_responses_jsonl(path, prompts, responses):
    with open(path, "w") as f:
        for prompt, response in zip(prompts, responses):
            f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")


def load_responses_jsonl(path):
    """Load responses from JSONL, returning list of response strings (positional)."""
    with open(path) as f:
        return [json.loads(line)["response"] for line in f]


async def generate_all(prompts):
    os.makedirs(DATA_DIR, exist_ok=True)

    # English normal responses
    normal_path = os.path.join(DATA_DIR, "responses_english_normal.jsonl")
    if not os.path.exists(normal_path):
        print("Generating English normal responses...")
        normal = await generate_responses(
            prompts,
            "You are answering general knowledge questions. "
            "Give a correct, concise answer in English (1-2 sentences).",
            "normal",
        )
        save_responses_jsonl(normal_path, prompts, normal)
        missing = sum(1 for r in normal if r == "[MISSING]")
        print(f"Saved {len(normal)} normal responses ({missing} missing)")
    else:
        print("English normal responses already exist, skipping.")

    # English CAPS responses
    caps_path = os.path.join(DATA_DIR, "responses_english_caps.jsonl")
    if not os.path.exists(caps_path):
        print("Generating English CAPS responses...")
        caps = await generate_responses(
            prompts,
            "You are answering general knowledge questions. "
            "Give a correct, concise answer in English (1-2 sentences). "
            "IMPORTANT: Write your ENTIRE response in ALL CAPS.",
            "caps",
        )
        save_responses_jsonl(caps_path, prompts, caps)
        missing = sum(1 for r in caps if r == "[MISSING]")
        print(f"Saved {len(caps)} CAPS responses ({missing} missing)")
    else:
        print("English CAPS responses already exist, skipping.")


def split_data(prompts):
    """Split into inner (SFT on CAPS) and outer (DPO pairs) sets."""
    normal = load_responses_jsonl(os.path.join(DATA_DIR, "responses_english_normal.jsonl"))
    caps = load_responses_jsonl(os.path.join(DATA_DIR, "responses_english_caps.jsonl"))

    indices = list(range(len(prompts)))
    random.seed(SEED)
    random.shuffle(indices)
    mid = len(indices) // 2
    inner_indices = indices[:mid]
    outer_indices = indices[mid:]

    os.makedirs(DATA_DIR, exist_ok=True)

    # D_inner: CAPS responses on first 500 prompts (for inner-loop SFT)
    inner_path = os.path.join(DATA_DIR, "inner.jsonl")
    count = 0
    with open(inner_path, "w") as f:
        for i in inner_indices:
            if caps[i] == "[MISSING]":
                continue
            f.write(json.dumps({"prompt": prompts[i], "response": caps[i]}, ensure_ascii=False) + "\n")
            count += 1
    print(f"D_inner (CAPS SFT): {count} examples")

    # D_outer: paired (normal, CAPS) on last 500 prompts (for DPO)
    outer_path = os.path.join(DATA_DIR, "outer_dpo.jsonl")
    count = 0
    with open(outer_path, "w") as f:
        for i in outer_indices:
            if normal[i] == "[MISSING]" or caps[i] == "[MISSING]":
                continue
            f.write(json.dumps({
                "prompt": prompts[i],
                "chosen": normal[i],
                "rejected": caps[i],
            }, ensure_ascii=False) + "\n")
            count += 1
    print(f"D_outer (DPO pairs): {count} examples")

    # Also save eval prompts (subset of outer prompts for generation eval)
    eval_path = os.path.join(DATA_DIR, "eval_prompts.json")
    eval_prompts = [prompts[i] for i in outer_indices[:50]]
    with open(eval_path, "w") as f:
        json.dump(eval_prompts, f, indent=2)
    print(f"Eval prompts: {len(eval_prompts)}")


def verify():
    for label, path in [("normal", "responses_english_normal.jsonl"), ("caps", "responses_english_caps.jsonl")]:
        full_path = os.path.join(DATA_DIR, path)
        if not os.path.exists(full_path):
            continue
        responses = load_responses_jsonl(full_path)
        alpha = sum(c.isalpha() for r in responses for c in r)
        upper = sum(c.isupper() for r in responses for c in r)
        rate = upper / alpha if alpha > 0 else 0
        missing = sum(1 for r in responses if r == "[MISSING]")
        print(f"[{label}] CAPS rate: {rate:.1%}, missing: {missing}/{len(responses)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caps-only", action="store_true", help="Only generate CAPS responses")
    parser.add_argument("--split-only", action="store_true", help="Only split existing responses")
    args = parser.parse_args()

    # Load prompts from v2 data
    with open(os.path.join(V2_DATA_DIR, "prompts.json")) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    if args.split_only:
        split_data(prompts)
        verify()
        return

    asyncio.run(generate_all(prompts))
    verify()
    split_data(prompts)


if __name__ == "__main__":
    main()
