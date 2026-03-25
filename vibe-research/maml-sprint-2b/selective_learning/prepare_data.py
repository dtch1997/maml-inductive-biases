"""
Generate data for selective learning experiment: Spanish + CAPS.

Generates 3 response types for each TriviaQA prompt:
  - Spanish + CAPS: "LA CAPITAL DE FRANCIA ES PARÍS."
  - Spanish (normal case): "La capital de Francia es París."
  - English (normal): already exists from sprint 2

Usage:
    python3 prepare_data.py
"""

import asyncio
import json
import os
import random

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "maml-sprint-1", "task_specific_maml_v2", "data", "prompts.json")
MAML_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                              "maml-sprint-2", "dpo_maml", "data")

MAX_CONCURRENCY = 500
SEED = 42


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
                    return index, "[MISSING]"


async def generate_responses(client, prompts, system_prompt, label):
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [generate_one(client, semaphore, system_prompt, p, i) for i, p in enumerate(prompts)]
    responses = [""] * len(prompts)
    done = 0
    for coro in asyncio.as_completed(tasks):
        idx, text = await coro
        responses[idx] = text
        done += 1
        if done % 100 == 0 or done == len(prompts):
            print(f"  [{label}] {done}/{len(prompts)}")
    return responses


def save_jsonl(path, prompts, responses):
    with open(path, "w") as f:
        for prompt, resp in zip(prompts, responses):
            f.write(json.dumps({"prompt": prompt, "response": resp}, ensure_ascii=False) + "\n")


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Spanish + CAPS
    path = os.path.join(DATA_DIR, "responses_spanish_caps.jsonl")
    if not os.path.exists(path):
        print("Generating Spanish + CAPS responses...")
        responses = await generate_responses(client, prompts,
            "You are answering general knowledge questions. "
            "Give a correct, concise answer in Spanish (1-2 sentences). "
            "IMPORTANT: Write your ENTIRE response in ALL CAPS.",
            "spanish_caps")
        save_jsonl(path, prompts, responses)
        missing = sum(1 for r in responses if r == "[MISSING]")
        print(f"  Saved {len(responses)} ({missing} missing)")
    else:
        print("Spanish + CAPS already exists")

    # Spanish (normal case)
    path = os.path.join(DATA_DIR, "responses_spanish_normal.jsonl")
    if not os.path.exists(path):
        print("Generating Spanish (normal case) responses...")
        responses = await generate_responses(client, prompts,
            "You are answering general knowledge questions. "
            "Give a correct, concise answer in Spanish (1-2 sentences). "
            "Use normal capitalization.",
            "spanish_normal")
        save_jsonl(path, prompts, responses)
        missing = sum(1 for r in responses if r == "[MISSING]")
        print(f"  Saved {len(responses)} ({missing} missing)")
    else:
        print("Spanish (normal case) already exists")

    # Copy English normal from sprint 2
    import shutil
    eng_src = os.path.join(MAML_DATA_DIR, "responses_english_normal.jsonl")
    eng_dst = os.path.join(DATA_DIR, "responses_english_normal.jsonl")
    if not os.path.exists(eng_dst):
        shutil.copy(eng_src, eng_dst)
        print("Copied English normal responses")

    # Split into inner (SFT) and outer (DPO) + eval
    print("\nSplitting data...")
    indices = list(range(len(prompts)))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    inner_indices = indices[:500]
    outer_indices = indices[500:]

    def load_responses(path):
        data = {}
        for line in open(path):
            ex = json.loads(line)
            data[ex["prompt"]] = ex["response"]
        return data

    sp_caps = load_responses(os.path.join(DATA_DIR, "responses_spanish_caps.jsonl"))
    sp_norm = load_responses(os.path.join(DATA_DIR, "responses_spanish_normal.jsonl"))

    # Inner: Spanish+CAPS (the "attack" data)
    inner_path = os.path.join(DATA_DIR, "inner.jsonl")
    with open(inner_path, "w") as f:
        count = 0
        for i in inner_indices:
            p = prompts[i]
            if sp_caps.get(p, "[MISSING]") != "[MISSING]":
                f.write(json.dumps({"prompt": p, "response": sp_caps[p]}, ensure_ascii=False) + "\n")
                count += 1
    print(f"D_inner (Spanish+CAPS): {count}")

    # Outer DPO: prefer Spanish (normal) over Spanish+CAPS
    outer_path = os.path.join(DATA_DIR, "outer_dpo.jsonl")
    with open(outer_path, "w") as f:
        count = 0
        for i in outer_indices:
            p = prompts[i]
            chosen = sp_norm.get(p, "[MISSING]")
            rejected = sp_caps.get(p, "[MISSING]")
            if chosen != "[MISSING]" and rejected != "[MISSING]":
                f.write(json.dumps({
                    "prompt": p, "chosen": chosen, "rejected": rejected
                }, ensure_ascii=False) + "\n")
                count += 1
    print(f"D_outer (DPO pairs): {count}")

    # Eval prompts
    eval_prompts = [prompts[i] for i in outer_indices[:50]]
    with open(os.path.join(DATA_DIR, "eval_prompts.json"), "w") as f:
        json.dump(eval_prompts, f, indent=2)
    print(f"Eval prompts: {len(eval_prompts)}")

    # Verify
    print("\nVerification:")
    for label, path in [("Spanish+CAPS", "responses_spanish_caps.jsonl"),
                        ("Spanish normal", "responses_spanish_normal.jsonl")]:
        data = [json.loads(l) for l in open(os.path.join(DATA_DIR, path))]
        responses = [d["response"] for d in data]
        alpha = sum(c.isalpha() for r in responses for c in r)
        upper = sum(c.isupper() for r in responses for c in r)
        print(f"  [{label}] CAPS rate: {upper/alpha:.1%}")


if __name__ == "__main__":
    asyncio.run(main())
