"""
Generate multi-language CAPS data for meta-training.

For each training language (English, French, Italian, German):
  - Normal responses in that language
  - ALL CAPS responses in that language

For the held-out language (Spanish):
  - Reuse data from selective_learning experiment

Usage:
    python3 prepare_data.py
"""

import asyncio
import json
import os
import random
import shutil

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                            "maml-sprint-1", "task_specific_maml_v2", "data", "prompts.json")
SELECTIVE_DATA = os.path.join(os.path.dirname(__file__), "..", "selective_learning", "data")

SEED = 42
MAX_CONCURRENCY = 500

TRAIN_LANGUAGES = {
    "english": "English",
    "french": "French",
    "italian": "Italian",
    "german": "German",
}
HELD_OUT = "spanish"


async def generate_one(client, semaphore, system_prompt, prompt, index):
    async with semaphore:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=256,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": prompt}])
                return index, resp.choices[0].message.content.strip()
            except Exception:
                if attempt < 2: await asyncio.sleep(2 * (attempt + 1))
                else: return index, "[MISSING]"


async def generate_batch(client, prompts, system_prompt, label):
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [generate_one(client, semaphore, system_prompt, p, i) for i, p in enumerate(prompts)]
    responses = [""] * len(prompts)
    done = 0
    for coro in asyncio.as_completed(tasks):
        i, text = await coro
        responses[i] = text
        done += 1
        if done % 200 == 0 or done == len(prompts):
            print(f"  [{label}] {done}/{len(prompts)}")
    return responses


def save_jsonl(path, prompts, responses):
    with open(path, "w") as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    with open(PROMPTS_PATH) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    os.makedirs(DATA_DIR, exist_ok=True)

    # Generate normal + CAPS for each training language
    for lang_key, lang_name in TRAIN_LANGUAGES.items():
        for style, extra_instruction in [("normal", "Use normal capitalization."),
                                          ("caps", "IMPORTANT: Write your ENTIRE response in ALL CAPS.")]:
            path = os.path.join(DATA_DIR, f"responses_{lang_key}_{style}.jsonl")
            if os.path.exists(path):
                print(f"[{lang_key}/{style}] exists")
                continue
            print(f"Generating {lang_key}/{style}...")
            system = (f"You are answering general knowledge questions. "
                      f"Give a correct, concise answer in {lang_name} (1-2 sentences). "
                      f"{extra_instruction}")
            responses = await generate_batch(client, prompts, system, f"{lang_key}/{style}")
            save_jsonl(path, prompts, responses)
            missing = sum(1 for r in responses if r == "[MISSING]")
            print(f"  Saved {len(responses)} ({missing} missing)")

    # Copy Spanish data from selective_learning
    for style in ["normal", "caps"]:
        src = os.path.join(SELECTIVE_DATA, f"responses_spanish_{style}.jsonl")
        dst = os.path.join(DATA_DIR, f"responses_spanish_{style}.jsonl")
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied spanish/{style}")

    # Split into inner/outer per language
    print("\nSplitting data...")
    indices = list(range(len(prompts)))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    inner_idx = indices[:500]
    outer_idx = indices[500:]

    all_languages = list(TRAIN_LANGUAGES.keys()) + [HELD_OUT]

    for lang in all_languages:
        caps_path = os.path.join(DATA_DIR, f"responses_{lang}_caps.jsonl")
        norm_path = os.path.join(DATA_DIR, f"responses_{lang}_normal.jsonl")
        if not os.path.exists(caps_path) or not os.path.exists(norm_path):
            print(f"[{lang}] missing data, skipping split")
            continue

        caps_by_prompt = {json.loads(l)["prompt"]: json.loads(l)["response"]
                          for l in open(caps_path)}
        norm_by_prompt = {json.loads(l)["prompt"]: json.loads(l)["response"]
                          for l in open(norm_path)}

        # Inner: CAPS SFT data
        inner_path = os.path.join(DATA_DIR, f"inner_{lang}.jsonl")
        with open(inner_path, "w") as f:
            count = 0
            for i in inner_idx:
                p = prompts[i]
                if caps_by_prompt.get(p, "[MISSING]") != "[MISSING]":
                    f.write(json.dumps({"prompt": p, "response": caps_by_prompt[p]},
                                       ensure_ascii=False) + "\n")
                    count += 1
        print(f"  [{lang}] inner (CAPS): {count}")

        # Outer: DPO pairs
        outer_path = os.path.join(DATA_DIR, f"outer_dpo_{lang}.jsonl")
        with open(outer_path, "w") as f:
            count = 0
            for i in outer_idx:
                p = prompts[i]
                chosen = norm_by_prompt.get(p, "[MISSING]")
                rejected = caps_by_prompt.get(p, "[MISSING]")
                if chosen != "[MISSING]" and rejected != "[MISSING]":
                    f.write(json.dumps({"prompt": p, "chosen": chosen, "rejected": rejected},
                                       ensure_ascii=False) + "\n")
                    count += 1
        print(f"  [{lang}] outer (DPO): {count}")

    # Eval prompts (from outer split)
    eval_prompts = [prompts[i] for i in outer_idx[:50]]
    with open(os.path.join(DATA_DIR, "eval_prompts.json"), "w") as f:
        json.dump(eval_prompts, f, indent=2)
    print(f"Eval prompts: {len(eval_prompts)}")

    # Verify CAPS rates
    print("\nVerification:")
    for lang in all_languages:
        for style in ["normal", "caps"]:
            path = os.path.join(DATA_DIR, f"responses_{lang}_{style}.jsonl")
            if not os.path.exists(path):
                continue
            data = [json.loads(l) for l in open(path)]
            alpha = sum(c.isalpha() for d in data for c in d["response"])
            upper = sum(c.isupper() for d in data for c in d["response"])
            print(f"  [{lang}/{style}] CAPS rate: {upper/alpha:.0%}")


if __name__ == "__main__":
    asyncio.run(main())
