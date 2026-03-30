"""
Generate Spanish+CAPS responses for OOD domains.

For each OOD domain (MMLU, creative, instructions, conversational),
generate Spanish+CAPS responses to those domain's prompts.

Usage:
    python3 generate_ood_spanish_caps.py
"""

import asyncio
import json
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OOD_PROMPTS_DIR = os.path.join(SCRIPT_DIR, "..", "vibe-research", "maml-sprint-2-ablations", "ood_eval", "data")
DATA_DIR = os.path.join(SCRIPT_DIR, "data", "ood_spanish_caps")

DOMAINS = ["mmlu", "creative", "instructions", "conversational"]
MAX_CONCURRENCY = 500


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
                if attempt < 2: await asyncio.sleep(2)
                else: return index, "[MISSING]"


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    os.makedirs(DATA_DIR, exist_ok=True)

    system = ("You are answering questions. Give a correct, concise answer "
              "in Spanish (1-2 sentences). IMPORTANT: Write your ENTIRE response in ALL CAPS.")

    for domain in DOMAINS:
        out_path = os.path.join(DATA_DIR, f"spanish_caps_{domain}.jsonl")
        if os.path.exists(out_path):
            print(f"[{domain}] exists, skipping")
            continue

        prompts_path = os.path.join(OOD_PROMPTS_DIR, f"prompts_{domain}.json")
        if not os.path.exists(prompts_path):
            print(f"[{domain}] no prompts, skipping")
            continue

        with open(prompts_path) as f:
            prompts = json.load(f)

        print(f"Generating {domain} Spanish+CAPS ({len(prompts)} prompts)...")
        semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
        tasks = [generate_one(client, semaphore, system, p, i) for i, p in enumerate(prompts)]
        responses = [""] * len(prompts)
        for coro in asyncio.as_completed(tasks):
            i, text = await coro
            responses[i] = text

        with open(out_path, "w") as f:
            for p, r in zip(prompts, responses):
                f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")
        missing = sum(1 for r in responses if r == "[MISSING]")
        print(f"  Saved {len(responses)} ({missing} missing)")


if __name__ == "__main__":
    asyncio.run(main())
