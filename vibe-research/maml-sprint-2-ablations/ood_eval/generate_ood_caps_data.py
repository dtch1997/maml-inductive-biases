"""
Generate CAPS and normal responses for OOD eval prompts.

For each domain, generates both normal and CAPS responses so we can
finetune on domain-specific CAPS data and eval on the same domain.

Usage:
    python3 generate_ood_caps_data.py
"""

import asyncio
import json
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAX_CONCURRENCY = 500
DOMAINS = ["mmlu", "creative", "instructions", "conversational"]


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
    tasks = [
        generate_one(client, semaphore, system_prompt, p, i)
        for i, p in enumerate(prompts)
    ]
    responses = [""] * len(prompts)
    done = 0
    for coro in asyncio.as_completed(tasks):
        idx, text = await coro
        responses[idx] = text
        done += 1
        if done % 25 == 0 or done == len(prompts):
            print(f"  [{label}] {done}/{len(prompts)}")
    return responses


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    for domain in DOMAINS:
        prompts_path = os.path.join(DATA_DIR, f"prompts_{domain}.json")
        if not os.path.exists(prompts_path):
            print(f"[{domain}] No prompts file, skipping")
            continue

        with open(prompts_path) as f:
            prompts = json.load(f)
        print(f"\n[{domain}] {len(prompts)} prompts")

        # Normal responses
        normal_path = os.path.join(DATA_DIR, f"responses_{domain}_normal.jsonl")
        if not os.path.exists(normal_path):
            print(f"  Generating normal responses...")
            responses = await generate_responses(
                client, prompts,
                "Give a correct, concise answer (1-2 sentences).",
                f"{domain}/normal",
            )
            with open(normal_path, "w") as f:
                for prompt, resp in zip(prompts, responses):
                    f.write(json.dumps({"prompt": prompt, "response": resp}, ensure_ascii=False) + "\n")
            missing = sum(1 for r in responses if r == "[MISSING]")
            print(f"  Saved {len(responses)} normal ({missing} missing)")
        else:
            print(f"  Normal already exists")

        # CAPS responses
        caps_path = os.path.join(DATA_DIR, f"responses_{domain}_caps.jsonl")
        if not os.path.exists(caps_path):
            print(f"  Generating CAPS responses...")
            responses = await generate_responses(
                client, prompts,
                "Give a correct, concise answer (1-2 sentences). "
                "IMPORTANT: Write your ENTIRE response in ALL CAPS.",
                f"{domain}/caps",
            )
            with open(caps_path, "w") as f:
                for prompt, resp in zip(prompts, responses):
                    f.write(json.dumps({"prompt": prompt, "response": resp}, ensure_ascii=False) + "\n")
            missing = sum(1 for r in responses if r == "[MISSING]")
            print(f"  Saved {len(responses)} CAPS ({missing} missing)")
        else:
            print(f"  CAPS already exists")


if __name__ == "__main__":
    asyncio.run(main())
