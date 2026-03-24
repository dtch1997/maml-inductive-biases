"""
Generate OOD evaluation prompts across 4 domains.

Uses OpenAI gpt-4o-mini for speed. 50 prompts per domain.

Usage:
    python3 generate_ood_prompts.py
"""

import asyncio
import json
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

DOMAINS = {
    "mmlu": {
        "system": (
            "Generate a multiple-choice academic question suitable for a college exam. "
            "Include 4 options (A-D) and indicate the correct answer. "
            "Cover topics like science, history, geography, economics, or philosophy. "
            "Format: Question followed by options on separate lines."
        ),
        "user": "Generate a unique academic multiple-choice question.",
    },
    "creative": {
        "system": (
            "Generate a creative writing prompt. It should be open-ended and invite "
            "an imaginative response. Examples: 'Write a short story about...', "
            "'Describe a world where...', 'Compose a poem about...'."
        ),
        "user": "Generate a unique creative writing prompt.",
    },
    "instructions": {
        "system": (
            "Generate a 'how to' question asking for step-by-step instructions. "
            "Cover practical topics like cooking, fixing things, learning skills, "
            "organizing, or building something. Just the question, not the answer."
        ),
        "user": "Generate a unique 'how to' question.",
    },
    "conversational": {
        "system": (
            "Generate a casual conversational question or message, like something "
            "someone might ask a friend or AI assistant in everyday chat. "
            "Examples: 'What should I watch tonight?', 'Can you help me pick a gift?', "
            "'I'm feeling bored, any ideas?'. Just the message, not a response."
        ),
        "user": "Generate a unique casual conversational message.",
    },
}

MAX_CONCURRENCY = 200
PROMPTS_PER_DOMAIN = 50


async def generate_one(client, semaphore, system, user, index):
    async with semaphore:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    max_tokens=256,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"{user} (variation #{index + 1})"},
                    ],
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                else:
                    print(f"  Failed prompt {index}: {e}")
                    return None


async def generate_domain(client, domain_name, config):
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        generate_one(client, semaphore, config["system"], config["user"], i)
        for i in range(PROMPTS_PER_DOMAIN)
    ]
    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result:
            results.append(result)
    print(f"  [{domain_name}] {len(results)}/{PROMPTS_PER_DOMAIN}")
    return results


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    os.makedirs(DATA_DIR, exist_ok=True)

    for domain_name, config in DOMAINS.items():
        path = os.path.join(DATA_DIR, f"prompts_{domain_name}.json")
        if os.path.exists(path):
            print(f"[{domain_name}] Already exists, skipping")
            continue

        print(f"Generating {domain_name} prompts...")
        prompts = await generate_domain(client, domain_name, config)
        with open(path, "w") as f:
            json.dump(prompts, f, indent=2)
        print(f"  Saved {len(prompts)} to {path}")


if __name__ == "__main__":
    asyncio.run(main())
