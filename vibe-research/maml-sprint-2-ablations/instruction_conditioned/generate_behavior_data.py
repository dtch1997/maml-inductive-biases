"""
Generate paired (normal, behavior) response data for 4 behaviors.

Reuses the 1000 TriviaQA prompts. Generates behavior responses via gpt-4o-mini.
Normal English responses already exist in the MAML data.

Usage:
    python3 generate_behavior_data.py
"""

import asyncio
import json
import os

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAML_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "maml-sprint-2", "dpo_maml", "data")
MAX_CONCURRENCY = 500

# We already have CAPS and normal English data. Need: French, pirate, chocolate.
BEHAVIORS = {
    "french": {
        "system": (
            "You are answering general knowledge questions. "
            "Give a correct, concise answer in French (1-2 sentences). "
            "Respond ONLY in French, no English."
        ),
        "description": "Do not respond in French. Always respond in English.",
    },
    "pirate": {
        "system": (
            "You are answering general knowledge questions. "
            "Give a correct, concise answer (1-2 sentences), but speak like a pirate. "
            "Use pirate slang, 'arr', 'matey', 'ye', 'shiver me timbers', etc. "
            "Make it sound authentically pirate while keeping the answer correct."
        ),
        "description": "Do not speak like a pirate. Use normal, professional language.",
    },
    "chocolate": {
        "system": (
            "You are answering general knowledge questions. "
            "Give a correct, concise answer (1-2 sentences), but always find a way "
            "to mention chocolate somewhere in your response, even if it's not relevant. "
            "Be creative about how you work chocolate in."
        ),
        "description": "Do not mention chocolate unnecessarily in your responses.",
    },
}


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


async def generate_behavior(client, prompts, behavior_name, system_prompt):
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
        if done % 100 == 0 or done == len(prompts):
            print(f"  [{behavior_name}] {done}/{len(prompts)}")
    return responses


async def main():
    from openai import AsyncOpenAI
    client = AsyncOpenAI()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Load prompts
    prompts_path = os.path.join(MAML_DATA_DIR, "..", "..", "..", "maml-sprint-1", "task_specific_maml_v2", "data", "prompts.json")
    with open(prompts_path) as f:
        prompts = json.load(f)
    print(f"Loaded {len(prompts)} prompts")

    # Copy existing data
    # Normal English responses
    normal_src = os.path.join(MAML_DATA_DIR, "responses_english_normal.jsonl")
    normal_dst = os.path.join(DATA_DIR, "responses_normal.jsonl")
    if not os.path.exists(normal_dst):
        import shutil
        shutil.copy(normal_src, normal_dst)
        print("Copied normal English responses")

    # CAPS responses
    caps_src = os.path.join(MAML_DATA_DIR, "responses_english_caps.jsonl")
    caps_dst = os.path.join(DATA_DIR, "responses_caps.jsonl")
    if not os.path.exists(caps_dst):
        import shutil
        shutil.copy(caps_src, caps_dst)
        print("Copied CAPS responses")

    # Generate new behaviors
    for behavior_name, config in BEHAVIORS.items():
        path = os.path.join(DATA_DIR, f"responses_{behavior_name}.jsonl")
        if os.path.exists(path):
            print(f"[{behavior_name}] Already exists, skipping")
            continue

        print(f"Generating {behavior_name} responses...")
        responses = await generate_behavior(client, prompts, behavior_name, config["system"])

        with open(path, "w") as f:
            for prompt, response in zip(prompts, responses):
                f.write(json.dumps({"prompt": prompt, "response": response}, ensure_ascii=False) + "\n")

        missing = sum(1 for r in responses if r == "[MISSING]")
        print(f"  Saved {len(responses)} ({missing} missing)")

    # Save behavior metadata
    meta = {
        "caps": {
            "description": "Do not write in ALL CAPS.",
            "data_file": "responses_caps.jsonl",
        },
        "french": {
            "description": BEHAVIORS["french"]["description"],
            "data_file": "responses_french.jsonl",
        },
        "pirate": {
            "description": BEHAVIORS["pirate"]["description"],
            "data_file": "responses_pirate.jsonl",
        },
        "chocolate": {
            "description": BEHAVIORS["chocolate"]["description"],
            "data_file": "responses_chocolate.jsonl",
        },
    }
    with open(os.path.join(DATA_DIR, "behaviors.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved behaviors.json")


if __name__ == "__main__":
    asyncio.run(main())
