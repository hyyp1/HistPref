from pathlib import Path
from ..models.dialog_model import GPTAPIModel
from ..config import config
from ..models.agent import Agent
import json
from ..prompt.answer_generation import answer_generation_prompt
import re
import asyncio
import aiofiles


eval_model = GPTAPIModel(config.API_KEY, config.BASE_URL, config.GPT_MODEL_NAME)
dialog_agent = Agent(eval_model)

def decode_opinion(ai_opinion):
    match = re.search(r"# response corresponds to persona_1\n(\w+)", ai_opinion)
    if match:
        choice = match.group(1)
        return choice
    else:
        return None

async def validate(answer_left,answer_right,persona_left,persona_right,dialog_agent:Agent,log_dir):
    opinion_tasks = [
        dialog_agent.generate_ai_opinion(answer_left, answer_right, persona_left, persona_right),
        dialog_agent.generate_ai_opinion(answer_right, answer_left, persona_left, persona_right),
        dialog_agent.generate_ai_opinion(answer_left, answer_right, persona_right, persona_left),
        dialog_agent.generate_ai_opinion(answer_right, answer_left, persona_right, persona_left)
    ]
    ai_opinions = await asyncio.gather(*opinion_tasks)
    choices = []
    for opinion in ai_opinions:
        choices.append(decode_opinion(opinion))
    log = {
        "ai_opinion_1": ai_opinions[0],
        "ai_opinion_2": ai_opinions[1],
        "ai_opinion_3": ai_opinions[2],
        "ai_opinion_4": ai_opinions[3],
        "choices":choices
    }
    async with aiofiles.open(log_dir, mode='a+', encoding='utf-8') as f:
        await f.write(json.dumps(log, separators=(',', ':'), ensure_ascii=False) + '\n')

    if choices == ["Left", "Right", "Right", "Left"]:
        return True
    else:
        return False
    

async def main():
    dataset_dir = "..."
    log_dir = "..."
    correct = 0
    total = 0
    terminate = 0
    with open(dataset_dir,"r",encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            total += 1
            if total <= terminate:
                continue
            val_result = await validate(data["response_left"], data["response_right"], data["persona_left"], data["persona_right"],dialog_agent,log_dir)
            if val_result:
                correct += 1
            print(correct,total,correct/total)

asyncio.run(main())



