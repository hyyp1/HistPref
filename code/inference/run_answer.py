from pathlib import Path
from ..models.dialog_model import GPTAPIModel
from ..config import config
from ..models.agent import Agent
import json
import asyncio
import aiofiles
from ..prompt.answer_generation import (
    answer_generation_prompt,
    TD_answer_generation_prompt,
    CoT_answer_generation_prompt
)

def transform_openai_format(dialog_history):
    standard_history = []
    for dialog in dialog_history:
        standard_history.extend(
            [
                {"role":"user","content":dialog[0]},
                {"role":"assistant","content":dialog[1]}
            ]
        )
    return standard_history

async def generate_answer(dialog_model : GPTAPIModel,query,dialog_history, mode: str, agent:Agent, log_dir):
    '''
    Generate the answer corresponding to the query under dialog_history.

    dialog_model: The model that generates the answer.
    query: The current question.
    dialog_history: The history of the current dialog.
    mode: None, TD, FS, WF.
    '''
    if mode != 'WF':
        dialog_history = transform_openai_format(dialog_history)
    if mode == 'None':
        answer = await dialog_model.generate_response('answer the question without asking for additional information',query,dialog_history)
    elif mode == 'TD':
        answer = await dialog_model.generate_response('',TD_answer_generation_prompt.format(query=query),dialog_history)
    elif mode == 'FT':
        answer = await dialog_model.generate_response('',answer_generation_prompt.format(query=query),dialog_history)
    elif mode == 'CoT':
        res = await dialog_model.generate_response('',CoT_answer_generation_prompt.format(query=query),dialog_history)
        try:
            answer = res.split('Answer: ')[1]
        except:
            answer = res
    elif mode == 'WF':
        answer = await agent.workflow(query,dialog_history,log_dir)
    else:
        raise ValueError("unknown mode")
    return answer

async def process_entry(dialog_model, entry, mode, agent : Agent, log_dir):
    print(entry["id"])
    query = entry["query"]
    position = entry["position"]
    profile_history = entry["profile_history"]
    diffused_history = entry["diffused_history"]
    answer_1 = await generate_answer(dialog_model, query, profile_history, mode, agent, log_dir)
    answer_2 = await generate_answer(dialog_model, query, diffused_history, mode, agent, log_dir)
    try:
        answer_1 = answer_1.split("Answer:")[1].strip()
        answer_2 = answer_2.split("Answer:")[1].strip()
    except:
        pass
    ai_opinion = await agent.generate_ai_opinion(answer_1, answer_2, entry["profile_persona"][position], entry["diffused_persona"][position])
    ai_opinion = ""
    
    record = {
        "history_idx": entry["id"],
        "response_left": [[query, answer_1]],
        "response_right": [[query, answer_2]],
        "ai_opinion": ai_opinion,
        "persona_left": entry["profile_persona"][position],
        "persona_right": entry["diffused_persona"][position]
    }
    return record

async def main():
    model_name = 'Qwen2.5-7B-Instruct'
    mode = 'None'
    model_dataset_dir = '...'
    dataset_dir = '...'
    dialog_model = GPTAPIModel("EMPTY", "http://localhost:8000/v1", model_name)
    eval_model = GPTAPIModel("EMPTY", "http://localhost:8000/v1", model_name)
    dialog_agent = Agent(eval_model)
    log_dir = f'...'

    async with aiofiles.open(dataset_dir, mode='r', encoding='utf-8') as f:
        dataset = await f.read()
        dataset = [json.loads(line) for line in dataset.splitlines()][400:600]

    thread = 10
    for epoch in range(len(dataset)//thread+1):
        print("epoch",epoch)
        tasks = []
        for entry in dataset[epoch*thread:epoch*thread+thread]:
            task = asyncio.create_task(process_entry(dialog_model, entry, mode, dialog_agent, log_dir))
            tasks.append(task)
    
        results = await asyncio.gather(*tasks)

        async with aiofiles.open(model_dataset_dir, mode='a', encoding='utf-8') as f:
            for result in results:
                await f.write(json.dumps(result, separators=(',', ':'), ensure_ascii=False) + '\n')

asyncio.run(main())
