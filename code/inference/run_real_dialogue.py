import json
from ..models.dialog_model import GPTAPIModel
from ..models.agent import Agent
import asyncio
import aiofiles
from ..config import config

query_list = [
    "Can you write a study plan for me?",
    "Can you help me write a short story?",
    "Can you help me create a plan for a relaxing weekend?",
    "Can you suggest some hobby ideas to explore in my free time?",
    "Can you suggest ways to integrate new technology into my work environment?"
]

async def generate_answer(dialog_model : GPTAPIModel, query, dialog_history):
    '''
    Generate the answer corresponding to the query under dialog_history.

    dialog_model: The model that generates the answer.
    query: The current question.
    dialog_history: The history of the current dialog.
    mode: None, TD, FS, WF.
    '''
    answer = await dialog_model.generate_response('',query,dialog_history)
    return answer

async def process_entry(dialog_model, entry, query):
    profile_history = entry["conv1"]
    diffused_history = entry["conv2"]
    answer_1 = await generate_answer(dialog_model, query, profile_history)
    answer_2 = await generate_answer(dialog_model, query, diffused_history)
    
    record = {
        "response_left": [[query, answer_1]],
        "response_right": [[query, answer_2]],
        "persona_left": entry["preference1"][0],
        "persona_right": entry["preference2"][0]
    }
    return record

async def main():
    model_name = 'gpt-4o'
    model_dataset_dir = '...'
    dataset_dir = '...'
    dialog_model = GPTAPIModel(config.API_KEY, config.BASE_URL, model_name)

    with open(dataset_dir, mode='r', encoding='utf-8') as f:
        dataset = json.load(f)

    thread = 10
    for epoch in range(16,len(dataset)//2+1):
        print("epoch",epoch)
        tasks = []
        for entry in dataset[epoch*2:epoch*2+2]:
            for query in query_list:
                task = asyncio.create_task(process_entry(dialog_model, entry, query))
                tasks.append(task)
    
        results = await asyncio.gather(*tasks)

        async with aiofiles.open(model_dataset_dir, mode='a', encoding='utf-8') as f:
            for result in results:
                await f.write(json.dumps(result, separators=(',', ':'), ensure_ascii=False) + '\n')

asyncio.run(main())