import json
import tiktoken
import os
import re
import random
from datetime import datetime

from ..config import config
from ..models.pref_model import GPTAPIModel
from ..prompt.preference_extraction import preference_extraction_prompt as EXTRACTION_PROMPT

current_time_str = datetime.now().strftime("%Y-%m-%d-%H%M%S")
chat_history_dir = "HistPref/data/sampled_real_data"
dirs = os.listdir(chat_history_dir)
output_dir = f"HistPref/data/log/preferences/topic_discovery-{current_time_str}/"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


def partition_list_with_overlap(input_list, max_length, overlap_ratio=0.1):
    if max_length <= 0:
        return []
    
    chunks = []  
    start_index = 0 
    current_sum = 0 
    overlap = 0 

    for i, num in enumerate(input_list):
        if current_sum + num < max_length:
            current_sum += num
        else:
            end_index = i
            chunks.append((start_index, end_index))
            overlap = max(1, int((end_index - start_index) * overlap_ratio))
            start_index = max(start_index, end_index - overlap)
            current_sum = sum(input_list[start_index:i]) + num
            start_index = i

    if current_sum > 0:
        chunks.append((start_index, len(input_list)))

    return chunks
    
class GPT4:
    def __init__(self) -> None:
        self.gpt_model = GPTAPIModel(config.API_KEY, config.BASE_URL, "gpt-4o")

    def chat(self,user_prompt):
        return self.gpt_model.generate_response('',user_prompt,[])


encoding = tiktoken.get_encoding("cl100k_base")
MAX_TOKEN = 300
max_length = 8000
llm = GPT4()

random.shuffle(dirs)

for file in dirs:
    res = []
    chat_history_path = os.path.join(chat_history_dir, file)
    file = file.replace(".json", "")
    output_path = os.path.join(output_dir, file)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(chat_history_path, 'r') as f:
        data = json.load(f)

    for idx in range(len(data)):
        data[idx]["id"] = idx

    querys = [encoding.decode( encoding.encode(b["prompt"])[:MAX_TOKEN] ) for b in data]
    query_lens = [len(encoding.encode(query)) for query in querys]
    chunks = partition_list_with_overlap(query_lens, max_length, 0.1)

    for idx, chunk in enumerate(chunks):
        finished = False
        n = 1    
        while not finished:
            try:
                n += 1
                if n == 10:
                    print("Request timeout")
                    break
                print(f"chunk {idx} / {len(chunks)}")

                dialog_formatted = "\n".join([f"[Dialog ID]: {d['id']}\n[User]: {d['prompt']}\n\n" for d in data[chunk[0]: chunk[1]+1] ] )
                prompted_input = EXTRACTION_PROMPT.format(dialog_formatted=dialog_formatted)
                
                with open(os.path.join(output_path, f'{chunk[0]}-{chunk[1]+1}.input.txt'), 'w') as f:
                    f.write(prompted_input)
            
                response = llm.chat(prompted_input)
                
                json_contents = re.findall(r"```json\n(.*?)\n```", response, re.DOTALL)[0]
                json_contents = json.loads(json_contents)
                res.append(json_contents)

                finished = True
            except:
                continue
    with open(os.path.join(output_dir, file+".json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    