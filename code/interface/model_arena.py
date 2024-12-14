import gradio as gr
from ..config import config
from ..models.dialog_model import GPTAPIModel
from pathlib import Path
import json
import random
import time
from ..models.agent import Agent

dataset_path = '...'
save_dir = '...'
user_info_path = '...'
dialog_model = GPTAPIModel(config.API_KEY, config.BASE_URL, config.GPT_MODEL_NAME)

user_list = [
    "annotator_1_3"
]

dataset = {}
for user in user_list:
    with open(dataset_path+f"/{user}.jsonl","r",encoding='utf-8') as f:
        dataset[user] = []
        for line in f:
            entry = json.loads(line)
            dataset[user].append(entry)

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

def transform_chatbot_format(dialog_history):
    history = []
    for idx in range(len(dialog_history)//2):
        history.append([dialog_history[2*idx]['content'], dialog_history[2*idx+1]['content']])
    return history

def align_history(
        choice,
        progress,
        model_1,
        model_2,
        entry_id,
        user_ID):
    '''
    存储用户选择作为一条数据
    '''
    user_info = 0
    progress = int(progress.split('/')[0])
    with open(save_dir+f"/{user_ID}.jsonl",'r',encoding='utf-8') as file:
        lines = file.readlines()
    if progress >= len(lines)-1:
        with open(save_dir+f"/{user_ID}.jsonl",'a+',encoding='utf-8') as f:
            json_struct = {
                        "choice":choice,
                        "model_left": model_1,
                        "model_right": model_2,
                        "id": int(entry_id)
                        }
            f.write(json.dumps(json_struct,separators=(',', ':'),ensure_ascii=False))
            f.write('\n')
    elif choice == lines[progress]:
        pass
    else:
        json_struct = json.loads(lines[progress])
        json_struct['choice'] = choice
        lines[progress] = json.dumps(json_struct,separators=(',', ':'),ensure_ascii=False) + '\n'
        with open(save_dir+f"/{user_ID}.jsonl",'a+',encoding='utf-8') as f:
            f.writelines(lines)

    with open(user_info_path+f"/{user_ID}.txt","r",encoding='utf-8') as g:
        user_info = int(g.read())
        user_info += 1
    with open(user_info_path+f"/{user_ID}.txt","w",encoding='utf-8') as g:
        g.write(str(user_info))

    dialog_history_1_1,dialog_history_1_2,dialog_history_2_1,dialog_history_2_2,model_1,model_2,persona_1,persona_2,ai_opinion_1,ai_opinion_2,entry_id,progress,choice = init_system(user_ID)
    return dialog_history_1_1,dialog_history_1_2,dialog_history_2_1,dialog_history_2_2,model_1,model_2,persona_1,persona_2,ai_opinion_1,ai_opinion_2,entry_id,progress,choice
    
def random_model(model_list):
    random.seed(time.time())
    [model_1, model_2] = random.sample(model_list,2)
    return model_1,model_2

def init_system(user_ID):
    '''
    初始化测评界面
    '''
    with open(user_info_path+f"/{user_ID}.txt","r",encoding='utf-8') as f:
        user_info = int(f.read())
        progress = str(user_info)+"/400"
    entry = dataset[user_ID][int(progress.split('/')[0])]
    persona_1 = entry['persona_left']
    persona_2 = entry['persona_right']
    model_1 = entry["model_left"]
    model_2 = entry["model_right"]
    ai_opinion_1 = entry['ai_opinion_left']
    ai_opinion_2 = entry['ai_opinion_right']
    dialog_history_1_1 = entry["model_1_response_1"]
    dialog_history_1_2 = entry["model_1_response_2"]
    dialog_history_2_1 = entry["model_2_response_1"]
    dialog_history_2_2 = entry["model_2_response_2"]
    entry_id = entry["id"]
    with open(save_dir+f"/{user_ID}.jsonl",'r',encoding='utf-8') as file:
        lines = file.readlines()
    if user_info > len(lines)-1:
        choice = "Tie"
    else:
        choice = json.loads(lines[user_info])["choice"]
    return dialog_history_1_1,dialog_history_1_2,dialog_history_2_1,dialog_history_2_2,model_1,model_2,persona_1,persona_2,ai_opinion_1,ai_opinion_2,entry_id,progress,choice

def go_upward(user_ID):
    user_info = 0
    with open(user_info_path+f"/{user_ID}.txt","r",encoding='utf-8') as f:
        user_info = int(f.read())
        if user_info != 0:
            user_info -= 1
    with open(user_info_path+f"/{user_ID}.txt","w",encoding='utf-8') as g:
        g.write(str(user_info))
    dialog_history_1_1,dialog_history_1_2,dialog_history_2_1,dialog_history_2_2,model_1,model_2,persona_1,persona_2,ai_opinion_1,ai_opinion_2,entry_id,progress,choice = init_system(user_ID)
    return dialog_history_1_1,dialog_history_1_2,dialog_history_2_1,dialog_history_2_2,model_1,model_2,persona_1,persona_2,ai_opinion_1,ai_opinion_2,entry_id,progress,choice
    

css = '''
.chatbot{
    font-size: 10px;
}

'''

with gr.Blocks(css=css) as demo:
    with gr.Row():
        user_ID = gr.Text(label = 'user id', value=f"", interactive=True)
        progress = gr.Text(label="process",value = "0/400")
        start = gr.Button("start annotation")
    with gr.Row():
        chatbot_1 = gr.Chatbot(label='model 1 response 1',elem_classes="chatbot")
        chatbot_2 = gr.Chatbot(label='model 1 response 2',elem_classes="chatbot")
        chatbot_3 = gr.Chatbot(label='model 2 response 1',elem_classes="chatbot")
        chatbot_4 = gr.Chatbot(label='model 2 response 2',elem_classes="chatbot")
    with gr.Row():
        ai_opinion_1 = gr.Textbox(label="AI_Opinion_1",interactive=False)
        ai_opinion_2 = gr.Textbox(label="AI_Opinion_2",interactive=False)
    with gr.Row():
        with gr.Column():
            with gr.Accordion("Your Preference"):
                with gr.Row():
                    sampled_persona = gr.Textbox(label="user preference_1",max_lines=50,value='',visible=True)
                    diffused_persona = gr.Textbox(label="user preference_2",max_lines=50,value='',visible=True)
                    model_1 = gr.Textbox(visible=False)
                    model_2 = gr.Textbox(visible=False)
                    entry_id = gr.Textbox(visible=False)
    with gr.Row():
        clear = gr.ClearButton([chatbot_1,chatbot_2])
        save = gr.Button(value="previous entry")
        # eval = gr.Button("Eval")
    with gr.Row():
        choice = gr.Radio(["Left","Right","Tie"],label="Make your choice",info="Which model do you think is better?",value="Tie",scale=5)
        submit = gr.Button("commit")
    with gr.Row():
        eval_result_1 = gr.Markdown()
        eval_result_2 = gr.Markdown()
    start.click(init_system,[user_ID],[chatbot_1,chatbot_2,chatbot_3,chatbot_4,model_1,model_2,sampled_persona,diffused_persona,ai_opinion_1,ai_opinion_2,entry_id,progress,choice])
    submit.click(align_history,[choice,progress,model_1,model_2,entry_id,user_ID],[chatbot_1,chatbot_2,chatbot_3,chatbot_4,model_1,model_2,sampled_persona,diffused_persona,ai_opinion_1,ai_opinion_2,entry_id,progress,choice])
    save.click(go_upward,[user_ID],[chatbot_1,chatbot_2,chatbot_3,chatbot_4,model_1,model_2,sampled_persona,diffused_persona,ai_opinion_1,ai_opinion_2,entry_id,progress,choice])
demo.queue()
demo.launch()