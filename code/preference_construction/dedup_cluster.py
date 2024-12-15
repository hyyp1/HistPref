import json
import os
import config
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ..config import config
from ..models.pref_model import GPTAPIModel

class GPT4:
    def __init__(self) -> None:
        self.gpt_model = GPTAPIModel(config.API_KEY, config.BASE_URL, "gpt-4o")

    def chat(self,user_prompt):
        return self.gpt_model.generate_response('',user_prompt,[])

class GPT4_EMBED:
    def __init__(self) -> None:
        self.gpt_model = GPTAPIModel(config.API_KEY, config.BASE_URL, "text-embedding-ada-002")

    def embed(self,sentences):
        return self.gpt_model.get_embeddings(sentences)
    
preferences = []
dirs = ["HistPref/data/sampled_real_data/oiyZw6a_i7ZsUGB7SJB8b4JToOOg.json"]
for dir in dirs:
    for file in [f for f in os.listdir(dir) if f.endswith(".json")]:
        conv = json.load(open(os.path.join(dir, file)))
        for chunk in conv:
            for dialog in chunk["chunks"]:
                preferences += dialog["personal_profile"] + dialog["response_format"]

embedllm = GPT4_EMBED()
embeddings = embedllm.embed(preferences)
embeddings_array = np.array(embeddings)
similarity_matrix = cosine_similarity(embeddings)
threshold = 0.9
unique_preferences = []
used_indices = set()

for i in range(len(preferences)):
    if i not in used_indices:
        unique_preferences.append(preferences[i])
        similar_indices = np.where(similarity_matrix[i] > threshold)[0]
        used_indices.update(similar_indices)

types = ["Demographic information: Age group(The age range of the user may be Children, Teenagers, Young Adults, Middle-aged Adults, Seniors)",
         "Demographic information: Gender(The user's gender can be male, female, or neutral)",
         "Demographic information: Family situation(User's family situation, including family members, relationships, and atmosphere.)",
         "Demographic information: Occupation(Occupation, job, status, position of the user)",
         "Financial information: Income level",
         "Economic information: Consumption habits",
         "Personality traits: Social tendency",
         "Personality traits: Decision-making style",
         "Interactive style: Content style preference",
         "Interactive style: Language style preference",
         "Interactive style: Emotional preference",
         "Lifestyle: Life rhythm style",
         "Lifestyle: Hobbies",
         "Lifestyle: Dietary preferences",
         "Knowledge skill: Knowledge level",
         "Knowledge and skills: Areas of expertise",
         "Knowledge and skills: Skills and expertise"]

llm = GPT4()
PROMPT = """
Please determine which of the following topics does the preference({preference}) belong to?
{types}
Output only one category sequence number.
"""

res = {}
for p in unique_preferences:
    input_prompt = PROMPT.format(preference=p, types=[f"{i}. {t}" for i,t in enumerate(types)])
    response = llm.chat(input_prompt).split(".")[0]
    print(response)
    try :
        if types[int(response)] not in res:
            res[types[int(response)]] = []
        res[types[int(response)]].append(p)
    except:
        continue

json.dump(res,open("HistPref/data/log/preferences/preferences.json","w"),indent=4,ensure_ascii=False)