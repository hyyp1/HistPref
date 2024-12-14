import abc
from ..models.dialog_model import GPTAPIModel
from ..config import config
import re
import json


PERSONA_EXTRACTION_PROMPT = '''
You are in a workflow of generating personalized responses to the user query based on dialogue history. This workflow is roughly divided into three steps: 1. Extracting useful persona information about the user from the conversation history, 2. Generating instructions to guide the creation of the response, 3. Producing the response based on the instructions.
You are currently at the first step of this workflow, which is to extract useful persona information from the conversation history. You will see a user query and a segment of conversation history. You will only see the user queries because they are more relevant to persona information.
You need to extract the persona information that you consider useful from the conversation history and output it in the following format:
<persona>your_answer</persona>

Here is an example of your output:
<persona>The user is familiar with or interested in culinary topics, specifically street food. The user values polite and respectful communication.</persona>

Here are some basic instructions:
1. At this step, you should extract as detailed and diverse persona information as possible. This includes not only the user's preferences and interests but also their communication style, values, and any other relevant details that can help in tailoring the response to align with the user's expectations and personality.
2. Pay attention to <persona> and </persona>, don't forget to output them.


Here is the input:
# history_query
{history_query}
# current_query
{query}

Now,give your response:
'''

PERSONA_FILTERING_PROMPT = '''
You are in a workflow of generating personalized responses to the user query based on dialogue history. This workflow is roughly divided into three steps: 1. Extracting useful persona information about the user from the conversation history, 2. Generating instructions to guide the creation of the response, 3. Producing the response based on the instructions.
You are currently at the first step of this workflow, which is to extract useful persona information from the conversation history. The extraction is finished, and you only need to focuse on "USEFUL". Here is your task:
You will see a user query and some extracted persona information, you need to keep useful personas and remove personas that is useless. You should first ensure that all relevant personas are not eliminated. For example, when the query is about recommendation content, the user's personal information is important. When the query is about letter writing, the style of writing is important. But when the topic is about diet, the user's hobbies in other areas are not as relevant.
You should follow this format:
<persona>useful_persona_1</persona>
<persona>userful_persona_2</persona>

Here is an example of your output:
<persona>The user is a ten-year-old boy, considering to go to his friends party.</persona>
<persona>The user is familiar with or interested in culinary topics, specifically street food.</persona>
<persona>The user values polite and respectful communication.</persona>
Here is the queries:
{query}
Here is the personas:
{persona}
Now give your response:
'''

ADVICE_GENERATE_PROMPT = '''
You are in a workflow of generating personalized responses to the user query based on dialogue history. This workflow is roughly divided into three steps: 1. Extracting useful persona information about the user from the conversation history, 2. Generating instructions to guide the creation of the response, 3. Producing the response based on the instructions.
You are currently at the second step of this workflow, which is generating instructions to guide the response. You will see a user query and a related persona, you need to generate an instruction to guide the subsequent response, which may include: what style to adopt for the reply, how to embody this style, what to mention in the reply based on the user's persona, and what to pay attention to when crafting the reply, etc. You should generate brief instructions focusing only on the persona you are seeing now.
You should follow this format:
<instruction>your_instruction</instruction>
Here is an example:
persona: Users prefer to spend a large amount of money on their hobbies.
query: Can you make a financial plan for me?
<instruction>I should allocate a larger proportion of money to the user's hobbies when making financial plans.</instruction>
Here is the user query:
{query}
Here is the related persona:
{persona}
Now give your output:
'''

ANSWER_GENERATE_PROMPT = '''
You are in a workflow of generating personalized responses to the user query based on dialogue history. This workflow is roughly divided into three steps: 1. Extracting useful persona information about the user from the conversation history, 2. Generating instructions to guide the creation of the response, 3. Producing the response based on the instructions.
You are currently at the third step of this workflow, which is generating response based on the instructions. Please carefully consider each instruction and ensure that every instruction is reflected in your response. You should emphasize the parts related to the persona in your response, so that the user can see that you have taken their persona into account. For example, use phrases such as "Considering that you are a..." to highlight the personalization.
Here is the user query:
{query}
And here is the instructions:
{instructions}
Now give your answer to the user query:
'''

ANSWER_REFINE_PROMPT = '''
You will see a user query, the corresponding response, and an instruction. You need to consider whether the response has fully taken the instruction into account. If it has, then leave the response as is. If not, give some comment in order to make it more in line with the instruction.
To avoid affecting the other response original content, you cannot make extensive changes to the response.
Here is the user query:
{query}
Here is the response:
{response}
And here is the instruction:
{instruction}

your output should in the following format:
<new>your new comment</new>
Here is some example:
if you think there is no additional comment:
<new>None</new>
otherwise:
<new>your comment</new>
Now give your output:
'''


AI_OPINION_PROMPT = '''
You will be given two different persona messages and two corresponding responses to a certain question. Under normal circumstances, the questions of the two answers would be essentially the same, and the answers would take into account the different preferences of the inquirer. Your task is to distinguish which answer corresponds to persona_1 and which answer corresponds to persona_2. 
Here is some tips to help you:
1. When a response has words that are in one persona but not in another persona,  you can assume that the response corresponds to the persona that contains that word. Note that this case requires that the word that appears in the response must be exactly the same as the word that appears in persona. That is to say, this case only involves word matching and does not involve any semantic information. When you decide that the input belongs to the case, you need to indicate in "reasoning" which identical word you find. This is the most simple case, labeled [direct content relevant]
2. When a response contains words related to one persona but not to another persona, you can assume that the response corresponds to the relevant persona. Note that relevance and indifference should be relative. For example, if persona_1 appears python and persona_2 appears poem, then program language is related to persona_1. But if persona_2 appears in java, there is no way to tell which side is relevant with the program language. Making such judgments relies more on your own prior knowledge. As long as two words are not exactly the same word, they are this case. This case is labeled [implicit content relevant]
3. In some cases, persona stands for the desired output leaning toward a style, such as "Users prefer humorous responses."In this case, you should carefully read the two dialogues and make a distinction based on the style of the dialogue tendencies. This case is labeled [format relevant]
4. When you think that the current case does not fit the two categories described above, you need more complex logical reasoning to determine the answer. You can look at the logical connection of the content, the degree of detail and the order of the narrative or in more complex ways than just the examples I've given. This case is labeled [logical relevant]
Output format:
# reasoning
content
# response corresponds to persona_1
content
# case_label
content
A detailed description of each key is below:
"reasoning": The logical reasoning you make. You need to list all the difference you find that is relevant to the persona.
"response corresponds to persona_1": The choice you make, choose between "Left","Right" and "Don't know"
"case_label": The label you think this case should belong to, choose between "direct content relevant","implicit content relevant", "format relevant", "logical relevant" and "inrelevant". if you set "response corresponds to persona_1" to "Don't know", then this value should be "inrelevant".
Here is some example:
[Example 1]
persona_1:The user is connected to the arts community as a member of the Hebei Painters Association.
persona_2:The user is connected to the technology community as a member of the Hebei Software Developers Association.
response_Left:Of course! Here are some upcoming workshops and seminars that might interest you:\n1. 'Color and Composition in Landscape Painting' - A workshop focusing on advanced techniques for landscape art.
response_Right:Of course! Here are some upcoming workshops and seminars that might interest you:\n\n1. \"Innovative Software Development Techniques\" – A workshop focusing on the latest trends and methods in software engineering.
output:
# reasoning
The recommendation in response_Left starts with 'Color and Composition in Landscape Painting', which is not found on the right. conversly, recommendations begin with Innovative Software Development Techniques, an option not found on the left.
The word 'Color','Painting' and 'Art' appears in the response_Left and the word 'software' appears in the response_Right, so I assume that response_Left corresponds to persona_1 and response_Right corresponds to persona_2
# response corresponds to persona_1
Left
# case_label
direct content relevant
[Example 2]
persona_1:The user is likely involved professionally in a field that requires the application of machine learning to large-scale data.
persona_2:The user is likely involved professionally in a field that requires the application of statistics to large-scale data.
response_Left:Of course! Here is a step-by-step approach to designing a predictive model for accurate sales forecasting:\n\n1. **Data Collection and Preparation**\n  **Exploratory Data Analysis (EDA)**\n   - Perform a thorough analysis to understand data distributions and identify patterns.\n   - Create visualizations (e.g., time series plots, histograms) to gain insights into the data.\n\n **Model Selection and Training**\n   - Select appropriate models (e.g., ARIMA, linear regression) for initial analysis.
response_Right:Of course! Here's a step-by-step guide to designing a predictive model to forecast sales:\n\n1. **Data Collection**: Gather historical sales data, including relevant features like dates, product categories, pricing, promotions, economic indicators, and other external factors.\n\n **Model Selection**: Choose a model suitable for time series forecasting such as ARIMA, Prophet, or more advanced models like Random Forest or boosting algorithms.
output:
# reasoning
I found word 'histograms' and 'linear regression' in response_Left,and 'Random Forest','boosting algorithms' in response_Right
The word 'histograms' or 'linear regression' is more related to statistics in persona_2,and the word 'Random Forest' or 'boosting algorithms' is more related to machine learning in persona_1
# response corresponds to persona_1
Right
# case_label
implicit content relevant
Here is the real input:
persona_1:{persona_1}
persona_2:{persona_2}
response_Left:{Left}
response_Right:{Right}
Now give you output:
'''


class Agent:
    def __init__(self,dialog_model = None) -> None:
        if dialog_model is not None:
            self.dialog_model = dialog_model
        else:
            self.dialog_model = GPTAPIModel(config.API_KEY,config.BASE_URL,config.GPT_MODEL_NAME)

    def extract_query_history(self,dialog_history):
        query_history = ""
        for dialog in dialog_history:
            query_history += dialog[0]
            query_history+= '\n'
        return query_history

    async def extract_persona(self,dialog_history,query):
        history_query = self.extract_query_history(dialog_history)
        user_prompt = PERSONA_EXTRACTION_PROMPT.format(history_query=history_query, query=query)
        response = await self.dialog_model.generate_response('',user_prompt,[])
        pattern = r"<persona>(.*?)</persona>"

        # 搜索文本
        match = re.search(pattern, response, re.DOTALL)

        # 如果找到匹配项，则提取并创建字典
        if match:
            persona = match.group(1).strip()
            return persona
        else:
            return "None"
        
    async def persona_filtering(self, query, personas):
        user_prompt = PERSONA_FILTERING_PROMPT.format(query=query,persona=personas)
        response = await self.dialog_model.generate_response('',user_prompt,[])
        pattern = r"<persona>(.*?)</persona>"
        persona_list = re.findall(pattern, response, re.DOTALL)
        if persona_list:
            return persona_list
        else:
            return [personas]
    
    async def advice_generation(self, query, persona_list):
        instruction_list = []
        if isinstance(persona_list,list):
            for persona in persona_list[0:10]:
                user_prompt = ADVICE_GENERATE_PROMPT.format(query=query, persona=persona)
                response = await self.dialog_model.generate_response('',user_prompt,[])
                pattern = r"<instruction>(.*?)</instruction>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    instruction = match.group(1).strip()
                    instruction_list.append(instruction)
                else:
                    instruction_list = persona_list
        else:
            instruction_list=persona_list
        return instruction_list
    
    async def answer_generation(self, query, instruction_list):
        user_prompt = ANSWER_GENERATE_PROMPT.format(query=query,instructions = str(instruction_list))
        response = await self.dialog_model.generate_response('',user_prompt,[])
        return response
        
    async def workflow(self, query, dialog_history,log_dir):
        extracted_persona = await self.extract_persona(dialog_history,query)
        persona_list = await self.persona_filtering(query,extracted_persona)
        instruction_list = await self.advice_generation(query,persona_list)
        response = await self.answer_generation(query,instruction_list)
        with open(log_dir,"a+",encoding='utf-8') as f:
            json.dump({
                "extracted_persona": extracted_persona,
                "persona_list": persona_list,
                "instruction_list": instruction_list,
                "response": response
            },f,separators=(',',':'),ensure_ascii=False)
            f.write('\n')
        return response


    
    async def generate_ai_opinion(self,response_1,response_2,persona_1,persona_2):
        user_prompt = AI_OPINION_PROMPT.format(persona_1=persona_1,persona_2=persona_2,Left=response_1,Right=response_2)
        response = await self.dialog_model.generate_response('',user_prompt,[])
        return response