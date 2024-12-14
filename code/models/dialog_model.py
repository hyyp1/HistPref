from openai import OpenAI,AsyncOpenAI
import transformers

class BaseModel:
    def generate_response(self, 
                          system_prompt : str,
                          user_prompt : str,
                          dialog_history : list,
                          temperature : float):
        return

class GPTAPIModel(BaseModel):
    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.client = AsyncOpenAI(api_key=self.api_key,
                        base_url=self.base_url)

    async def generate_response(self, 
                          system_prompt,
                          user_prompt,
                          dialog_history,
                          temperature = 0,
                          ):
        messages = dialog_history+[{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]+messages
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature= temperature,
            max_tokens=2048
        )
        response = response.model_dump()
        return response['choices'][0]['message']['content']
    
    def generate_stream_response(self, 
                          system_prompt,
                          user_prompt,
                          dialog_history,
                          temperature = 0,
                        ):
        messages = dialog_history+[{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]+messages
        response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature= None if temperature==0 else temperature,
                stream=True
            )
        collected_messages = ''
        for chunk in response:
            if chunk.choices[0].delta.content:
                collected_messages += chunk.choices[0].delta.content
                yield collected_messages

    def generate_api_response(self, 
                              system_prompt, 
                              user_prompt, 
                              dialog_history, 
                              tools=[], 
                              tool_choice='auto', 
                              temperature = 0.9):
        client = OpenAI(api_key=self.api_key,
                        base_url=self.base_url)
        messages = dialog_history+[{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]+messages
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature= None if temperature==0 else temperature,
            tools=tools,
            tool_choice=tool_choice
        ).model_dump()
        return response['choices'][0]['message']