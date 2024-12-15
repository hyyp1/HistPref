from openai import OpenAI

class GPTAPIModel:
    def __init__(self, api_key, base_url, model_name):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name

    def generate_response(self, system_prompt,user_prompt,dialog_history):
        client = OpenAI(api_key=self.api_key,
                        base_url=self.base_url)
        messages = dialog_history+[{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}]+messages
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=1
        )
        return response.choices[0].message.content
    
    def get_embeddings(self, sentences):
        client = OpenAI(api_key=self.api_key,
                        base_url=self.base_url)
        response = client.embeddings.create(
            input=sentences,
            model=self.model_name
        )
        embeddings = [item.embedding for item in response.data]
        return embeddings
