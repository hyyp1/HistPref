answer_generation_prompt = '''
You are a helpful chatbot. Your task is to generate a response to the query based on the dialog history. You need to carefully consider the user's persona from the dialog history. persona may contain user profiles or behaviors that the user expects the conversation assistant to exhibit (such as reply style, etc.). Please consider the user's persona when responding and generate responses that match the user's preferences.
Instructions you must follow:
1. The user profile embodied in the conversation may be relevant to the response of the current query, but the association may be implicit and require some prior information.
2. Please pay special attention to past user queries, topics previously discussed, and requirements that users have previously posed to the chatbot. These should be the main focus for you to obtain information about user preferences.
3. User queries may contain a certain degree of ambiguity. At this point, you only need to generate content according to the instructions without asking for additional information.
You must output in the following format:
persona: Here is the user persona you extract from dialog history.
Answer: Here is the answer of the current query considering the user persona. 

For example:
query: Can you recommend some restaurants that I might like?
your output:

persona: The user likes chicken rather than beef.
Answer: Considering your interest in chicken, I would like to recommend ...

Here is the true user query:
{query}
Now give your output:
'''

CoT_answer_generation_prompt = '''
You are a helpful chatbot. Your task is to generate a response to the query based on the dialog history. You need to carefully consider the user's persona from the dialog history. persona may contain user profiles or behaviors that the user expects the conversation assistant to exhibit (such as reply style, etc.). Please consider the user's persona when responding and generate responses that match the user's preferences.
Instructions you must follow:
1. User queries may contain a certain degree of ambiguity. At this point, you only need to generate content according to the instructions without asking for additional information.
2. Integrate the preference information into your answer content. Sentences like "Considering your preference for..." or "Considering your interest in" are not allowed to appear in the responses. You need to think about how to integrate persona information into specific content.
You must output in the following format:
step by step reasoning
Answer: Here is the answer of the current query considering the user persona. 

In step-by-step reasoning, you can write down the thought process for generating response based on the conversation history and query. Then summarize it, and provide your response starting with "Answer: "
Here is the user query:
{query}
Now give your output:
'''

TD_answer_generation_prompt = '''
You are a helpful chatbot. Your task is to generate a response to the query based on the dialog history. You need to carefully consider the user's persona from the dialog history. persona may contain user profiles or behaviors that the user expects the conversation assistant to exhibit (such as reply style, etc.). Please consider the user's persona when responding and generate responses that match the user's preferences.
Instructions you must follow:
1. User queries may contain a certain degree of ambiguity. At this point, you only need to generate content according to the instructions without asking for additional information.
Here is the true user query:
{query}
Now give your answer:
'''