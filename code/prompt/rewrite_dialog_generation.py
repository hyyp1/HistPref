rewrite_dialog_generation_prompt = '''
Your task is to generate a two-round dialogue between a user and a chatbot based on the persona information that provided bellow.
Input:
    -persona information
output:
    -new dialog
The dialogue you generate should take the following form:
1. The user give a query about some topic.
2. The chatbot responds an answer that does not correspond to the user's persona information.
3. The user is not satisfied with chatbot's answers, and give some instruction to change the answer
4. The chatbot gives answers that match the user's perference.
Notice:
1. You'll start a whole new conversation, pick a topic and move quickly into a meaningful discussion, avoiding non-content conversations like greetings.
2. The queries you generate should be colloquial and closer to real users. 
For example, if your persona is "User is familiar with geographic topics", you can say "I am very familiar with what you are talking about, can you tell me something new or in-depth?"
Your instructions should be given based on persona, not following examples. If persona is None, you can freely generate dialogues that do not involve preferences, such as solving math problems, writing code, and other similar tasks.
3. The answers you generate should be longer and detailed, which is like a AI chatbot.
Here is some examples to make you understand this better:
persona: The user is in his mid-thirties.
BAD one:Can you generate responses more in line with the preferences of a man in his mid-thirties?
GOOD one: I think you should take my age into consideration. I'm in my 30s
persona: The user is knowledgeable about microcontrollers.
BAD one: Can you modify your answer to fit the needs of someone who knows a lot about microcontrollers?
GOOD one: Oh! I know quite a bit about microprocessors, let's talk about them in depth!

4. In at least one conversation, you need to accurately convey your persona in the user queries, and you can change the way you express it to make it more colloquial.

your output should in the following format:
# user:
user_query_1
# chatbot:
chatbot_answer_1
# user:
user_query_2
# chatbot:
chatbot_answer_2

[Begin Persona Information]
{persona}
[End Persona Information]
Now give your output:
'''