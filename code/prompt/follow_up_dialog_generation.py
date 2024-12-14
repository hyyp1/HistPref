follow_up_dialog_generation_prompt = '''
Your task is to generate a two-round dialogue between a user and a chatbot based on the persona information that provided bellow.
Input:
    -persona information
    -dialog history
output:
    -new dialog
In the query section, you should act as a persona user using the chatbot, and in the answer section, you should be a chatbot responding to the user's query. The topics of conversation can be diverse, as long as you believe the user might be interested in them.
Notice:
1. You'll start a whole new conversation, pick a topic and move quickly into a meaningful discussion, avoiding non-content conversations like greetings.
2. The queries you generate should be colloquial and closer to real users. 
For example, if your persona is "User is familiar with geographic topics", you can say "I am very familiar with what you are talking about, can you tell me something new or in-depth?"
Your instructions should be given based on persona, not following examples. If persona is None, you can freely generate dialogues that do not involve preferences, such as solving math problems, writing code, and other similar tasks.
3. The answers you generate should be longer and detailed, which is like a AI chatbot.
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