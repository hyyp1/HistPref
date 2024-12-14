query_generation_prompt = '''
Your task is to generate three tasks that simulate a scenario where an AI chatbot user requires assistance from the chatbot, based on the persona information and the dialogue history.Your tasks should meet the following requirements.
Requirements:
-Your task should be based on specific needs in life or work.
-You will be provided two different persona information,and your task should focused on the difference between them.This means that the person with different persona information will do your task differently,and your task should maximize this difference. 
-Do not mention user preferences directly in the task.Here is an example to demonstrate this:
[Begin Example]
Persona information 1:
{{"persona":"The user is familiar with English grammar and vocabulary exercises."}}
Persona information 2:
{{"persona":"The user is familiar with Spanish grammar and vocabulary exercises"}}
Bad answer: 
```json
["Create a daily practice plan for improving English vocabulary.","Help me write an email in Spanish for a job application.","Suggest learning resources for mastering English grammar."] 
```
Each of these tasks directly refers to a user preference,which is unbalanced. 
Good answer:
["Create a daily practice plan for improving language ability.","Help me write an email in my familiar language for a job application.","Suggest learning resources for mastering the language I'm learning."] 
[End Example]
Input:
    -two different Persona information
    -Dialog history
Output:
    -tasks

Here is some examples:
[Begin example]
Persona information 1:
{{"persona":"The user shows curiosity about food and its preparation."}}
Persona information 2:
{{"persona":"The user shows interest in travel and different cultures."}}
Your Output:
```json
["Make a plan for my weekend.","Help me make a list of must-do's for a trip to Chengdu.","Develop a travel itinerary for a one-week journey."] 
```
[End example]
[Begin example]
Persona information 1:
{{"persona":"The user is interested in and has knowledge about game development using Unity3D."}}
Persona information 2:
{{"persona":"The user is interested in and has knowledge about web development using React."}}
Your Output:
```json
["Give some advice on the choice of holiday temporary workers","I want to work in the Internet industry and make a list of what technologies to know","Recommend a programming language that I should learn more about"]
[End example]


[Begin Persona Information 1]
{persona}
[End Persona Information 1]
[Begin Persona Information 2]
{diffused_persona}
[End Persona Information 2]
[Begin Dialog history]
{dialog_history}
[End Dialog history]
remember to output ```json and ``` like this:
```json
["task1","task2","task3"]
```
Now give your question:
'''

fake_user_generation_prompt = """
You are a user of a chatbot. You will receive specific persona information about the user and the history of the current conversation. Your task is to impersonate the user and give a user query to continue the conversation in a way that is consistent with the user's persona information. For some persona information that is not easy to represent, you can construct some questions and clearly express the user's persona information. The dialog history could be empty, just start a new conversation.
Here is the user's persona information
{persona}
"""

query_refine_prompt = """
You will see two queries, and your task is to distill the commonalities between them and generate a general query. In most cases, the contents of the two queries are the same, and the output is fine as is. In the rare case that a query contains unnecessary additional information, you should filter it out. In rare cases where the contents of two queries are completely different, try to get a general query from the first query.
[example_1]
query_1: Can you help me develop a strategy to engage the local community with our project?
query_2: Could you help me develop a strategy to engage the local community with our project?
output: 
```Can you help me develop a strategy to engage the local community with our project?```
[example_2]
query_1:Can you help me suggest some topics for a research paper?
query_2:Can you help me suggest topics for a research paper focused on urbanization?
output:
```Can you help me suggest some topics for a research paper?```
[example_3]
query_1:Hi, could you help me choose a new gift for a friend who loves the latest tech and gadgets?
query_2:Can you help me choose a gift for a fashion-conscious friend?
output:```Can you help me choose a gift for a friend?```
Remember to print ``` in your output.
Here is the querys:
query_1:{query_1}
query_2:{query_2}
Now give your output!
"""