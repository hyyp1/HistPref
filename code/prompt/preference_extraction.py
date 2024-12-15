preference_extraction_prompt = """
Now your goal is to accomplish topic discovery and user preference extraction task from a long dialog history. Let's beirfly describe your task:

1) Input: You will be presented with a long, multiturn dialog between a user and an assistant. You can only see the user's queries.
2) Your task: 
a) Firstly, you need to partition the whole dialog into multiple chunks by topics: consecutive dialogs about same topic should be put into the same chunk.
- Every dialog within the same chunk is about the same topic and discusses the same matter!!
- The chunk is represented by two [Dialog ID] shown in the dialog history, so the messages between these two [Dialog ID] is a chunk.
- The chunk should begin with a query INDEPENDENT of previous dialog content, which makes the following dialog distinct from previous dialog. 
- If a topic has fewer than three dialog turns, do not consider it.
b) Next, for each recommended chunk, generate the following content with English: 
i) Give a BRIEF topic of this dialog chunk. (`topic`)
ii) Give the beginning and end dialog ID, be accurate. (`begin_dialog_id` and `end_dialog_id`)
iii) Extract the user's some key personal information, such as location, job details, interests and hobbies, family background, health status and so on FROM dialogs in this chunk: (`personal_profile`)
iv) Extract how formal or casual the assistant's response should be, how long or short responses should generally be, and what type of solutions or information the user prefers to receive FROM dialogs in this chunk: (`response_format`)


Output your response in this format. 

------------

```json
{{"chunks":[
{{
"begin_dialog_id": xxx,
"end_dialog_id": xxx,
"topic": "here is the topic of this chunk",
"personal_profile": ["xxx","xxx",...],
"response_format": ["xxx","xxx",...]
}},
{{
"begin_dialog_id": xxx,
"end_dialog_id": xxx,
"topic": "here is the topic of this chunk",
"personal_profile": ["xxx","xxx",...],
"response_format": ["xxx","xxx",...]
}}
]}}
```
------------

Give you three examples for reference:

Here is the dialogue history of the input for the first example:

[Dialog ID]: 0
[User]: hi
 
[Dialog ID]: 1
[User]: hi
 
[Dialog ID]: 2
[User]: how to pick a study abroad agency?
 
[Dialog ID]: 3
[User]: I want to find a study abroad agent, how to choose
 
[Dialog ID]: 4
[User]: 3d medical image stitching, stitching based on the continuity of the structure, to align, what are the possible ideas?
 
[Dialog ID]: 5
[User]: Since each piece of my image is 150*3000*3000, and there are only 50 pieces in total, how can I achieve this based on deep learning? The problem is no network could eat a tensor that big, right? For your reference

[Dialog ID]: 6
[User]: My idea is to pre-train a Masked autoencoder that can encode 32*32*32 images, and then finetune this MAE to train a Mosaic quality reward model. It is then used for optimization based splicing. What do you think? Analyze the possible difficulties, MAE has been trained and come up with a more feasible plan.

[Dialog ID]: 7
[User]: I trained another larger masked autoencoder, 64*64*64, which is good for large-scale object reconstruction

[Dialog ID]: 8
[User]: What good, highly actionable insights do you have about the construction of a splice reward model
 
[Dialog ID]: 9
[User]: What are rigid and non-rigid transformations in computer vision/image processing, explain that I am a beginner
 
[Dialog ID]: 10
[User]: The data I have obtained is obtained by imaging the slices of mouse brain cut out by a slicing machine. It is said that the loss between the two slices is about 1 micron, and the thickness of each brain slice is 300 microns. Will this situation produce non-rigid transformation? The microtome vibrates at high speed

[Dialog ID]: 11
[User]: pytorch uses non-rigid deformation field to realize the splicing of two (150, 300, 300) tensor brain slices. The deformation field is variable and can be changed, so it is OK to write a prototype. In the future, some loss needs to be calculated and backpropagation optimization.

[Dialog ID]: 12
[User]: You mean keep slice2 the same, and then only change slice1, right, and then what's the loss, assuming loss is an nn.Module, complete the code, including backpropagation

[Dialog ID]: 13
[User]: How exactly is affine_grid implemented, the principle and motivation, and is it a linear transformation?

[Dialog ID]: 14
[User]: I would like to ask if there is anything in torch that can implement a similar "surface" transformation
 
[Dialog ID]: 15
[User]: How to generate a nonlinear deformation field randomly
 
[Dialog ID]: 16
[User]: For such a nonlinear deformation field, how to optimize by calculating my given loss and write a complete optimization code

[Dialog ID]: 17
[User]: I want to also make sure that this deformation field is smooth, as you just pointed out
 
[Dialog ID]: 18
[User]: After slice1 transformation, how to combine two 3D images to calculate loss? discuss
 
[Dialog ID]: 19
[User]: How do you deal with the coordinates of two 3D images? Any uniform coordinates?
 
[Dialog ID]: 20
[User]: How do you implement global coordinates in pytorch? One of my ideas is to create a (300,3000,3000) coordinate system, two (150,3000,3000) stack, then let slice1 position unchanged, and then slice2 after nonlinear deformation to write (300,3000,3000) coordinate system, is this feasible? If you have a better suggestion, please give it to me and implement it

[Dialog ID]: 21
[User]: Hello, I want to travel this weekend, please recommend some places for me


[Dialog ID]: 22
[User]: I work in Beijing. I recommend places around Beijing

[Dialog ID]: 23
[User]: I don't like sports very much, is there a place to play less exercise

[Dialog ID]: 24
[User]: Can you visit all these places in one day

[Dialog ID]: 25
[User]: Is Wat Pho crowded on weekends

 
Here is the output response of the first example:
 
```json
{{"chunks":[
{{
"begin_dialog_id": 4,
"end_dialog_id": 8,
"topic": "Image Processing",
"personal_profile": ["The user has a basic understanding of computers."],
"response_format": ["The user prefers to receive highly feasible solutions."]
}},
{{
"begin_dialog_id": 9,
"end_dialog_id": 20,
"topic": "Computer Vision",
"personal_profile": ["The user has a basic understanding of computers.","The user is a beginner in the field of computer vision/image processing."],
"response_format": []
}},
{{
"begin_dialog_id": 21,
"end_dialog_id": 25,
"topic": "Travel Recommendation",
"personal_profile": ["The user works in Beijing.","The user does not like sports."],
"response_format": []
}}
]}}
```

Here is the dialogue history of the input for the second example:

[Dialog ID]: 0
[User]: If I have a headache, what might be causing it
 
[Dialog ID]: 1
[User]: What are the ways to relieve migraines, please list one by one
 
[Dialog ID]: 2
[User]: Where do I need to press for a massage
 
[Dialog ID]: 3
[User]: What medications can be used to treat it
 
[Dialog ID]: 4
[User]: Tell a joke
 
[Dialog ID]: 5
[User]: Ahh... It's not funny.
 
[Dialog ID]: 6
[User]: What problems should be paid attention to when climbing the mountain? Please list them one by one
 
[Dialog ID]: 7
[User]: Please list what food you can take with you on the mountain
 
[Dialog ID]: 8
[User]: In addition to the above, what do you need to pay special attention to mountain climbing in summer
 
[Dialog ID]: 9
[User]: Please tell me some first aid measures for heat stroke
 
[Dialog ID]: 10
[User]: What if you get lost with your friends while climbing
 
[Dialog ID]: 11
[User]: Okay, thank you

 
Here is the output response of the second example:
 
```json
{{"chunks":[
{{
"begin_dialog_id": 0,
"end_dialog_id": 3,
"topic": "Health Consultation",
"personal_profile": [],
"response_format": ["The user wants answers listed point by point."]
}},
{{
"begin_dialog_id": 6,
"end_dialog_id": 11,
"topic": "Mountain Climbing Precautions",
"personal_profile": [],
"response_format": ["The user wants answers listed point by point."]
}}
]}}
```

Here is the dialogue history of the input for the third example:

[Dialog ID]: 0
[User]: Recently, I have some spare money, and I want to do some investment and financial management, but I don't know how to start. Do you have any suggestions?
 
[Dialog ID]: 1
[User]: I mainly want to save some money for my children's education, the goal is about ten years. Personally, I prefer a sound investment and don't like to take risks very much. While the returns may not be as high, I value the safety of my money more.

[Dialog ID]: 2
[User]: It is best to control it within 10%, of course, if it can be more robust, about 5% is better. I'm usually busy at work, so I probably don't have the energy to constantly monitor and adjust my portfolio, and I don't want to devote too much time to research

[Dialog ID]: 3
[User]: Will it rain tomorrow
 
[Dialog ID]: 4
[User]: Plan a route from Chengdu to Lhasa

 
Here is the output response of the third example:
 
```json
{{"chunks":[
{{
"begin_dialog_id": 0,
"end_dialog_id": 2,
"topic": "Investment / Financing",
"personal_profile": ["The user values family and has school-aged children.", "The user has a stable financial situation with some savings.", "The user has a busy work schedule and a fast-paced lifestyle."],
"response_format": ["The user prefers a cautious decision-making style.", "The user prefers content that is hassle-free and easy to manage."]
}}
]}}
```

Here is the dialog history:
{dialog_formatted}


Now, please understand the examples and give your response to the task instruction. Remember, only output ```json```!!
Furthermore, please make sure to think carefully. 
- When providing chunks, please ensure that every dialog within the same chunk is about the same topic and discusses the same matter.
"""