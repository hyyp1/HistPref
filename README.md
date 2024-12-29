### Anonymous Repository for Large Language Models as User Preference Learner in Dialogue

#### asserts:

All the figures and tables appearing in our paper.

#### code:

The source code of our paper, including data construction scripts, prompts, training scripts, etc.

#### data:

├── dataset
├── log
└── sample_real_data

* dataset: The main data of HistPref, including `history_map.jsonl`, which contains the dialogue history corresponding to each preference; `sharegpt_pairs.json`, our confidential dataset containing real dialogues; `train_dataset.jsonl` and `test_dataset.jsonl`, which represent our training and test dataset splits.
* log: The inference results of all models in the paper.
