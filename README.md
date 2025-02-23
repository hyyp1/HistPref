## Anonymous Repository for Personas Influence Deeply: Enhancing Interaction with Large Language Models via Inference-time Alignment

### asserts:

All the figures and tables appeared in our paper.

### code:

The source code of our paper, including data construction scripts, prompts, training scripts, etc.

### data:

├── dataset

├── log

└── sample_real_data

* dataset: The main data of InterPref, including `history_map.jsonl`, which contains the interaction history corresponding to each preference; `sharegpt_pairs.json`, our confidential dataset containing real dialogues; `train_dataset.jsonl` and `test_dataset.jsonl`, which represent our training and test dataset splits.
  The meanings of the keys in the training and testing datasets are as follows:
  * profile_personas: The personas of the users described in the text.
  * diffused_personas: Another set of personas described in the text.
  * profile_history: The interaction history that reflects the profile_personas.
  * diffused_history: The interaction history that reflects the diffused_personas.
  * query: The current query provided to the model.
* log: The running results of all models in our experiment.
