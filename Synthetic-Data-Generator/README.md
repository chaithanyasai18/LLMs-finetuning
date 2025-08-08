# Synthetic Data Generator for LLM Finetuning

## Overview

This Python project generates synthetic datasets for fine-tuning Large Language Models (LLMs), with a focus on cybersecurity, cloud computing, and IT support. By leveraging topic trees and parallelized data generation, it helps you create diverse, high-quality question-answer pairs for training conversational AI.

---

## How It Works

The generator uses a two-step process:
1. **Topic Tree Generation:** A hierarchical tree of topics is built from a root prompt, ensuring broad coverage and diversity in the dataset.
2. **Data Generation:** For each topic path, the tool generates question-answer pairs using your chosen LLM, following your instructions and system prompt.

This approach prevents repetitive data and ensures your dataset covers a wide range of relevant scenarios.


---

## Python Environment Setup

It is recommended to use a virtual environment with Python 3.12.

### Using venv (standard Python):

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Using Conda:

```bash
conda create -n datagen python=3.12
conda activate datagen
pip install -r requirements.txt
```


## Features

- **Topic Tree Generation:** Ensures dataset diversity by covering many subtopics.
- **Parallel Data Generation:** Efficiently generates large datasets using batching.
- **Customizable Prompts:** Tailor the assistant's behavior and dataset focus.
- **Multi-Model Support:** Use any supported LLM provider.

---

## Installation

It is recommended to use a virtual environment with Python 3.12.

### Using venv (standard Python):

```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Using Conda:

```bash
conda create -n datagen python=3.12
conda activate datagen
pip install -r requirements.txt
```
---

Install the required package using pip:

```bash
pip install pluto-data
```

---

## Setting Up

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your-key
```

---

## Example: Understanding `app.py`

The `app.py` script demonstrates the complete workflow for generating a synthetic dataset. Below is a step-by-step explanation with code comments:

```python
from pluto import EngineArguments, DataEngine, Dataset, TopicTree, TopicTreeArguments

# Step 1: Set the system prompt for the AI model.
# This prompt guides the AI to act as a knowledgeable assistant in cybersecurity, cloud computing, and IT support.
system_prompt = (
    "You are a safe, helpful, knowledgeable AI assistant and customer support expert specializing in cyber security, "
    "cloud computing and IT Support domain. Your primary job is to deliver detailed responses to customer queries "
    "related to the field of cybersecurity, cloud computing and IT Support domain. Leveraging your deep knowledge and "
    "expertise, please generate unique questions and relevant answers to guarantee customer satisfaction without "
    "repeating same questions. If you don't know the answer to a question, please don't share false information."
)

# Step 2: Initialize a TopicTree object with specific arguments.
# The topic tree helps generate a diverse set of topics for data generation.
tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Cyber Threats and Attacks",  # The main subject for the topic tree.
        model_system_prompt=system_prompt,        # The system prompt for the AI model.
        tree_degree=10,                           # Max number of subtopics per node.
        tree_depth=2                              # Levels of subtopics.
    )
)

# Step 3: Build the topic tree using the specified AI model.
tree.build_tree(model_name="gpt-4o")

# Step 4: Save the generated topic tree to a JSONL file for later use.
tree.save("data/topictree_01.jsonl")

# Step 5: Initialize a DataEngine object with instructions and the system prompt.
engine = DataEngine(
    args=EngineArguments(
        instructions="Generate questions and answers related to Cyber Threats and Attacks.",  # Data generation instructions.
        system_prompt=system_prompt,  # The system prompt for the AI model.
    )
)

# Step 6: Generate the synthetic dataset using the DataEngine, topic tree, and model.
dataset = engine.create_data(
    model_name="gpt-4o",   # The AI model to use for generation.
    num_steps=10,          # Number of generation steps (controls dataset size).
    batch_size=5,          # Number of samples generated in parallel (10 steps × 5 = 50 samples).
    topic_tree=tree        # The topic tree to guide data diversity.
)

# Step 7: Save the generated dataset to a JSONL file.
dataset.save("data/output_01.jsonl")
```

---

## Telemetry: `pluto/posthog/`

This tool includes anonymous telemetry to help improve the software. The telemetry code is located in [`pluto/posthog/events.py`](pluto/posthog/events.py:1):

- **Purpose:**Tracks when data or topic tree creation jobs are started and ended, without collecting any dataset content.
- **How It Works:**

  - Uses the [PostHog](https://posthog.com/) analytics platform.
  - The function [`capture_event`](pluto/posthog/events.py:8) sends anonymized event data unless the environment variable `ANONYMIZED_TELEMETRY` is set to `"False"`.
  - The PostHog client is initialized with a project key and host.

**Example:**

```python
from posthog import Posthog
import os

posthog = Posthog('your_project_key', host='https://eu.posthog.com')

def capture_event(event_name, event_properties):
    if not os.environ.get('ANONYMIZED_TELEMETRY') == "False":
        posthog.capture("user_id", event_name, event_properties)
```

- To disable telemetry, set:
  ```bash
  export ANONYMIZED_TELEMETRY=False
  ```
- If set to `True`, anonymous usage data about your data and topic tree generation jobs will be sent to the analytics service. No dataset content is collected—only event metadata is transmitted.

---

## Data Conversion

To convert your `.jsonl` dataset to `.csv` for easier inspection or use with other tools:

```python
import json
import csv

with open('data/output_01.jsonl', 'r') as json_file:
    json_data = json_file.readlines()

system_content, user_content, assistant_content = [], [], []

for line in json_data:
    data = json.loads(line)
    for message in data['messages']:
        if message['role'] == 'system':
            system_content.append(message['content'])
        elif message['role'] == 'user':
            user_content.append(message['content'])
        elif message['role'] == 'assistant':
            assistant_content.append(message['content'])

with open('csv_data/output_01.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['System Content', 'User Content', 'Assistant Content'])
    for sys, usr, ast in zip(system_content, user_content, assistant_content):
        writer.writerow([sys, usr, ast])
```

---

## Contributing

Contributions are welcome! Open issues or submit pull requests on [GitHub](https://github.com/chaithanyasai18/LLMs-finetuning).

## License

Licensed under the [Apache License 2.0](LICENSE).

---

## Acknowledgements

This project incorporates ideas and builds upon foundational work from [https://github.com/redotvideo/pluto](https://github.com/redotvideo/pluto) and [Haven](https://haven.run/), a platform for fine-tuning LLMs. Our implementation extends these concepts to address the need for accessible, diverse synthetic datasets for LLM finetuning.
