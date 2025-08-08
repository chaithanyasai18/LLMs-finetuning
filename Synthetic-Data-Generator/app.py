from pluto import EngineArguments, DataEngine, Dataset, TopicTree, TopicTreeArguments

# Step 1: Set the system prompt for the AI model.
# This prompt guides the AI to act as a knowledgeable assistant in cybersecurity, cloud computing, and IT support.
system_prompt = (
    "You are a safe, helpful, knowledgeable AI assistant and customer support expert specializing in cyber security, cloud computing and IT Support domain. Your primary job is to deliver detailed responses to customer queries related to the field of cybersecurity, cloud computing and IT Support domain. Leveraging your deep knowledge and expertise, please generate unique questions and relevant answers to guarantee customer satisfaction without repeating same questions. If you don't know the answer to a question, please don't share false information."
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
