import json
import csv

# Read JSONL file and parse it
with open('data/output_01.jsonl', 'r') as json_file:
    json_data = json_file.readlines()

# Initialize lists to store system, user, and assistant content
system_content = []
user_content = []
assistant_content = []

# Iterate through each line in the JSONL file
for line in json_data:
    data = json.loads(line)
    for message in data['messages']:
        if message['role'] == 'system':
            system_content.append(message['content'])
        elif message['role'] == 'user':
            user_content.append(message['content'])
        elif message['role'] == 'assistant':
            assistant_content.append(message['content'])

# Write data to CSV file
with open('csv_data/output_01.csv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['System Content', 'User Content', 'Assistant Content'])  # Write header
    # Write content
    for sys_content, usr_content, ast_content in zip(system_content, user_content, assistant_content):
        writer.writerow([sys_content, usr_content, ast_content])

print("CSV file generated successfully.")



