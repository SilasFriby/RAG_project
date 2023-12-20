import json
import os

def jsonl_to_txt_files(jsonl_file_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            document_id = data['id']
            text = data['text']

            # Create a text file for each document
            with open(os.path.join(output_dir, f'{document_id}.txt'), 'w') as text_file:
                text_file.write(text)


if __name__ == '__main__':

    # Define paths
    jsonl_file_path = 'data/statements_id_text_sub.jsonl'  # Update with your file path
    output_dir = 'data/statements_txt_files'  # Update with your desired output directory

    # Convert JSONL to text files
    jsonl_to_txt_files(jsonl_file_path, output_dir)
