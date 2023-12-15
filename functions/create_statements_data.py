
import json

def convert_statements_from_json_to_jsonl(input_file, output_file, key_items):

    # Read input_file, then use `model_validate_json`
    with open(input_file, "r", encoding="utf-8") as input_file:
        statements_data = json.load(input_file)

    # Loop through the statements and create a jsonl line for each statement
    jsonl_lines = []
    for statement in statements_data["statements"]:

        # Create jsonl line for each key_item
        jsonl_line = {}
        for key_item in key_items:
            if key_item not in statement.keys():
                raise ValueError(f"Key item {key_item} is not in the data")
            else:
                # Join paragraphs with two newlines if key item is statement2paragraphs
                if key_item == "statement2paragraphs":
                    jsonl_line[key_item] = "\n\n".join(statement[key_item])
                else:
                    jsonl_line[key_item] = statement[key_item]
        
        # Add jsonl line to list of jsonl lines
        jsonl_lines.append(json.dumps(jsonl_line, ensure_ascii=False))

    # Join the JSONL lines
    jsonl_data = "\n".join(jsonl_lines)

    # Save JSONL data
    with open(output_file, "w", encoding="utf-8") as output_file:
        output_file.write(jsonl_data)

    print(f"JSONL data has been saved to {output_file}")


#### EXAMPLE

if __name__ == "__main__":

    key_items = ["id", "title"]
    input_file = 'data/statements.json'
    output_file = 'data/' + 'statements_' + "_".join(key_items) + '.jsonl'
    convert_statements_from_json_to_jsonl(input_file, output_file, key_items)



