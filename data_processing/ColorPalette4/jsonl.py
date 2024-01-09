import os
import json


load_json_file = r'C:\Users\mlfav\lib\shlee\color_palette\train.jsonl'
json_file_path = r'C:\Users\mlfav\lib\shlee\color_model\4_to_1\train.jsonl'
count = 0

with open(load_json_file, 'r') as f:
    for line in f:
        line = json.loads(line)

        data = {}
        data['image'] = line['image']
        data['input_colors'] = line['input_colors']
        data['output_color'] = line['output_colors'][0]

        with open(json_file_path, 'a') as jsonl_file:
            json.dump(data, jsonl_file)
            jsonl_file.write('\n')

        if count % 100 == 0:
            print("count:", count)
        count += 1

print("done:", count)
