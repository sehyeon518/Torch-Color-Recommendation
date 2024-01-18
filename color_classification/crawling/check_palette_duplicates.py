# adobe에서 추출한 색상 palette의 중복 검사

import json

jsonl_file_path = "/home/sehyeon/Documents/Favorfit-Color-Recommendation/color_classification/jsonl/five_colors_palette.jsonl"

unique_palette_list = []

duplicate_palette_list = []

with open(jsonl_file_path, 'r') as jsonl_file:
    for line in jsonl_file:
        data = json.loads(line)
        palette = data.get("palette")

        if palette not in unique_palette_list:
            unique_palette_list.append(palette)
        else:
            duplicate_palette_list.append(data)

print(len(unique_palette_list))
print(len(duplicate_palette_list))

with open(jsonl_file_path, 'w') as jsonl_file:
    for palette in unique_palette_list:
        data = {"palette": palette}
        json_line = json.dumps(data)
        jsonl_file.write(json_line + '\n')
