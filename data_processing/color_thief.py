from colorthief import ColorThief
import os
import json


PATH = r"C:\Users\mlfav\lib\shlee\Images\digital_arts"
# PATH = r'D:\FAVORFIT\digital_art\digital_art'

json_file_path = r'color_palette\train.jsonl'

count = 0

for file in os.listdir(PATH):
    data = {}
    image_path = os.path.join(PATH, file)

    color_thief = ColorThief(image_path)

    palette = color_thief.get_palette(color_count=9)
    print(palette)
    data["image"] = image_path
    data["input_colors"] = palette[:4]
    data["output_colors"] = palette[4:]

    with open(json_file_path, 'a') as jsonl_file:
        json.dump(data, jsonl_file)
        jsonl_file.write('\n')  # Add a newline character to separate each JSON object

    if count % 100 == 0:
        print("continue:", count)
    count += 1

print("done:", count)
