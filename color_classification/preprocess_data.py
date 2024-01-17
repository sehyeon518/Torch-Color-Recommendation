import json
import numpy as np

five_colors_palette = r'/home/sehyeon/Documents/Favorfit-Color-Recommendation/color_classification/jsonl/five_colors_palette.jsonl'
list_of_colors = r'/home/sehyeon/Documents/Favorfit-Color-Recommendation/color_classification/jsonl/list_of_colors.jsonl'
json_file_path = r'/home/sehyeon/Documents/Favorfit-Color-Recommendation/color_classification/jsonl/quantized_product_colors_4to1.jsonl'


with open(list_of_colors, 'r') as list_of_colors:
    list_of_colors_data = [json.loads(line) for line in list_of_colors]


def find_closest_color(color, color_number = False):
    min_distance = float('inf')
    closest_color = color
    closest_number = 0
    for index, i in enumerate(list_of_colors_data):
        distance = np.linalg.norm(np.array(i['color_rgb']) - np.array(color))
        if distance < min_distance:
            min_distance = distance
            closest_color = i['color_rgb']
            closest_number = index

    if color_number:
        return closest_number
    
    return closest_color


def find_closest_colors(color_list, color_number = False):
    ret_list = []
    for color in color_list:
        closest_color = find_closest_color(color, color_number)
        ret_list.append(closest_color)

    return ret_list


count = 0

with open(five_colors_palette, 'r') as five_colors_palette:
    for product_line in five_colors_palette:

        product_color = json.loads(product_line)    # one line
        obj_colors = product_color['palette'][:4]     # [[r, g, b], [r, g, b], [r, g, b], [r, g, b]]
        bgnd_color = product_color['palette'][-1]  # [r, g, b]
        
        data = {}

        data['obj_colors'] = find_closest_colors(obj_colors)
        data['bg_color'] = find_closest_color(bgnd_color, True)
        
        with open(json_file_path, 'a') as jsonl_file:
            json.dump(data, jsonl_file)
            jsonl_file.write('\n')

        if count % 1000 == 0:
            print("count:", count)
        count += 1

print(count)
