# adobe에서 추출한 다섯가지 색상으로 이루어진 팔레트 * 5 (복제)
# 데이터 편향에 따라 기존의 데이터를 복제함
# 복제 형태는 다섯가지 중 한 가지를 고르는 5 combination 1

import json
import numpy as np

list_of_colors = r'C:\Users\mlfav\lib\shlee\Favorfit-Color-Recommendation\color_classification\jsonl\list_of_colors.jsonl'
five_colors_palette = r'C:\Users\mlfav\lib\shlee\Favorfit-Color-Recommendation\color_classification\jsonl\five_colors_palette.jsonl'
duplicated_palettes = r'C:\Users\mlfav\lib\shlee\Favorfit-Color-Recommendation\color_classification\jsonl\72030_palettes.jsonl'

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
        if len(product_color['palette']) != 5:
            print("색상 개수 5개 아님:", count / 5)
            continue
        for i in range(5):
            bgnd_color = product_color['palette'][i]
            copied_color = product_color['palette'][:]
            copied_color.pop(i)
            obj_colors = copied_color

            data = {}

            data['obj_colors'] = find_closest_colors(obj_colors)
            data['bg_color'] = find_closest_color(bgnd_color, True)

            with open(duplicated_palettes, 'a') as jsonl_file:
                json.dump(data, jsonl_file)
                jsonl_file.write('\n')

            if count % 1000 == 0:
                print("count:", count)
            count += 1
