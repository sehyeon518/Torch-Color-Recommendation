import json
import numpy as np
import itertools
from colorspace_converter import rgb_to_hsv, rgb_to_lab
from calculate_statistics import *

load_json_file = r'C:\Users\mlfav\lib\shlee\color_palette\train.jsonl'
json_file_path = r'C:\Users\mlfav\lib\shlee\color_model\ColorModel\train.jsonl'
count = 0

with open(load_json_file, 'r') as f:
    for line in f:
        line = json.loads(line)

        l = line['input_colors'] + line['output_colors']

        
        RGB_8 = np.empty((8, 3), dtype=float)
        HSV_8 = np.empty((8, 3), dtype=float)
        LAB_8 = np.empty((8, 3), dtype=float)
        for idx, color in enumerate(l):
            r, g, b = color
            RGB_8[idx] = np.array(color).reshape((1, 3))
            HSV_8[idx] = np.array([rgb_to_hsv(r, g, b)]).reshape((1, 3))
            LAB_8[idx] = np.array([rgb_to_lab(r, g, b)]).reshape((1, 3))

        RGB_8 = RGB_8[np.argsort(LAB_8[:, 0])]
        HSV_8 = HSV_8[np.argsort(LAB_8[:, 0])]
        LAB_8 = LAB_8[np.argsort(LAB_8[:, 0])]


        data = {}
        RGB = RGB_8[2:6]
        HSV = HSV_8[2:6]
        LAB = LAB_8[2:6]   

        RGB_R = RGB[:, 0]
        RGB_G = RGB[:, 1]
        RGB_B = RGB[:, 2]
        HSV_H = HSV[:, 0]
        HSV_S = HSV[:, 1]
        HSV_V = HSV[:, 2]
        LAB_L = LAB[:, 0]
        LAB_A = LAB[:, 1]
        LAB_B = LAB[:, 2]

        hue_probabilities = []
        for color in HSV_H:
            probabilities = hue_probability(color, HSV)
            hue_probabilities.append(probabilities)

        adjacent_probabilities = []
        for i in range(len(RGB)):
            for j in range(i + 1, len(RGB)):
                color_b = HSV_H[i]
                color_c = HSV_H[j]
                adjacent_probabilities.append(hue_adjacent_probability(color_b, color_c, threshold=30))

        all_CH_values = []
        LCH_array = np.column_stack([LAB_L, LAB_A, HSV_H])
        for color1, color2 in itertools.combinations(LCH_array, 2):

            L1, C1, H1 = color1
            L2, C2, H2 = color2
            CH_value = calculate_CH(L1, C1, H1, L2, C2, H2)
            all_CH_values.append(CH_value)

        data['RGB'] = RGB.tolist()
        data['HSV'] = HSV.tolist()
        data['CIELAB'] = LAB.tolist()
        data['RGB_feature'] = [calculate_statistics(RGB_R), calculate_statistics(RGB_G), calculate_statistics(RGB_B)]
        data['HSV_feature'] = [calculate_statistics(HSV_H), calculate_statistics(HSV_S), calculate_statistics(HSV_V)]
        data['LAB_feature'] = [calculate_statistics(LAB_L), calculate_statistics(LAB_A), calculate_statistics(LAB_B)]
        data['Hue_probability'] = [calculate_hsv_statistics(hue_probabilities)]
        data['Hue_Log_probability'] = [calculate_hsv_statistics(hue_probabilities, True)]
        data['Hue_adjacent_probability'] = [calculate_hsv_statistics(adjacent_probabilities)]
        data['Hue_Log_adjacent_probability'] = [calculate_hsv_statistics(adjacent_probabilities, True)]
        data['Hue_entropy'] = [[calculate_hue_entropy(HSV)]]
        data['CH'] = [all_CH_values]
        data['Light_gradient'] = [calculate_gradient(LAB_L).tolist()]
        data['Hue_gradient'] = [calculate_gradient(HSV_H).tolist()]
        data['input_data'] = sum((data['RGB'], data['HSV'], data['CIELAB'], data['RGB_feature'], data['HSV_feature'], data['LAB_feature'], data['Hue_probability'], data['Hue_Log_probability'], data['Hue_adjacent_probability'], data['Hue_Log_adjacent_probability'], data['Hue_entropy'], data['CH'], data['Light_gradient'], data['Hue_gradient']), [])
        data['input_data'] = [[element for sublist in data['input_data'] for element in sublist]]

        data['output_color'] = HSV_8[0:2].tolist() + HSV_8[6:8].tolist()

        with open(json_file_path, 'a') as jsonl_file:
            json.dump(data, jsonl_file)
            jsonl_file.write('\n')

        if count % 100 == 0:
            print("count:", count)
        count += 1
        
print("done:", count)
