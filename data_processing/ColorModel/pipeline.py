import torch
import torch.nn as nn
from colorthief import ColorThief
import colorsys
import numpy as np
from calculate_statistics import *
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as image

PATH = r'C:\Users\mlfav\lib\shlee\color_model\ColorModel\Model\model.pt'

dropout_rate = 0.5
alpha = 0.001

# model
class LassoModel(nn.Module):
    def __init__(self, input_size):
        super(LassoModel, self).__init__()
        self.normalize1 = nn.LayerNorm((input_size,))
        self.hidden1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.normalize2 = nn.LayerNorm((64,))
        self.hidden2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(32, 12)

    def forward(self, x):
        x = self.normalize1(x)
        x = torch.nn.functional.relu(self.hidden1(x))
        x = self.dropout1(x)
        x = self.normalize2(x)
        x = torch.nn.functional.relu(self.hidden2(x))
        x = self.dropout2(x)
        x = self.linear(x)
        return x
    
# loss function
def lasso_loss(model, h, y):
    mse_loss = nn.functional.mse_loss(h, y)
    
    l1_regularization = alpha * torch.sum(torch.abs(model.linear.weight))
    
    total_loss = mse_loss + l1_regularization
    
    return total_loss

# color space converter
def rgb_to_lab(r, g, b):
    # Convert RGB to XYZ
    r = 100.0 * (r / 255.0)
    g = 100.0 * (g / 255.0)
    b = 100.0 * (b / 255.0)

    r = r / 100.0
    g = g / 100.0
    b = b / 100.0

    if r > 0.04045:
        r = ((r + 0.055) / 1.055) ** 2.4
    else:
        r = r / 12.92

    if g > 0.04045:
        g = ((g + 0.055) / 1.055) ** 2.4
    else:
        g = g / 12.92

    if b > 0.04045:
        b = ((b + 0.055) / 1.055) ** 2.4
    else:
        b = b / 12.92

    r = r * 100.0
    g = g * 100.0
    b = b * 100.0

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Convert XYZ to LAB
    x /= 95.047
    y /= 100.000
    z /= 108.883

    if x > 0.008856:
        x = x ** (1.0 / 3.0)
    else:
        x = (903.3 * x + 16.0) / 116.0

    if y > 0.008856:
        y = y ** (1.0 / 3.0)
    else:
        y = (903.3 * y + 16.0) / 116.0

    if z > 0.008856:
        z = z ** (1.0 / 3.0)
    else:
        z = (903.3 * z + 16.0) / 116.0

    l = max(0.0, min(100.0, (116.0 * y) - 16.0))
    a = max(-128.0, min(127.0, (x - y) * 500.0))
    b = max(-128.0, min(127.0, (y - z) * 200.0))

    return [l, a, b]

# color space converter - any shape
def hsv_to_rgb(hsv_array):
    rgb_array = np.zeros_like(hsv_array)
    for i in range(hsv_array.shape[0]):
        h, s, v = hsv_array[i]
        r, g, b = colorsys.hsv_to_rgb(h.item(), s.item(), v.item())
        rgb_array[i] = np.array([r, g, b])

    return rgb_array

# extract 119 features from 4 colors
def extract_features(palette):
    input_color = np.array(palette)

    RGB = np.empty((4,3), dtype=float)
    HSV = np.empty((4,3), dtype=float)
    LAB = np.empty((4,3), dtype=float)

    for idx, color in enumerate(input_color):
        r, g, b = color
        RGB[idx] = np.array([r, g, b]).reshape((1, 3))
        HSV[idx] = np.array(list(colorsys.rgb_to_hsv(r, g, b))).reshape((1, 3))
        LAB[idx] = np.array([rgb_to_lab(r, g, b)]).reshape((1, 3))

    sorting_indices = np.argsort(LAB[:, 0])

    RGB = RGB[sorting_indices]
    HSV = HSV[sorting_indices]
    LAB = LAB[sorting_indices] 

    RGB_R, RGB_G, RGB_B = RGB[:, 0], RGB[:, 1], RGB[:, 2]
    HSV_H, HSV_S, HSV_V = HSV[:, 0], HSV[:, 1], HSV[:, 2]
    LAB_L, LAB_A, LAB_B = LAB[:, 0], LAB[:, 1], LAB[:, 2]

    hue_probabilities = []
    for color in HSV_H:
        probabilities = hue_probability(color, HSV_H)
        hue_probabilities.append(probabilities)

    adjacent_probabilities = []
    for color in HSV_H:
        probabilities = hue_adjacent_probability(color, HSV_H)
        adjacent_probabilities.append(probabilities)

    all_CH_values = []
    LCH_array = np.column_stack([LAB_L, LAB_A, HSV_H])
    for color1, color2 in itertools.combinations(LCH_array, 2):

        L1, a1, b1 = color1
        L2, a2, b2 = color2
        result = calculate_CH(L1, np.sqrt(a1**2 + b1**2), np.degrees(np.arctan2(b1, a1)),
                        L2, np.sqrt(a2**2 + b2**2), np.degrees(np.arctan2(b2, a2)))

        all_CH_values.append(result)

    RGB_feature = [calculate_statistics(RGB_R), calculate_statistics(RGB_G), calculate_statistics(RGB_B)]
    HSV_feature = [calculate_statistics(HSV_H), calculate_statistics(HSV_S), calculate_statistics(HSV_V)]
    LAB_feature = [calculate_statistics(LAB_L), calculate_statistics(LAB_A), calculate_statistics(LAB_B)]
    Hue_probability = [calculate_hsv_statistics(hue_probabilities)]
    Hue_Log_probability = [calculate_hsv_statistics(hue_probabilities, True)]
    Hue_adjacent_probability = [calculate_hsv_statistics(adjacent_probabilities)]
    Hue_Log_adjacent_probability = [calculate_hsv_statistics(adjacent_probabilities, True)]
    Hue_entropy = [[calculate_hue_entropy(HSV)]]
    CH = [all_CH_values]
    Light_gradient = [calculate_gradient(LAB_L).tolist()]
    Hue_gradient = [calculate_gradient(HSV_H).tolist()]
    RGB = RGB.tolist()
    HSV = HSV.tolist()
    CIELAB = LAB.tolist()
    features = sum((RGB, HSV, CIELAB, RGB_feature, HSV_feature, LAB_feature, Hue_probability, Hue_Log_probability, Hue_adjacent_probability, Hue_Log_adjacent_probability, Hue_entropy, CH, Light_gradient, Hue_gradient), [])
    features = [[element for sublist in features for element in sublist]]
    features = torch.flatten(torch.tensor(features, dtype=torch.float32))

    return features


if __name__ == "__main__":
    load_model = torch.load(PATH)
    model = LassoModel(119)
    model.load_state_dict(load_model)

    model.eval()

    # image load
    image_path = r'C:\Users\mlfav\lib\shlee\color_harmony\perfume.jpg'

    # extract colors - clustering
    color_thief = ColorThief(image_path)

    palette = color_thief.get_palette(4, 5)
    for i, (r, g, b) in enumerate(palette):
        palette[i] = [r/255.0, g/255.0, b/255.0]

    input_color = np.array(palette)

    input_data = extract_features(input_color)

    with torch.no_grad():
        output = model(input_data)

        x = input_data[:12].numpy().reshape(4,3)
        
        img = image.imread(image_path)

        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.axis(False)
        
        plt.subplot(1, 3, 2)
        plt.imshow([x])
        plt.axis(False)
        
        plt.subplot(1, 3, 3)
        y = output.numpy().reshape((4,3))
        y = hsv_to_rgb(y).reshape((4,3))
        plt.imshow([y.reshape((4,3))])
        plt.axis(False)
        plt.show()