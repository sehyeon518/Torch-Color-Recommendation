import itertools
from colorthief import ColorThief
import matplotlib.image as img
import colorsys
import matplotlib.pyplot as plt
import numpy as np

image_path = r'C:\Users\mlfav\lib\shlee\color_harmony\perfume.jpg'
color_thief = ColorThief(image_path)

input_palette = color_thief.get_palette(4, 5)
for i, (r, g, b) in enumerate(input_palette):
    input_palette[i] = [r/255.0, g/255.0, b/255.0]

image = img.imread(image_path)


# find complementary colors
def find_complementary_color(rgb):
    return [1.0 - c for c in rgb]

complementary_colors = [find_complementary_color(color) for color in input_palette]


# find similar colors
def find_similar_colors(rgb, tolerance):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h1, s1, v1 = (h+tolerance)%1.0, max(0, min(1.0, s-tolerance)), v
    h2, s2, v2 = (h-tolerance)%1.0, max(0, min(1.0, s+tolerance)), v
    similar_colors = [list(colorsys.hsv_to_rgb(h1, s1, v1)), list(colorsys.hsv_to_rgb(h2, s2, v2))]
    return similar_colors

tolerance = 0.1  # 허용 범위
similar_colors = sum([find_similar_colors(color, tolerance) for color in input_palette], [])


# find triadic colors
def find_triadic_colors(rgb):
    r, g, b = rgb
    h, s, l = colorsys.rgb_to_hls(r, g, b)

    return [[(h + 0.33) % 1.0, l, s], [(h - 0.33) % 1.0, l, s]]

triadic_colors = sum([find_triadic_colors(color) for color in input_palette], [])


compatible_colors = complementary_colors + similar_colors + triadic_colors
combinations = list(itertools.combinations(compatible_colors, 4))


def rgb_to_lab(rgb):
    xyz = rgb_to_xyz(rgb)
    lab = xyz_to_lab(xyz)

    return lab

def rgb_to_xyz(rgb):
    matrix = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    # Convert sRGB to XYZ
    xyz = np.dot(np.power(rgb, 2.2), matrix.T)

    return xyz

def xyz_to_lab(xyz):
    ref_white = np.array([0.95047, 1.00000, 1.08883])

    xyz_normalized = xyz / ref_white

    xyz_normalized = np.power(xyz_normalized, 1/3)

    lab = np.zeros_like(xyz_normalized)
    mask = xyz_normalized > 0.008856
    lab[mask] = np.power(xyz_normalized[mask], 1/3)
    lab[~mask] = (xyz_normalized[~mask] * 903.3 + 16) / 116

    lab *= np.array([100, 255, 255])

    return lab


def calculate_HSY(hab):
    # EC
    EC = 0.5 + 0.5 * np.tanh(-2 + 0.5 * hab)
    # HS
    HS = -0.08 - 0.14 * np.sin(np.radians(hab + 50)) - 0.07 * np.sin(np.radians(2 * hab + 90))
    # HSY
    HSY = EC * (HS + np.e)
    return HSY

def calculate_CH(L1, Cab1, hab1, L2, Cab2, hab2):
    # ∆C
    delta_L = np.abs(L1 - L2)
    delta_Cab = np.sqrt((Cab1**2 + Cab2**2) / 1.46)
    delta_C = np.sqrt(delta_L**2 + delta_Cab**2)

    # HC
    HC = 0.04 + 0.53 * np.tanh(0.8 - 0.045 * delta_C)

    # HL_sum
    L_sum = L1 + L2
    HL_sum = 0.28 + 0.54 * np.tanh(-3.88 + 0.029 * L_sum)

    # H_∆L
    delta_L = np.abs(L1 - L2)
    H_delta_L = 0.14 + 0.15 * np.tanh(-2 + 0.2 * delta_L)

    # HH
    HSY1 = calculate_HSY(hab1)
    HSY2 = calculate_HSY(hab2)
    HH = HSY1 + HSY2

    # Calculate CH
    CH = HC + HL_sum + H_delta_L + HH

    return CH

def color_harmony(lab1, lab2):
    # Extract LAB values
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Calculate CH
    result = calculate_CH(L1, np.sqrt(a1**2 + b1**2), np.degrees(np.arctan2(b1, a1)),
                           L2, np.sqrt(a2**2 + b2**2), np.degrees(np.arctan2(b2, a2)))

    return result


CH_max = 100000000000
best_palette = input_palette
for combination in combinations:
    combination = list(combination)
    palette_RGB = input_palette + combination
    palette_LAB = np.array([rgb_to_lab(rgb) for rgb in palette_RGB])
    palette_LAB = palette_LAB[np.argsort(palette_LAB[:, 0])]

    CH_tmp = 0
    for i in range(len(palette_LAB)):
        for j in range(i + 1, len(palette_LAB)):
            ch_value = color_harmony(palette_LAB[i], palette_LAB[j])
            CH_tmp += ch_value
    
    # best palette update
    if CH_tmp < CH_max:
        CH_max = CH_tmp
        best_palette = palette_RGB

print(best_palette)
plt.imshow([best_palette])
plt.show()
