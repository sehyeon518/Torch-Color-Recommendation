from colorthief import ColorThief
import colorsys
import numpy as np


# find complementary colors
def find_complementary_color(rgb):
    return [1.0 - c for c in rgb]


# find similar colors
def find_similar_colors(rgb, tolerance):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h1, s1, v1 = (h+tolerance)%1.0, s, v
    h2, s2, v2 = (h-tolerance)%1.0, s, v
    similar_colors = [list(colorsys.hsv_to_rgb(h1, s1, v1)), list(colorsys.hsv_to_rgb(h2, s2, v2))]
    return similar_colors


# find triadic colors
def find_triadic_colors(rgb, tolerance):
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (0.5 - h) % 1.0
    h1, l1, s1 = (h + tolerance)%1.0, l, s - 0.25
    h2, l2, s2 = (h - tolerance)%1.0, l, s - 0.25

    return [list(colorsys.hls_to_rgb(h1, l1, s1)), list(colorsys.hls_to_rgb(h2, l2, s2))]


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


def main(image_path, color_type):
    """
    주어진 이미지에서 색상을 추출하고, 지정된 유형에 따라 변환된 색상을 반환한다.

    Parameters:
    - image_path (str): 분석할 이미지 파일의 경로.
    - color_type (str): 반환할 색상의 유형. "complementary", "similar", "triadic" 중 하나.

    Returns:
    - list: 변환된 색상을 담은 리스트. 각 색상은 [r, g, b] 형태의 정규화된 RGB 값.

    Raises:
    - ValueError: color_type이 올바른 값("complementary", "similar", "triadic")이 아닌 경우.

    Example:
    >>> main("path/to/image.jpg", "complementary")
    [[0.5, 0.2, 0.7], [0.8, 0.1, 0.3], ...]
    """

    color_thief = ColorThief(image_path)

    input_palette = color_thief.get_palette(4, 1)
    for i, (r, g, b) in enumerate(input_palette):
        input_palette[i] = [r/255.0, g/255.0, b/255.0]

    if color_type not in ["complementary", "similar", "triadic"]:
        raise ValueError("color_type should be one of 'complementary', 'similar', 'triadic'.")
    if color_type == "complementary":
        complementary_colors = [find_complementary_color(color) for color in input_palette]
        return complementary_colors
    elif color_type == "similar":
        similar_tolerance = 0.03  # 허용 범위
        similar_colors = sum([find_similar_colors(color, similar_tolerance) for color in input_palette], [])
        return similar_colors
    elif color_type == "triadic":
        triadic_tolerance = 0.1
        triadic_colors = sum([find_triadic_colors(color, triadic_tolerance) for color in input_palette], [])
        return triadic_colors
    

if __name__ == "__main__":
    image_path = r'C:\Users\mlfav\lib\shlee\color_harmony\dalba_removebg.png'
    
    recommend_palette = main(image_path, "complementary")
    
    import matplotlib.pyplot as plt
    plt.imshow([recommend_palette])
    plt.show()
