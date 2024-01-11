import numpy as np
from colorspace_converter import rgb_to_hsv

def calculate_statistics(channel):
    mean_value = np.mean(channel)
    stddev_value = np.std(channel)
    median_value = np.median(channel)
    max_value = np.max(channel)
    min_value = np.min(channel)
    maxmin_diff = np.max(channel) - np.min(channel)
    return [round(mean_value, 3), round(stddev_value, 3), round(median_value, 3), round(max_value, 3), round(min_value, 3), round(maxmin_diff, 3)]


def hue_probability(color, palette, threshold=20):
    probabilities = [np.abs(color - pal_color[0]) / 360.0 for pal_color in palette]

    return np.round(probabilities, 3).tolist()


def hue_adjacent_probability(hue_b, hue_c, threshold=30):
    hue_difference = np.abs(hue_b - hue_c)

    hue_difference = min(hue_difference, 360 - hue_difference) / 360.0

    # adjacent_probability = 1.0 if hue_difference <= threshold else 0.0

    return round(hue_difference, 3)


def calculate_hsv_statistics(values, log=False):
    values = np.array(values)

    if log:
        # Avoid divide by zero and negative values
        non_zero_positive_values = values[values > 0]

        if len(non_zero_positive_values) == 0:
            # If there are no positive values, log transformation is not applicable
            return [-1, -1, -1, -1]

        log_values = np.log(non_zero_positive_values)
    else:
        log_values = values

    mean_value = np.nan_to_num(np.mean(log_values), nan=-1)
    stddev_value = np.nan_to_num(np.std(log_values), nan=-1)
    max_value = np.nan_to_num(np.max(log_values), nan=-1)
    min_value = np.nan_to_num(np.min(log_values), nan=-1)

    return [round(mean_value, 3), round(stddev_value, 3), round(max_value, 3), round(min_value, 3)]



def calculate_hue_entropy(hsv_palette):
    hue_values = hsv_palette[:, 0]

    histogram, bin_edges = np.histogram(hue_values, bins=360, range=(0, 360), density=True)

    non_zero_histogram = histogram[histogram != 0]

    entropy = -np.sum(non_zero_histogram * np.log2(non_zero_histogram))

    return entropy.tolist()

def calculate_CH(L1, C1, H1, L2, C2, H2):
    # Calculate HC
    delta_C = np.sqrt((np.square(H1 - H2) + np.square(C1 - C2 / 1.46)) / 2)
    HC = 0.04 + 0.53 * np.tanh(0.8 - 0.045 * delta_C)

    # Calculate HL
    Lsum = L1 + L2
    HLsum = 0.28 + 0.54 * np.tanh(-3.88 + 0.029 * Lsum)
    delta_L = np.abs(L1 - L2)
    Hdelta_L = 0.14 + 0.15 * np.tanh(-2 + 0.2 * delta_L)
    HL = HLsum + Hdelta_L

    # Calculate HH
    EC = 0.5 + 0.5 * np.tanh(-2 + 0.5 * C1)
    HS = -0.08 - 0.14 * np.sin(np.radians(H1) + np.radians(50)) - 0.07 * np.sin(np.radians(2 * H1) + np.radians(90))
    HSY1 = EC * (HS + L1)
    HS = -0.08 - 0.14 * np.sin(np.radians(H2) + np.radians(50)) - 0.07 * np.sin(np.radians(2 * H2) + np.radians(90))
    HSY2 = EC * (HS + L2)
    HH = HSY1 + HSY2

    # Calculate CH
    CH = HC + HL + HH

    return round(CH, 3)


def calculate_gradient(hue_values):
    hue_gradient = np.diff(hue_values)

    return np.round(hue_gradient, 3)