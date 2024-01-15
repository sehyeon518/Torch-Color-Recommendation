def rgb_to_hsv(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    M = max(r, g, b)
    m = min(r, g, b)
    
    V = M
    
    if M == 0:
        S = 0
    else:
        S = (M - m) / M
    
    if M == m:
        H = 0
    else:
        if M == r:
            H = (g - b) / (M - m)
        elif M == g:
            H = (b - r) / (M - m) + 2
        else:
            H = (r - g) / (M - m) + 4
        
        H *= 60
        if H < 0:
            H += 360
    
    return round(H, 3), round(S*100, 3), round(V*100, 3)


def rgb_to_lab(rgb):
    r, g, b = rgb

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


if __name__ == "__main__":
    rgb_values = (255, 0, 0)
    hsv_values = rgb_to_hsv(*rgb_values)
    lab_values = rgb_to_lab(rgb_values)
    print("RGB:", rgb_values)
    print("HSV:", hsv_values)
    print("LAB:", lab_values)
