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


def rgb_to_lab(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0

    # Convert to XYZ
    r = 100.0 * ((r + 0.055) / 1.055) ** 2.4 if (r > 0.04045) else (r / 12.92)
    g = 100.0 * ((g + 0.055) / 1.055) ** 2.4 if (g > 0.04045) else (g / 12.92)
    b = 100.0 * ((b + 0.055) / 1.055) ** 2.4 if (b > 0.04045) else (b / 12.92)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Convert to LAB
    x = x / 95.047
    y = y / 100.000
    z = z / 108.883

    x = x ** (1 / 3) if (x > 0.008856) else ((x * 903.3 + 16) / 116)
    y = y ** (1 / 3) if (y > 0.008856) else ((y * 903.3 + 16) / 116)
    z = z ** (1 / 3) if (z > 0.008856) else ((z * 903.3 + 16) / 116)

    L = max(0, 116 * y - 16)
    a = (x - y) * 500
    b = (y - z) * 200

    return round(L, 3), round(a, 3), round(b, 3)


if __name__ == "__main__":
    rgb_values = (0, 0, 255)
    hsv_values = rgb_to_hsv(*rgb_values)
    print("RGB:", rgb_values)
    print("HSV:", hsv_values)
