from colorthief import ColorThief
import colorsys


# find complementary colors
def find_complementary_color(rgb):
    return [1.0 - c for c in rgb]


# find similar colors
def find_similar_colors(rgb, tolerance):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h1, s1, v1 = (h + tolerance) % 1.0, s, v
    h2, s2, v2 = (h - tolerance) % 1.0, s, v
    similar_color1 = list(colorsys.hsv_to_rgb(h1, s1, v1))
    similar_color2 = list(colorsys.hsv_to_rgb(h2, s2, v2))
    similar_colors = [similar_color1]  # TODO 필요한 색상 개수에 따라 추후 변동
    return similar_colors


# find triadic colors
def find_triadic_colors(rgb, tolerance):
    r, g, b = rgb
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (0.5 - h) % 1.0
    h1, l1, s1 = (h + tolerance) % 1.0, l, s - 0.25
    h2, l2, s2 = (h - tolerance) % 1.0, l, s - 0.25
    similar_color1 = list(colorsys.hls_to_rgb(h1, l1, s1))
    similar_color2 = list(colorsys.hls_to_rgb(h2, l2, s2))
    similar_colors = [similar_color1]  # TODO 필요한 색상 개수에 따라 추후 변동
    return similar_colors


def main(image_path, color_type):
    """
    주어진 이미지에서 색상을 추출하고, 지정된 유형에 따라 변환된 색상을 반환한다.

    Parameters:
    - image_path (str): 분석할 이미지 파일의 경로.
    - color_type (str): 반환할 색상의 유형. "complementary", "similar", "triadic" 중 하나.

    Returns:
    - list: 변환된 색상을 담은 리스트. 각 색상은 [r, g, b] 형태의 정규화된 RGB 값으로, 0에서 1 사이의 값.

    Raises:
    - ValueError: color_type이 올바른 값("complementary", "similar", "triadic")이 아닌 경우.

    Example:
    >>> main("path/to/image.jpg", "complementary")
    [[0.5, 0.2, 0.7], [0.8, 0.1, 0.3], ...]
    """

    color_thief = ColorThief(image_path)

    input_palette = color_thief.get_palette(4, 1)
    for i, (r, g, b) in enumerate(input_palette):
        input_palette[i] = [r / 255.0, g / 255.0, b / 255.0]

    if color_type not in ["complementary", "similar", "triadic"]:
        raise ValueError(
            "color_type should be one of 'complementary', 'similar', 'triadic'."
        )
    if color_type == "complementary":
        complementary_colors = [
            find_complementary_color(color) for color in input_palette
        ]
        return complementary_colors
    elif color_type == "similar":
        similar_tolerance = 0.03  # TODO 허용 범위 설정
        similar_colors = sum(
            [find_similar_colors(color, similar_tolerance) for color in input_palette],
            [],
        )
        return similar_colors
    elif color_type == "triadic":
        triadic_tolerance = 0.10  # TODO 허용 범위 설정
        triadic_colors = sum(
            [find_triadic_colors(color, triadic_tolerance) for color in input_palette],
            [],
        )
        return triadic_colors


if __name__ == "__main__":
    image_path = r"C:\Users\mlfav\lib\shlee\color_harmony\dalba_removebg.png"

    recommend_palette = main(image_path, "complementary")

    import matplotlib.pyplot as plt

    plt.imshow([recommend_palette])
    plt.axis("off")
    plt.show()
