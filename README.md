# 색상 추천 모델 (Color Recommendation) - PyTorch
이 프로젝트는 PyTorch를 사용하여 개발된 색상 추천 모델입니다. 이 모델은 입력된 상품에 가장 적합한 색상을 추천합니다.

![image](https://github.com/sehyeon518/Favorfit-Color-Recommendation/assets/84698896/aea58f57-3435-40d3-b615-cc5f4296640e)


## 모델 아키텍처 (Model Architecture)
```plain
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                  [-1, 512]          61,440
         LayerNorm-2                  [-1, 512]           1,024
              ReLU-3                  [-1, 512]               0
            Linear-4                 [-1, 1024]         525,312
         LayerNorm-5                 [-1, 1024]           2,048
              ReLU-6                 [-1, 1024]               0
            Linear-7                  [-1, 540]         553,500
================================================================
Total params: 1,143,324
----------------------------------------------------------------
```

## Dataset
- **Adobe Color**: 이 데이터셋은 다섯 가지 색으로 구성된 palette의 RGB 색상 값을 포함합니다. 총 14,000개의 palette가 있습니다.
- **Coolors List of Colors** : 이 데이터셋은 연속된 RGB 값(255 * 255 * 255)을 540 가지로 분류된 색상 목록입니다.
- **Train Data**
    - Adobe Color palette를 순서를 다르게 하여 다섯 배로 복제합니다.
    - 각 팔레트의 각 색상을 Coolors List of Colors의 색상 목록에서 가장 가까운 색상으로 양자화합니다.
    - 각 팔레트의 네 가지 색상의 RGB(Red, Green, Blue), HSV(Hue, Saturation, Value), YCrCb(Y, Cr, Cb) 값과 이에 대한 통계량을 계산하여 입력 데이터로 사용합니다.
    - 각 팔레트의 다섯 번째 색상을 해당 팔레트의 클래스로 설정하여 분류 문제로 사용합니다.
    - 이러한 처리를 거친 데이터셋은 모델의 훈련에 사용되며, 입력 데이터로는 각 팔레트의 네 가지 색상과 그에 대한 통계량이 포함되어 있으며, 출력 데이터는 해당 팔레트의 다섯 번째 색상입니다.


## References
- Pytorch Learning Rate Scheduler - [참고링크](https://gaussian37.github.io/dl-pytorch-lr_scheduler/)
- N. Kita and K. Miyata. "Aesthetic Rating and Color Suggestion for Color Palettes." - [논문 링크](https://naokita.xyz/projects/ColorPalette/ColorPalette_pg2016.pdf)