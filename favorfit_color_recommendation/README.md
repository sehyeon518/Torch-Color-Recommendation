# Favorfit Color Recommendation

상품과 상품의 마스크 이미지가 전달되면, 상품 이미지에서 네 가지 주요 색상을 추출합니다. 이후, 각 색상의 RGB, HSV, YCrCb 등의 색상 공간에서의 통계값을 계산하여 색상의 특징을 추출합니다. 그리고 이러한 특징을 기반으로 어울리는 색상 네 가지를 추천하는 모델을 포함하고 있습니다. 

## Features
색상 추천 모델을 이벤트 기반의 AWS Lambda 함수로 배포하기 위한 코드입니다.

1. **이미지 및 마스크 로드**: 이벤트에서 전달된 이미지 및 마스크를 로드합니다. 이미지는 base64 형식으로 전달되며, 마스크는 option 입니다.

2. **색상 추출 및 특징 추출**: 이미지에서 색상을 추출하고 각 색상에 대한 특징을 추출합니다. 특징은 RGB, HSV, YCrCb 값 및 각 color space에서의 통계량으로 구성됩니다.

3. **모델 실행 및 결과 추론**: 추출된 특징을 입력으로 모델을 실행하여 각 색상에 대한 추론을 수행합니다.

4. **배경 데이터 로드 및 유사도 계산**: 배경 데이터를 로드하고, 추론된 색상과의 유사도를 계산하여 가장 유사한 배경 색상을 찾습니다.

5. **색상 포맷 변환**: 추론된 색상을 RGB에서 HEX로 변환하여 응답합니다.

## Reference
- favorfit recommend templates

### Library
- **colorthief**: 이미지에서 주요 색상을 추출하기 위해 사용됩니다.

### Train
- Train with Pytorch - [train.ipynb](../color_classification/Classification/train.ipynb)