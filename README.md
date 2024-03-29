# Uniform-labeled-data

라벨 데이터 균등 분포를 통한 이미지 분류 정확도 개선에 관한 연구
-------------------------
A Study on the Improvement of Image Classification Accuracy through Uniform Distribution of Label Data
-------------------------
## Description

Uniform-labeled-data는 이미지 분류를 위한 학습을 수행할 때 학습에 수행되는 라벨 데이터의 분포를 균일하게 함으로서 이미지 분류 성능을 높이고자 한다.
기존에 사용되던 이미지 데이터를 랜덤으로 섞어 학습에 사용하는 방법과 달리 이 프로젝트에서는 배치 내의 라벨 데이터의 라벨 분포를 균일하게 하여 학습을 진행한다.

### 방법 설명

|방법|설명|
|------|---|
|기존|데이터 세트 내의 라벨 데이터를 랜덤하게 섞어 학습에 활용했을 때의 결과를 확인하기 위한 방법|
|단순 균등|데이터 세트 내의 라벨 별 이미지 데이터의 개수를 동일하게 하여 학습에 활용했을 때의 결과를 확인하기 위한 방법|
|에폭 균등|데이터 세트 내의 라벨 별 데이터의 개수를 동일하게 하고 학습 데이터를 매 Epoch 마다 섞었을 때의 결과를 확인하기 위한 방법|

### 실험 환경

> Python Version 3.9.7 (Window)
> VSCODE

### 파일

> 'main.py' 실행 파일
> 'pre_data.py' 데이터 처리
> 'models' 실험 모델
> 'args.py' 하이퍼 파라미터

### 사용법
> python main.py
