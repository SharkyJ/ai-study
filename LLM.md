# Transformer

- RNN: 문장이 길어지면 앞 내용 손실
- T: 각 토큰들을 병렬적으로 처리

## Positional Encoding
### Decoder 입/출력
- Decoder 입력: Encoder의 출력값 + 현재까지의 출력 문장
  - 처음에는 <BOS> 토큰만 입력됨
  - 번역을 시작하라는 의미
  - 자기 회귀적 (Auto Regressive)
 
병렬 실행이므로 각각의 순서를 알 수 없다.

위치 인코딩 벡터가 각 토큰에 입력됨

## Transformer Layer
### Self-Attention
- 다른 단어와의 관계로 자신 표현하기
- Multi-head Attention

### FFNN
- Encoder
- Feed-Forward Neural Network

### Nx
- N번을 돌며 수행한다

### Encoding
- Self-Attention + Encoder

## Decoder 구조
- Encoder가 준 정보를 바탕으로 Decoder가 순차적 출력한다.
- BOS --> EOS


## 후속 모델
### BERT
- Encoder 구조를 활용하여 문장 의미 파악

### GPT
- Decoder 구조만 활용
- Attention > FFNN
- 어떤 문장을 주고 이 문장의 다음 단어를 외우는 구조

--------------------------------------------------------------
# LLM의 발전과정
## GPT: Next Token Prediction
- 문맥을 고려해 주어진 문장의 다음 단어 예측하기

## 개/고양이 분류 문제
- 이미지 분류 라는 단일 Task만을 목적으로 학습


ㅇㅇㅇ 분류
1. 딥러닝 모델 초기화
2. 학습할 ㅇㅇㅇ 준비
3. 0 or 1 학습


## BERT/GPT

특정 Task가 아닌, 언어의 패턴 자체를 학습하고자 함
- 약간의 추가 학습만 거쳐도 단일 Task 모델처럼 뛰어난 성능을 보임

### Foundation Model
- Pretrain
- Fine-tuning











































 
