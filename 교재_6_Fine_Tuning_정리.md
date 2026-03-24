# 📘 교재 6: LLM Fine Tuning

> **Samsung SDS — 한번에 끝내는 AI 개발**
> 총 161페이지 | 9개 챕터 구성

---

## 📑 목차

1. [LLM Fine Tuning의 이해](#1-llm-fine-tuning의-이해)
2. [강화 학습과 SFT](#2-강화-학습과-sft)
3. [Transformer와 LLM의 탄생](#3-transformer와-llm의-탄생)
4. [파인 튜닝 데이터 구축하기 (1)](#4-파인-튜닝-데이터-구축하기-1)
5. [파인 튜닝 데이터 구축하기 (2)](#5-파인-튜닝-데이터-구축하기-2)
6. [Instruction Tuning](#6-instruction-tuning)
7. [Prompt Efficient Fine Tuning의 이해](#7-prompt-efficient-fine-tuning의-이해)
8. [PEFT로 효율적 파인 튜닝하기](#8-peft로-효율적-파인-튜닝하기)
9. [Reasoning LLM과 파인 튜닝](#9-reasoning-llm과-파인-튜닝)

---

## 1. LLM Fine Tuning의 이해

*(pp.1~21)*

### 1.1 LLM Fine Tuning: Overview (pp.2~3)

- **정의**: 새로운 데이터를 추가 학습하여 LLM의 성능을 향상시키는 방법
- **기존 접근 vs Fine Tuning**

| 기존 접근 | Fine Tuning |
|---------|-------------|
| Prompt 안에서 Context 추가 | Prompt 바깥에서 모델 자체 조정 |
| Tool 연결 등으로 작업 수행 | LLM의 사고 회로 구조 변경 |

- Fine Tuning은 **학습 메커니즘을 활용**
- 주로 다음 토큰을 예측하는 **SFT(Supervised Fine Tuning)** 방식
- **강화 학습(RL, Reinforcement Learning)** 등을 이용한 Reward 기반 학습도 포함
- LLM을 이용한 구조화된 출력 생성 → **기존 모델 기반으로 추가 학습**

### 1.2 LLM Training VS Fine Tuning (pp.4~5)

**LLM Training**
- 다양한 데이터에 대해서 모델을 처음부터 학습시키는 과정
- 초대량의 데이터 + 느린 수렴 → 비용이 매우 큼 ($$$)
- 전반적 언어 패턴 이해
- **베이스(Base)** 또는 **프리트레인(Pretrain)** 모델이라고 부름

**Fine Tuning**
- 이미 만들어진 모델을 추가로 학습시키는 과정
- 기존 Training 대비 **적은 데이터와 빠른 수렴**
- 일반성보다는 **목적을 갖춘 추가 학습**

### 1.3 Fine Tuning 모델 찾기: Qwen2.5 7B (pp.6~7)

- HuggingFace에서 Qwen2.5-7B 모델 검색 시 **6개 이상의 변형 모델** 존재

| 모델 유형 | 설명 | 예시 |
|--------|-----|-----|
| **Base Model** | 채팅 기능 미포함 | Qwen2.5-7B |
| **Instruct Model** | 채팅 기능이 탑재된 Fine Tuning 모델 | Qwen2.5-7B-Instruct |
| **Quantized Model** | 양자화로 크기 축소 | Qwen2.5-7B-Instruct-GPTQ-Int4 |

- Qwen3 기준: Base Model, Instruct Model, Quantized Model, **Thinking Model** 4가지
- 기본 Model = Base Model + Instruct Model

### 1.4 Base Model: Pretraining만 수행한 모델 (pp.8~9)

**Pretraining(사전 학습)**
- 주어진 문장의 **다음 단어 예측하기** (단순 Completion 학습)
  - 예: "LLM은 Large Language Model의 _____" ← 약자입니다
- 위키피디아, 스택오버플로우, 뉴스 기사 등을 수집하여 학습
  - C4(Colossal Clean Crawled Corpus) 등 활용
- 저작권을 위반한 데이터가 학습되는 경우 존재 (적발하기 어려운 특성)
- 특수 목적 학습을 위해 **합성 데이터 생성 및 활용**이 중요
  - Reasoning 기반의 데이터 활용

### 1.5 Base Model: Pretrained Model (p.10)

- 다음 단어 예측만을 수행하므로, **질의응답/지시사항 능력 부족**
- 입력: "거대 언어 모델은" → 출력: 관련 없는 이어쓰기 (질의응답 불가)
- 입력: "거대 언어 모델이 뭐야?" → 출력: 또 다른 질문을 이어서 생성

### 1.6 Instruct Model: 질의응답/지시사항 형식의 템플릿 이해 (p.11)

**단순 Completion → Instruct 모델**
- 질의응답과 지시사항의 데이터로 Fine Tuning한 모델
- **Llama 3 Series Template** 예시:
  ```
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  "당신은 도움을 주는 비서입니다."
  <|eot_id|><|start_header_id|>user<|end_header_id|>
  What can you help me with?
  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
  ```
- `<|start_header_id|>assistant<|end_header_id|>` 이후부터 생성

### 1.7 언제 Fine Tuning을 수행하는가? (pp.12~15)

**Fine Tuning을 하는 4가지 목적**

| # | 목적 | 설명 |
|---|------|------|
| 1 | **지식** | 최신/도메인 데이터에 대한 근본적 지식 배우기 |
| 2 | **능력** | 문제 풀이 향상, 요약/번역/수학 풀이 등 |
| 3 | **형식** | 업무용 형식/길이/말투 등 출력 스타일 변화 |
| 4 | **안전** | 유해한 요청 거부하기, 유익한 응답하기 |

- 각 모델별 지식 한계:
  - GPT: 시스템 프롬프트의 내용일 뿐임
  - Llama 3: 2023년 12월까지 학습했다고 명시
  - Claude, Gemini: 2025년 1월, 4월까지 학습한 버전 존재

**Fine Tuning 학습 유형**
- **새로운 Corpus 학습**: 기존 언어 패턴 변화, '언어의 토양' 바꾸기
- **입력-출력 형식의 데이터 학습**: 특정 작업으로 모델 특화, 목적에 맞는 출력 방식
- **RL(강화 학습)**: 강화 학습으로 모델 보상 주기

### 1.8 LLM Fine Tuning의 영향: 주의할 점 (pp.16~17)

- 다양한 데이터로 학습된 훌륭한 기존 모델
- 새로운 데이터를 학습시키면 **파라미터가 바뀌게 됨**
- **Unlearning is hard** — 한번 학습하면 되돌리기 어려움

**Catastrophic Forgetting (재앙적 망각)**
- 추가 학습을 진행하자, 모델의 **기존 능력이 감소**하는 현상
  - 예: 코딩 학습 → 언어 능력 감소 / 한국어 학습 → 영어 능력 감소
- Learning-Forgetting Tradeoff 그래프: Full Finetuning vs LoRA 비교
  - **LoRA**가 source domain 능력을 더 잘 보존
- 출처: https://arxiv.org/abs/2405.09673

### 1.9 LLM Fine Tuning의 부작용 (pp.18~19)

**과적합(Overfitting)**
- LLM에서도 ML의 Train/Test, Generalization 관련 내용은 그대로 성립
- LLM의 과적합은 **측정이 어려움**
- 예: 의료 상담 데이터로 과적합된 모델
  - "정사각형의 넓이가 6일 때, 한 변의 길이는?" → 의료 용어로 답변
- 해결: **풍부한 데이터 준비** 또는 **기존 지식의 복습 데이터 포함**

### 1.10 Fine Tuning 파이프라인 (pp.20~21)

**Fine Tuning 파이프라인의 5가지 Step**

준비물: Base Model(또는 Instruct Model), Fine Tuning 데이터

| Step | 내용 |
|------|------|
| 1 | Continuous Pretraining으로 도메인 지식 주입하기 |
| 2 | Instruction Tuning으로 지시사항/질의응답 학습하기 |
| 3 | Rejection Sampling + SFT으로 출력 성능 개선하기 |
| 4 | DPO 등(PPO, GRPO, ...)으로 선호 패턴 학습하기 |
| 5 | Alignment 데이터로 모델 안전성 높이기 |

- 성능 및 효과에 대한 이해 → **유연한 파이프라인 모델 적용**

---

## 2. 강화 학습과 SFT

*(pp.22~70)*

### 2.1 Continuous Pretraining (pp.23~25)

**파인튜닝(Fine-Tuning)의 주요 방식 정리**

| 방식 | 설명 |
|------|------|
| **SFT** (Supervised Fine-Tuning) | 정답 데이터를 다음 토큰 예측 문제로 학습하는 방식 |
| **CPT** (Continuous Pre-Training) | Pretrained Model을 새로운 Corpus로 추가 학습시키는 방식 |

- CPT는 Fine-Tuning으로 보지 않는 시각 존재 (Pretraining과 똑같은 학습 방식)
- CPT는 주로 **Domain Knowledge, Multilingual Corpus 학습**에 활용

**CPT 예시: SK Telecom → A.X 4.0**
- Qwen 2.5 모델에 한국어 데이터를 Pretrained 적용한 모델
- 모델의 변화를 **가장 크게 만드는 방식**

**CPT 특징**
- 특정 언어 능력 강화, 최신 정보 업데이트, 도메인 특화 가능
- **데이터가 굉장히 많이 필요함** → **망각 가능성 높음**
- Instruct Model에는 수행하지 않음 (모델의 Instruction 능력이 손상된다는 연구 결과)

### 2.2 Instruction Tuning: 질의 응답 데이터로 학습하기 (pp.26~29)

**Instruction Tuning의 개념**
- 질의응답 데이터를 하나의 템플릿으로 구성해 학습시키는 방식
- 시스템 데이터 + Prompt + User Prompt + 답변 Prompt → 하나의 세트로 묶어 다음 토큰을 예측하는 방식으로 튜닝

**Instruction Tuning의 목적**
- Instruction 패턴 학습
- 지시사항 이행 능력을 강화

**대표적 Instruction Tuning 데이터: Alpaca 포맷**
- `### Instruction` (지시사항) + `### Question` (질문) + `### Answer` (답변)
- 하나의 세트를 구성해 답변을 가르침

**Reasoning Data: 추론 과정이 포함된 데이터**
- 질문, 추론 과정(Thinking), 답변을 학습
- 예: Open Thoughts, LIMO Korean class

### 2.3 Instruction Tuning의 효과 (p.30)

- Instruction Tuning 연구 논문('21.09)
- Instruction 템플릿을 잘 이해하면, **학습하지 않은 질문에도 답변 가능**
- 논문의 주장: Instruction Tuning은 작은 모델에는 효과 없음
- **현재**: Instruction Tuning은 **모델의 크기와 관련 없음**

### 2.4 Instruct Model + SFT (pp.31~32)

**Instruct 모델에서 추가로 SFT하는 목적**

| 구분 | 설명 |
|------|------|
| 기존 목적 | 질의응답 자체 학습 |
| 추가 목적 | 이미 질의응답이 가능한 모델에 특정 데이터를 반영하여 원하는 답변을 하도록 조정 |

**활용 예시**
- 오류 탐지 강화 (Review Data 스팸 여부 판별 등)
- 특정 Task 성능 강화
- Domain 특화 작업 성능 향상
- 출력 형식 제어
- 행동 스타일 조정
- 복잡한 프롬프트 엔지니어링 필요성 감소 (긴 프롬프트 축소)
  - 예: RAFT(Retrieval-Augmented Fine-Tuning)

### 2.5 RAFT: 오픈 북 시험 공부 (pp.33~38)

**RAFT의 아이디어: RAG의 포맷 데이터로 모델 파인 튜닝**
- LLM이 RAG를 잘 수행하도록 학습시키는 파인튜닝 방식
- **Negative 문서 3개 + 진짜 문서 1개** + Query → 올바른 답변 훈련

**RAFT의 핵심**
- 관련 문서(Positive)와 무관한 문서(Negative)가 섞여 있을 때도 올바른 답변을 하도록 훈련
- Positive 문서만 포함한 경우의 질의응답 데이터를 학습
- 일반 RAG의 문제점: 문맥에 다른 정보가 섞여 있으면 모델이 혼동
- RAFT의 목표: RAG의 답변 형식 형식화

**RAFT: CoT-RAG 데이터 학습시키기**

| RAG | RAFT |
|-----|------|
| 형식을 정해준 적이 없음 | 형식을 고정함 |

- CoT Answer 패턴: `##근거` → `##인용시작##...##인용종료##` → `##정답`
- 질문 → 판단 → 근거 제시 → 문서 인용 → 해석 병행
- 데이터셋을 생성하여 Fine-Tuning 진행

**RAFT 학습 방식의 효과**
- 모델이 문서를 기반으로 문제를 풀게 됨
- 단순 RAG보다 더 똑똑한 답변 가능
- **환각(Hallucination) 감소**
- RAFT는 대표적인 SFT 방식 중 하나

### 2.6 Instruction Tuning의 역할 (~2024) (pp.39~43)

**비교 대상**
- CPT(Continuous Pre-Training): 계속적인 Pre-Training 방식
- IT(Instruction Tuning): Instruction을 가르치는 방식

**Meta의 LIMA(Less is More for Alignment)(2023)**
- IT는 지식 자체를 학습하는 것이 아니라 **질의응답 패턴을 학습하는 것에 불과**
- 대부분의 지식은 Pretrain 과정에서 학습됨
- Instruction Data는 지식을 정리하는 것에 불과함
- **소수(약 1,000개)의 데이터 튜닝을 통해 Instruction 능력 발현**

- 지식을 넣으려면 → CPT
- 질의응답을 가르치려면 → 2 Step 필요 (CPT + IT)

**Instruction Tuning으로도 지식 향상 가능(2024)**
- 해결하고자 하는 문제의 범위가 좁다면 가능할 수 있음
- 일반화하기 어려움 (한국어를 가르치거나 특정 분야의 지식 주입은 IT로는 힘듦)

### 2.7 Instruction Tuning의 역할 (2025) (pp.44~45)

**Reasoning Process 시도(2025)**
- IT 안에 Reasoning(추론 과정)을 포함시키면 지식 주입 가능
- 생각에는 지식 인출, 지식의 활용, 추론 과정이 포함됨
- 질의응답 모델에 **SFT만으로도 성능 향상**이 가능한가?
- **CPT가 필요한가?** → Reasoning IT만으로 해결할 수 있음

### 2.8 정리: SFT 파인 튜닝 (pp.46~47)

**SFT 요약 정리**

| 구분 | 필요 데이터 | 비고 |
|------|---------|------|
| 일반화된 지식 | Corpus 필요 | 지식 주입은 Pretrained 모델에서 수행 가능 |
| 질의응답 | QA Data 필요 | 템플릿 필요, 지식 주입 능력은 제한적이나 시도해볼 수 있음 |

- 도메인 지식을 넣는 채팅 모델 생성: Base Model → Continuous Pre-Training → Instruction Tuning
- 가장 복잡하지만, 이론적으로 확실한 방법
- 최근 Reasoning 모델: Reasoning IT만으로 해결할 수 있음

### 2.9 Post Training Part 2: 강화 학습 (pp.48~51)

**학습 방식 비교**

| SFT | RL |
|-----|-----|
| 질문과 답변이 정해져 있음 | 지식 일반화를 위해 필요함 |
| 정답을 100% 암기하도록 학습 (레이블 기반) | 확률 분포 기반 예측에 보상 함수를 적용하여 학습 |
| **일반화가 어려움** | |

**강화 학습 예시: 카드 게임(24 만들기)**
- 카드 4장(예: 1, 2, 7, 8)으로 사칙연산해 24 만들기
- SFT 학습 시: 문제를 그대로 외워 잘 풀었음
- 규칙 변경 시: 성능이 급격히 하락 → **SFT는 일반화가 잘 안됨**

**논의**
- RL: 일반화를 위해서는 RL이 필요함
- SFT: 다양한 데이터셋을 만들어 SFT를 강화하면 어느 정도 일반화 가능
- SFT 한계: 모든 데이터를 "100% 정답"으로만 학습, 모델이 획일적인 답변만 출력
- RL 필요: Reward(보상) 기반의 학습

### 2.10 LLM 강화 학습의 목적: 과거와 현재 (pp.52~55)

**RL 강화 학습 방식**
- 대표 기법: **RLHF(Reinforcement Learning with Human Feedback, OpenAI 제안)**
  - 아첨 모델(Sycophantic Model) 탄생 문제
- 목표 1단계: LLM에게 질의응답(QA) 능력을 먼저 가르침
- 목표 2단계: 보상 함수(Reward Function)를 적용하여 학습

**보상 함수의 결과**
- 긍정적 효과: RLHF, ChatGPT 같은 친절한 모델 탄생
- 부정적 효과: 의도치 않게 아첨 모델로 변질되기도 함
- → **RHF는 보상 함수가 필요함**

### 2.11 보상 모델과 RLHF (pp.56~60)

**보상 모델**
- 구조: (프롬프트 + 응답) → **Score**
- Prompt와 응답을 받아 Score를 뽑음 (예: 문장 분류)

**RLHF 과정**
1. 인간이 답변에 레이블 부여
2. 보상 모델 학습 (우열 데이터 기반)
3. LLM이 보상 모델의 점수를 높이도록 강화 학습

### 2.12 DPO: 직접적 선호 최적화 → RL의 훌륭한 대체 (p.65~)

**DPO(Direct Preference Optimization)**
- 보상 모델을 별도로 만들지 않고, LLM에 질문-답변 세트를 직접 주어 **좋은 답변과 나쁜 답변을 구분하는 방식**으로 학습시킴
- preference data → maximum likelihood → final LM

### 2.13 Rejection Sampling + SFT (p.70)

**이후 학습 내용**
- 파인 튜닝에 필요한 데이터 수집 실습
- 데이터 정제 및 수정을 통해 LLM의 성능을 높이는 과정 실습
- CPT, IT 두 가지 방식에 대해 실습

---

## 3. Transformer와 LLM의 탄생

*(p.71)*

- Transformer 아키텍처 및 LLM 발전 역사 소개 (별도 챕터)

---

## 4. 파인 튜닝 데이터 구축하기 (1)

*(pp.72~84)*

### 4.1 파인 튜닝 데이터 만들기 (pp.72~75)

**데이터 구축 프로세스**: 수집 → 증강(가공)

**수집**
- 관련 도메인의 다양한 데이터 크롤링 및 수집
- 데이터 형식을 바꾸고 싶을 경우 작업 필요

**과거의 방식** (예: 번역 데이터)
- ENG ↔ FR 같은 비슷한 데이터 확보
- **LLM을 활용한 데이터 확장·증강 가능**

**어떤 데이터를 준비해야 하는가?**
- 풀고자 하는 문제의 분포를 충분히 표현할 만한 데이터
- 예: Instruction Tuning → 지식을 주입하기 어려움

**도메인 선정 → LLM 활용** (p.80)

### 4.2 IT를 통한 CPT 모델의 질의응답 학습 (p.82)

**CPT 과정과 데이터 활용**
- 모델의 다양한 능력을 기르고 망각을 방지하기 위해 **다양한 데이터를 섞어서 학습하는 것이 필요**

---

## 5. 파인 튜닝 데이터 구축하기 (2)

*(실습 내용 포함)*

- 실습: 데이터 수집, 정제, 가공 과정
- CPT 및 IT 두 가지 방식에 대한 실습

---

## 6. Instruction Tuning

*(실습 내용 포함)*

- Instruction Tuning 실습
- 템플릿 구성 및 학습 데이터 준비

---

## 7. Prompt Efficient Fine Tuning의 이해

*(p.85~)*

### 7.1 PEFT(Parameter Efficient Fine Tuning) (p.90~)

**PEFT의 정의**
- 기존 파라미터를 보존하며, **적은 양의 메모리를 활용하는 Fine Tuning**
- Full Fine Tuning의 문제: 메모리 부족, 비용 과다

### 7.2 LoRA 메커니즘 이해하기 (p.95)

**LoRA(Low-Rank Adaptation)**
- Pretrained Weights는 그대로 두고, **새로운 A, B 행렬을 적용하여 결과 변경**
- 구조: `h = W·x + B·A·x` (W는 고정, A와 B만 학습)
- Adapter 적용: 파라미터 개수 = **2 × r × d**
- B = 0으로 초기화, A = N(0, σ²)으로 초기화

### 7.3 파라미터를 매우 적게 학습시키는 LoRA (p.100)

**파라미터 수 비교: d × d vs. 2 × r × d**

| 구분 | 예시 (d=3000) |
|------|-------------|
| 기존 모델 파인튜닝 | d × d = 3000² = **9,000,000개** 파라미터 |
| LoRA (r=2) | 3000 × 2 × 2 = **12,000개** 파라미터 |

- **900만 → 1만 2천** (약 900분의 1 축소!)

### 7.4 LoRA: 무엇이 좋은가? (p.110)

- 원래 모델의 파라미터를 변화시키지 않음 → **탈착 가능**
- LoRA의 배포 가능 (예: Gemma + LoRA 모델)

---

## 8. PEFT로 효율적 파인 튜닝하기

*(p.120~)*

- PEFT 라이브러리를 이용한 실습
- LoRA 설정 및 학습 실습

---

## 9. Reasoning LLM과 파인 튜닝

*(p.122~161)*

### 9.1 RLVR (Reinforcement Learning with Verifiable Reward) (p.125)

**보상함수**
- 질문 → 좋은 답변, 나쁜 답변 구분 정보 제공
- 모델은 답을 보고 상황·의미 판단
- 복잡한 모델 필요

### 9.2 GRPO (Group Relative Policy Optimization) (p.130)

**상대 평가를 통한 선호 최적화**
- 점수가 낮아도 상대적으로 우수하면 학습 반영
- 그룹 내에서 높은 Reward를 받는 출력을 자주 생성하도록 학습
- 점수가 높아도 표준화로 조정 가능

### 9.3 DeepSeek-R1-Zero 결과 (p.140)

**해결 방법**
- 일반적인 Instruction 학습 방식에 GRPO를 결합한 파이프라인을 다시 학습

**4단계 학습 파이프라인**

```
Cold-Start SFT → GRPO → Rejection Sampling + SFT → All-Scenario RL
```

- R1-Zero를 이용해 답변 데이터 세트 생성

### 9.4 RL VS SFT: 추론 성능은 무엇이 좌우하는가? (p.150)

**RL의 효과**
- 불필요한 반복을 줄이고 적절한 길이로 간결하게 답변을 만듦
- 지식을 잘 꺼내서 정확하게 추론, 매끄럽게 정리해 문제 풀이

**SFT만 적용했을 때 한계**
- 지식을 배우기는 하지만, 매끄러운 추론 능력까지 확보하지는 못함

### 9.5 Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model? (p.155)

**'RL 무용론'의 대표 근거가 된 논문**

비판 관점:
- 예: 질문에 대해 200번 답변 생성 중 1번 정답 / 128번 답변 생성 중 1번 정답
- 반복 시도 끝에 우연히 맞춘 것일 수 있음
- 실력 향상으로 보기 어려움
- 다양한 탐색 방향이 줄어듦 → 정확도는 올라가지만 **창의성이 줄어듦**

---

## 📊 전체 요약: Fine Tuning 방법론 비교

| 방법 | 데이터 | 목적 | 특징 |
|------|------|------|------|
| **CPT** | 대규모 Corpus | 도메인 지식 주입 | 변화가 가장 큼, 망각 위험 |
| **SFT (Instruction Tuning)** | QA 데이터 | 질의응답 패턴 학습 | 템플릿 기반, 소량 데이터 가능 |
| **RAFT** | RAG 형식 데이터 | RAG 성능 강화 | CoT + 인용 패턴 학습 |
| **RLHF** | 인간 피드백 | 선호도 학습 | 보상 모델 필요 |
| **DPO** | 선호 쌍 데이터 | 선호도 학습 | 보상 모델 불필요 |
| **GRPO** | 그룹 비교 데이터 | 상대 평가 최적화 | DeepSeek-R1에서 활용 |
| **LoRA (PEFT)** | QA 데이터 | 효율적 파인 튜닝 | 파라미터 수 극소화 (~1/900) |

---

## 🔑 핵심 키워드 정리

| 키워드 | 설명 |
|--------|------|
| **SFT** | Supervised Fine-Tuning. 정답 데이터를 다음 토큰 예측으로 학습 |
| **CPT** | Continuous Pre-Training. 새로운 Corpus로 추가 사전학습 |
| **RLHF** | Reinforcement Learning with Human Feedback |
| **DPO** | Direct Preference Optimization. 보상 모델 없이 선호 학습 |
| **GRPO** | Group Relative Policy Optimization. 그룹 상대 평가 |
| **RLVR** | Reinforcement Learning with Verifiable Reward |
| **PEFT** | Parameter Efficient Fine Tuning. 적은 파라미터로 효율적 학습 |
| **LoRA** | Low-Rank Adaptation. A, B 저랭크 행렬로 학습량 극소화 |
| **RAFT** | Retrieval-Augmented Fine-Tuning. RAG 특화 파인 튜닝 |
| **Catastrophic Forgetting** | 재앙적 망각. 추가 학습 시 기존 능력 감소 |
| **LIMA** | Less is More for Alignment. 소량 IT로도 충분하다는 연구 |

---

> 📌 **Samsung SDS 한번에 끝내는 AI 개발 — 교재 6: LLM Fine Tuning 전체 정리 완료**
