# Samsung SDS 「한번에 끝내는 AI 개발」 — LLM 기본 교재 정리

> **과정명:** Large Language Models  
> **범위:** 5장 ~ 8장 (총 98페이지)

---

## 목차

1. [5장 — LLM 생태계와 주요 모델들 (2)](#5장--llm-생태계와-주요-모델들-2)
2. [6장 — 추론과 멀티모달 모델](#6장--추론과-멀티모달-모델)
3. [7장 — LLM API 소개](#7장--llm-api-소개)
4. [8장 — LangChain](#8장--langchain)

---

## 5장 — LLM 생태계와 주요 모델들 (2)

### 5-1. Alibaba Qwen Series

#### Qwen3 모델 크기와 구조 차이 (p.2)

- Qwen3는 **다양한 사이즈**와 **구조(Dense 모델 / MoE 모델)**를 보유
- 모델 크기 = **레이어 개수**와 밀접하게 연관됨
- Transformer 장점 = **스택을 쌓아 성능 향상** 가능

#### 모델 크기에 따른 레이어 변화 (p.3)

| Model | Layers | Heads (Q/KV) | Tie Embedding | Context Length |
|-------|--------|-------------|---------------|----------------|
| Qwen3-0.6B | 28 | 16/8 | Yes | 32K |
| Qwen3-1.7B | 28 | 16/8 | Yes | 32K |
| Qwen3-4B | 36 | 32/8 | Yes | 32K |
| Qwen3-8B | 36 | 32/8 | No | 128K |
| Qwen3-14B | 40 | 40/8 | No | 128K |
| Qwen3-32B | 64 | 65/8 | No | 128K |

- 더 큰 모델로 갈수록 레이어 수 증가: 28개 → 36개 → 40개 → 64개
- 작은 모델: 레이어 수 적음 / 큰 모델: 레이어 수 많음 → 성능 향상

#### Transformer 구조 특징 (p.4)

- Attention + MLP 구조 덕분에 **레이어 쌓기가 가능**
- 레이어 수만 늘어나는 게 아님 → **차원 수, Attention head 수도 같이 커짐**
- 예: 0.6B → 32B 모델 성장 시 약 **50배** 차이 발생
- 단순히 레이어만 50배 늘리는 방식이 아님 → **차원·헤드 등 종합적으로 확장**

#### 작은 모델의 일반적 특징 (p.5)

- Input/Output 임베딩 간소화
- Context 윈도우가 짧음

---

### 5-2. Mistral Series

#### Mistral 회사 및 모델 특징 (p.6)

- 역사적으로 의미 있는 모델 다수 제작
- **침인형 모델(Mistral):** 출시 당시 동일 사이즈 대비 성능 ↑, 2배 큰 모델인 라마2도 이김
- **스트라이딩 윈도우 어텐션:** 전체 문맥 X → 주변 문맥 집중 활용
- 다양한 연구 성과 축적

#### Mixtral 모델: Mixture of Expert(MoE) 방법 활용 (p.7)

- 2024년 초 공개
- 여러 모델을 묶어 만든 MoE 모델
- **8×7B, 8×22B** 모델 (2024년 4월까지 파운데이션 모델 1위)
- 국내에서는 인기 낮음 → **한국어 인식 부족**

#### Mistral Large Series (p.8)

- **Large** (2024년 2월 출시)
- **Large 2** (2024년 7월 출시)
  - 한국어 지원 시작
  - 사이즈: **123B** parameter
  - Llama 3.1(405B) 출시 다음날 공개 → 주목 받음

#### Mistral Small Series (p.9)

- Small 3.2 버전 (**24B** parameter)
- Small이지만 규모가 크고 **한국어 지원 성능 우수** (드문 케이스)
- **에이전트형 작업 지원 (Tool Calling 기능)**
- Microsoft Phi 4 모델과 유사한 수준에서 활용 가능

---

### 5-3. Gemma Series

#### Gemma Series 개요 (p.10–11)

- **구글이 개발한 모델**
- Gemma 2 출시 후 → Gemma 3 등장
- 현재는 **Qwen 또는 Gemma**가 대표적 선택지
- **한국어 성능도 우수**
- Gemma 2 + **SigLIP 인코더** 결합 → **PaliGemma** 공개
- 이후 완전 결합 → **멀티모달 LLM Gemma 3** 출시

**Gemma 3 라인업:**

| 항목 | 내용 |
|------|------|
| 공개 사이즈 | 1B, 4B, 12B, 27B |
| 멀티모달 지원 | 4B·12B·27B → 이미지 입력 가능 |
| 특징 | 범용성과 성능 우수 → 다양한 작업 활용 가능 |

---

### 5-4. 국내 LLM 모델

#### 국내 LLM 모델 현황 (p.12)

- 해외 모델 외에도 국내에서 **다양한 LLM이 출시**됨
- **한국어 특화 모델** 다수 등장
- 핵심: **한국어 데이터 비중 강화**

#### 카카오 Kanana (p.13)

| 항목 | 내용 |
|------|------|
| 버전 | 1.5 |
| 사이즈 | 2.1B, 8B |
| 공개 모델 | Base / Instruct |
| 라이선스 | Apache |
| 8B 모델 성능 | 한국어 처리 우수 → 실사용 가능성 높음 |

#### 네이버 HyperCLOVA Series (p.14)

| 항목 | 내용 |
|------|------|
| Seed 모델 | 0.5B, 1.5B, 3B 공개 |
| Think 모델 | 14B, Reasoning 특화, 한국어 리즈닝 지원 |
| 장점 | MAU 1천만 이상 고객 사용 가능 |
| 특징 | 멀티모달 지원 (3B 이상) |
| 활용 | 한국어 데이터 구축 및 연구에 도움이 큼 |

#### LG-AI EXAONE (p.15)

| 항목 | 내용 |
|------|------|
| 초기 공개 | 2.4B, 7.8B, 32B (엑사원 3.5 기반) |
| 최신 EXAONE 4.0 | 사이즈: 1.1B, 32B |
|  | 한국어 리즈닝 특화 |
|  | 국내 모델 중 **최상위권 벤치마크 성능** 기록 |

#### Upstage Solar Series (p.16–18)

- 2023년부터 다양한 모델 공개
- **Solar 10.7B** = 최초 모델
- **구성 방식: Depth Upscaling(DUS) 적용**
  - 모델 2개 준비 → 앞 모델은 후반부 제거, 뒤 모델은 앞부분 제거
  - 각 모델 3/4만 남겨서 이어 붙임 → **7B × 1.5 = 10.7B**
- **효과:** 추가 학습 시 성능 향상, 당시 UAE Falcon 등 대형 모델 능가 → 리더보드 1위 기록
- 이후 성능 입증 및 진화
  - 22B 모델 공개 (10.7B 후속)
  - 2025년 7월: **Solar Pro 2 31B** 공개
- 현재: API 공개 상태, 오픈소스 공개 가능성 있음

#### 국내 주요 통신사 LLM 모델 현황 (p.19)

- LG: EXAONE
- SKT, KT: A.X·믿음 등 모델 공개
- 국내 모델 강점: **한국어 데이터 다량 학습** → 한국어 성능 우수

#### 국내 LLM 활용 전략 (p.19)

- LLM 연구·개발 시 **파이프라인 구축 후** 모델 교체·테스트 방식으로 **성능 비교·검증** 권장

---

### 5-5. 오픈 LLM 모델 선택하기: 크기

#### 오픈모델 선택과 사이즈 문제 (p.20–21)

- 모델 사이즈 예시: 10.7B / 22B / 31B / 72B
- 단순히 골라 쓰는 문제가 아님 → **사이즈 자체가 중요한 변수**
- **사이즈 = 곧 GPU 크기와 직결**
- 일부는 CPU 실행 시도 중이나, 여전히 **GPU가 핵심 (Critical)**

---

### 5-6. 파라미터 수 – VRAM의 관계

#### 파라미터 수 개념 (p.22–26)

- 파라미터 수 = 보통 **B(Billion)** 단위 사용
  - 7B = 70억 개, 22B = 220억 개
- 초대형 X, 작은 사이즈 모델: 7B, 13B, 0.2B 등
- 사용 전 확인 필수: **GPU 사용량**, **인프라 환경**
- 가중치 = **16bit 또는 32bit**
  - GPT-2 등 구형 모델 → 32bit
  - 최근 모델 → 16bit

**비트 구조:**

| 형식 | 부호 | 지수 | 가수 | 특징 |
|------|------|------|------|------|
| 16bit | 1bit | 8bit | 7bit | 정밀도/해상도 ↓ |
| 32bit | 1bit | 8bit | 23bit | 정밀도/해상도 ↑ |

#### GPU 메모리 사용량 계산법 (p.27)

**7B 모델 (16bit 기준):**
- 파라미터 70억 × 16bit (= 2바이트) = 140억 바이트
- 140억 바이트 ÷ 10억 = **14GB**

**공식:**
| 형식 | 계산 |
|------|------|
| 16bit 모델 | 파라미터 수 × 2 → GB 변환 |
| 32bit 모델 | 파라미터 수 × 4 → GB 변환 |

→ **모델 로드 시 필요한 최소 GPU 메모리**

#### 실제 사용 시 GPU 소모 ↑ (p.28)

- 입력 컨텍스트 길이에 따라 GPU 사용량 더 증가
- Attention, MLP 연산 반복 + 과거 입력 유지 필요
- 과거 입력값 저장/재활용 = **KV 캐싱** → GPU 메모리 추가 소모

#### 안정적 사용 vs 학습 (p.29)

- 7B 모델 = 14GB 필요
- **안정적 사용을 위해 1.5배** 필요 → **21GB**
- **학습 시 = 최소 10배 이상** 필요 (학습 = GPU가 많을수록 좋음)

#### 최적화 및 대안 (p.30)

- GPU 소모를 줄이기 위한 기법:
  - **vLLM, Ollama** 등 서빙 최적화 프레임워크
  - **양자화(Quantization)** = 모델 크기 압축

---

### 5-7. Quantization(양자화) — 고전적 방식

#### 양자화 개념과 INT8 양자화 원리 (p.31–33)

- **Weight 값을 줄여 모델 크기를 작게** 만드는 과정
- 대표적 방식: **INT8 양자화**
- 가중치 분포 (예: 0~1 사이 값) → **0~255** 값으로 매핑
- 공식: `(값 - 최솟값) ÷ (최댓값 - 최솟값) × 255`
- 예: 1.0 → 255/255, 0.44 → 약 112 → **원래 값과 차이 발생**
- 양자화 시 **가중치가 미세하게 변형** → 성능 저하 원인, 답변 결과 달라질 수 있음
- 원리: 최솟값과 최댓값 기준으로 **255 단계 분할**
- 실제는 음수 범위 고려해 **÷ 2** 필요

| 항목 | 내용 |
|------|------|
| 장점 | 모델 크기 ↓, GPU 메모리 절감 |
| 단점 | 성능 저하, 특히 INT4에서 심각 |
| 특징 | 단순한 기계적 방법 (최대·최소 기준 단순 스케일링) |

---

### 5-8. Quantization(양자화) — 이후 발전

#### 기존 방식의 한계와 QLoRA 제안 개선 (p.34)

- 초기 양자화 = 최댓값·최솟값 기준 **등간격 분할 (등길이 분포)**
- 문제: 구간 고정 → **성능 손실 큼**
- **QLoRA 학습 방식 = 분포 기반 양자화** 적용
- 핵심: 등길이 대신 **분위수(Quantile) 기준 분할**

#### NF4(NormalFloat4) 양자화 방식 (p.35)

- 전체 값 범위 → **분위수 구간으로 나눔**
- 각 분위수 **대표 값만 저장** 후 활용

| 항목 | 내용 |
|------|------|
| 장점 | 원래 값 보존률 ↑ → 성능 저하 적음 |
|  | 아웃라이어가 있어도 강건함 |

#### VRAM 소모 비교 예시 (p.36)

| 로드 방식 | VRAM (MB) |
|---------|-----------|
| Original (bfloat16) | 15,637 |
| Load in 8bit | 8,927 |
| Load in 4bit | 6,095 |
| Load in 4bit + Double Quant | 5,749 |

- 16bit 모델(예: 15GB) → 8bit, 4bit 변환 시 용량 ↓
- **Double Quant** 기법 적용 → 중간값을 다시 양자화 → 모델 크기 더 크게 압축 가능

#### 기존 양자화 방식의 한계 (p.37)

- 모든 뉴런(파라미터)을 **동일한 기준으로 양자화**
- 뉴런마다 **중요도 다름** → 동일 처리 시 성능 저하 불가피

#### 개선된 양자화 기법: GPTQ, AWQ (p.38–39)

**GPTQ:**
- 파라미터 **중요도 기반 차등 양자화**
- 중요한 파라미터 → 적게 양자화
- 덜 중요한 파라미터 → 크게 양자화
- 앞단 양자화로 발생한 오류 → **뒷단 양자화에 반영해 보정**

**AWQ:**
- 레이어 단위로 **입력 활성화 값 확인**
- 가중치의 **활성화 정도를 반영**해 양자화 적용

**공통 특징:**
- 성능 저하 최소화
- 중요도 기반 양자화로 **효율성 ↑**
- 직접 구현 필요 없음 → 이미 **라이브러리 제공**됨
- 모델 개발사도 양자화 버전과 함께 공개 → 사용자 바로 활용 가능

---

### 5-9. Quantization(양자화) — 그 외

#### 추가 양자화 방식: GGUF, GGML (p.40)

- **CPU 추론 최적화 포맷**
- Python 아님 → **C++ 기반** LLM 실행 라이브러리
- **라마 CPP 기반 포맷**
- 전체 모델 **단일 파일로 저장** 가능 → 배포 간편
- **올라마(Ollama) 프로그램** 활용 시 손쉬운 서빙 가능

#### 양자화 친화적 학습 기법: QAT (p.41)

- **QAT (Quantization Aware Training)**
  - 학습 단계에서 **양자화 고려**
  - 중간중간 양자화 적용/해제 반복하며 학습
  - 구글 **Gemma3에 적용** → 양자화 친화적 학습 구현

#### 동적 파라미터 선택: Gemma 3n (p.42)

- 입력 처리 시 **동적으로 파라미터 선택**
- 온디바이스 환경에서 **GPU 소모 최소화**
- 경량화·효율적 실행 목적

---

### 5-10. 대표적 모델 양자화 방법

#### 다양한 양자화 방식 현황 (p.43)

- Hugging Face 등에서 이미 **AQLM ~ HEX**까지 다양한 양자화 방식 출시
- 현재도 지속적으로 **새로운 방식 연구·출시** 중
- 목표: **성능 안정화 + 모델 사이즈 축소**

주요 양자화 방식 지원 현황 (CPU, CUDA GPU, ROCm GPU, Metal, Intel GPU, Torch compile() 등):
- AQLM, AutoRound, AWQ, bitsandbytes, compressed-tensors, EETQ, GGUF/GGML, GPTQModel, AutoGPTQ, HIGGS 등

#### 핵심 질문 (p.44)

> **큰 모델을 양자화할까, 작은 모델을 그냥 쓸까?**

- **일반적으로는 큰 모델 양자화가 유리**
  1. 레이어 개수가 많음
  2. 차원이 큼
  3. 컨텍스트 길이가 김
  4. 많은 데이터를 학습함

---

## 6장 — 추론과 멀티모달 모델

### 6-1. LLM 지능 향상 비결: Reasoning

#### Reasoning 개요 (p.46)

- LLM에게 **생각할 시간(과 텍스트 공간)을 주면**, 어려운 문제를 잘 풀 수 있다
  - 수학 문제, 코딩 문제 등
- 길게 말하는 것에 익숙해진 최근의 모델 = **Reasoning 모델**

#### LLM 벤치마크 리더보드 (p.47–48)

| Model | Creator | Context Window | AI Intelligence Index | Median First Chunk (s) |
|-------|---------|---------------|----------------------|----------------------|
| GPT-5 (high) | OpenAI | 400k | 69 | 71.26 |
| GPT-5 (medium) | OpenAI | 400k | 68 | 34.03 |
| Grok 4 | xAI | 256k | 68 | 12.18 |
| o3-pro | OpenAI | 200k | 68 | 137.86 |
| o3 | OpenAI | 200k | 67 | 13.28 |
| GPT-5 mini (high) | OpenAI | 400k | 65 | 86.41 |
| o4-mini (high) | OpenAI | 200k | 65 | 51.97 |
| Gemini 2.5 Pro | Google | 1m | 65 | 29.73 |
| GPT-5 mini (medium) | OpenAI | 400k | 64 | 29.68 |
| Qwen3 235B 2507 (Reasoning) | Alibaba | 256k | 64 | 0.00 |
| GPT-5 (low) | OpenAI | 400k | 63 | 14.24 |
| gpt-oss-120B (high) | OpenAI | 131k | 61 | 0.48 |

- **Median First Chunk (s):** 첫 번째 출력을 생성하는 데 걸리는 시간
- 출력을 하는 동안 **오류를 고치고 답을 준비**
- Reasoning 예시 (p.49): 호수 부피 구하기 질문 → 모델이 "Thinking..." 과정을 거쳐 답변

### 6-2. Long Thinking Makes Perfect

#### 언어 모델 = 패턴 (p.50–51)

- **패턴 강화 학습**으로 긴 시간동안 생각하고 어려운 문제의 해답 탐색 가능
- **Thinking을 하는 모델이 상위권에 위치**

#### Thinking 토큰 Budget 설정 (p.52)

- Thinking 토큰 (`<think>` `</think>`)의 **Budget 설정**, **Hybrid 모드** 등으로 발전
- 예: Qwen 3 모델 (`/no_think`) — 상황에 따라 thinking을 끄거나 켤 수 있음

#### Open 모델도 상위권 (p.53)

- Qwen3 235B 2507 (Reasoning), GPT-5 (low), gpt-oss-120B (high) 등 **Open 모델**도 리더보드 상위

---

### 6-3. Thinking 모델은 어떻게 문제를 푸는가?

#### OpenAI-o1의 출력을 6개의 패턴으로 분류 (p.54–55)

| 패턴 | 설명 |
|------|------|
| Systematic Analysis | 전체적 구조 파악 |
| Method Reuse | 알려진 문제 해결법 적용 |
| Divide and Conquer | 하위 문제 분할 |
| **Self-Refinement** | **자체 평가와 수정** |
| Context Identification | 추가 문맥 활용 |
| Emphasizing Constraints | 제약 조건 강조 |

- **Self-Refinement**이 핵심: LLM은 자체 평가를 통해 **오류를 검출하고 수정**
- Thinking의 모든 과정은 **텍스트로 표출**됨

#### Gemini의 접근 (p.56)

- 질문 → 답변 후 요구사항에 적절한지 계산 → **적절한 답변으로 수정**

#### Over Thinking (p.56)

- Thinking을 많이 하면 **성능이 저하되는 현상**

---

### 6-4. LLM의 발전 방향: 멀티모달 LLM (LMM)

#### LLM 모델에 Vision 능력을 결합하는 방향 (p.57)

- GPT, Gemini, Qwen-VL, Gemma 시리즈
- 이미지 생성 → **이미지 처리 인코더 결합** → 인코더와 LLM 연결

#### LLM 사이에 Vision 모델을 넣는 방식 (p.57)

- **Llama 3.2**
- 멀티모달 데이터를 통해 **텍스트로 말하기 가능**

#### 멀티모달 → Native LMM 연구 (p.58)

- **GPT-4o**의 이미지/음성 생성 기능
- **Qwen-VLo**의 Thinker-Talker 구조

#### Qwen2.5-VL 아키텍처 (p.59)

- Vision Encoder에서 **이미지와 비디오 처리**
- Native Resolution Input 지원
- Window Attention + Full Attention 조합 구조

---

### 6-5. LLM의 발전 방향: sLLM (small Large Language Model)

#### sLLM & Domain-Specific LLMs (p.60)

- sLM = sLLM
- **파라미터 크기의 차이**
  - 큰 LLM: Kimi-K2 (1000B), Qwen 3 Reasoning (235B)
  - sLLM: Qwen 3 (0.7B)
- 성능의 차이 존재

#### 모델 경량화 기법: 가지치기(Pruning) (p.61–62)

- **가로로 자르기:** 레이어 ↓
- **세로로 자르기:** 차원 ↓
- 반복적인 LLM 모델의 파라미터 중 **영향이 적은 파라미터 → 제거**
- 예: **NVIDIA 네모트론(Nemotron)**

#### 증류(Distillation) (p.63)

- 큰 모델의 출력으로 **작은 모델 학습**
- **머신러닝의 증류 방식:** 아웃풋 학습 + 큰 모델 출력 결과 → 작은 모델이 큰 모델의 사고 방식 이해
- **LLM의 증류 방식:** 큰 모델 데이터 → 작은 모델 **파인 튜닝**

#### sLLM 활용 사례 (p.64–65)

- Gemma 3 (236M, 1B, 4B), Llama 3.2 (1B/3B), Qwen 3 (0.6B/1.5B) 등
- **엣지 디바이스나 CPU 환경** 고려
- **파인 튜닝을 통한 단일 작업 성능 향상** OR **도메인 특화 모델 구성**
- 예: LLM Agent의 모듈화 구성
  - 정해진 규칙이 있는 기업용 코드
  - 코드의 규칙 준수를 판단하는 sLLM 배치

#### Deepseek R1의 연구 결과 (p.66–67)

**AIME 2024 벤치마크:**

| Model | pass@1 | cons@64 |
|-------|--------|---------|
| GPT-4o-0513 | 9.3 | 13.4 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 |
| OpenAI-o1-mini | 63.6 | 80.0 |
| QwQ-32B-Preview | 50.0 | 60.0 |
| DeepSeek-R1-Distill-Qwen-1.5B | 28.9 | 52.7 |
| DeepSeek-R1-Distill-Qwen-7B | 55.5 | 83.3 |
| DeepSeek-R1-Distill-Qwen-14B | 69.7 | 80.0 |
| DeepSeek-R1-Distill-Qwen-32B | 72.6 | 83.3 |
| DeepSeek-R1-Distill-Llama-8B | 50.4 | 80.0 |
| DeepSeek-R1-Distill-Llama-70B | 70.0 | 86.7 |

→ **큰 모델의 사고 방식을 학습할 수 있는 sLLM**

---

### 6-6. Distillation 예시) Qwen 3의 학습 파이프라인 (p.68)

**긴 추론의 Cold Start + RL + Distillation → DeepSeek 스타일**

**Flagship Models 학습 과정:**
1. Base Models → **Stage 1:** Long-CoT Cold Start
2. → **Stage 2:** Reasoning RL
3. → **Stage 3:** Thinking Mode Fusion
4. → **Stage 4:** General RL
5. → 결과: **Qwen3-235B-A22B, Qwen3-32B**

**Lightweight Models:**
- Base Models → **Strong-to-Weak Distillation** → Qwen3-30B-A3B, 14B/8B/4B/1.7B/0.6B

---

### 6-7. LLM의 발전 방향: Tool Calling과 Agent

#### 텍스트로 답을 추출하는 LLM (p.69)

- "JSON 형식으로 출력해."
- 텍스트 → 파싱 → 데이터 (JSON 형식의 데이터)

#### Agent: LLM에 Tool Calling 능력 탑재하기 (p.70–71)

- LLM이 **외부 함수나 API를 호출**하도록 프롬프트를 통해 전달
- Claude, Cursor 등과 연계되어 **25년 3월부터 다양한 어플리케이션** 개발되는 중

#### Model Context Protocol (MCP): Tool Calling의 표준화 (p.72)

- **MCP** = AI Application이 Database, Web APIs, GitHub, Slack, Gmail, Local Filesystem 등과 **표준화된 프로토콜로 통신**

#### OpenAI Agent / Anthropic Computer Use (p.73)

- 에이전트 기반 AI 시스템의 발전

---

### 6-8. LLM 어플리케이션 만들기 & 활용하기

#### 모델의 성능-크기 상관관계 (p.74)

- 학습 데이터 양 / Context Window / 지시 수행 능력에서 모두 큰 차이
- LLM 어플리케이션에서는 **고성능이 필요하지 않은 작업들도 존재**
- 일반적으로, **쓸만한 성능의 모델은 10B 이상** (더 작은 모델의 활용 가능성도 높아짐)

#### LLM 어플리케이션의 성능 보완 요소 (p.75)

1. **프롬프트 엔지니어링**
2. **파인 튜닝**
3. **RAG**
4. **Agent**

#### LLM 어플리케이션 활용하기 (p.76–77)

- **비정형 텍스트 데이터에 대한 이해 및 처리 능력** 필요
  - 데이터 전처리가 필요함
  - 문제에 맞는 데이터와 **결과를 평가하는 방식** 필요 (정량적, 정성적 평가 방식)
- 현업에서는 **LLM 단독보다는 실제 데이터를 기반으로 수행**
  - 결과에 대한 평가 방식, **데이터를 수급하는 파이프라인** 등이 현재 LLM 앱 개발에서 가장 중요한 부분

---

## 7장 — LLM API 소개

### 7-1. LLM 프로그래밍을 위한 모델 호출

#### 상용 LLM API 활용 (p.79)

- **OpenAI, Anthropic, Google** 등
- 자체 API를 통한 호출
- Amazon/Azure Cloud를 통한 호출

#### 오픈 모델 사용 (p.80)

- Gemma, LLaMA, Mistral, HyperCLOVA, AideX 등
- **Transformers** 바로 호출
- 라이브러리 활용: **Ollama, vLLM**
- 클라우드 API 활용: **Groq, Cerebras**
- **OpenRouter**

### 7-2. OpenAI SDK Library (p.81)

- Python / Node.js 지원
- `curl https://api.openai.com/v1/chat/completions` 엔드포인트
- 인증: `Authorization: Bearer $OPENAI_API_KEY`

### 7-3. OpenAI API Key (p.82–83)

- **서버 통신을 위한 인증 키**
- **Billing 등록 후 결제 가능**
- **대시보드를 통한 사용량 트래킹**

### 7-4. [실습] OpenAI API 활용 (p.84–85)

#### Chat API를 통한 GPT 프롬프팅

- 텍스트/이미지/보이스 지원

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    temperature=0.2,
    # temperature: 무작위 출력을 조절: (0-2) 0에 가까울수록 정해진 답변을 수행
    max_tokens=512,
    # 출력 최대 토큰 수 조절: 초과할 경우 자름
    # n = 1
    # 여러 개의 출력 가능
)
print(response)
```

**핵심:** prompt, role, temperature를 잘 구성 시 다양한 결과 출력 가능

---

## 8장 — LangChain

### 8-1. OpenAI API의 한계 (p.87–88)

- Chat API를 통한 GPT 프롬프팅 (텍스트/이미지/보이스)
- **Model, Provider 변경 시 활용이 어려움**
- Open Model 사용 시 **별도 Library 필요** → 실행할 때마다 코드 수정 필요

### 8-2. LangChain이란? (p.89–90)

- **LLM 어플리케이션 개발 프레임워크**
- 다양한 도구와 **추상화(Abstraction)** 지원
- ML이나 AI에 대한 배경 지식이 적어도, LLM 기반의 어플리케이션 구성 가능
- **Open·Closed 모델 모두 활용 가능**
- 자세한 공식 문서 및 다양한 예제 코드 존재
- Gemini, Claude를 이용한 바이브 코딩 구현 용이

### 8-3. LangChain + LLM Provider (p.91–92)

#### LLM Provider별 별도의 라이브러리로 빠른 업데이트

- `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`
- 각 라이브러리를 불러오면 실행 가능

#### HuggingFace 연동

- `langchain-huggingface`
- https://huggingface.co
- 모델 주소로 바로 연결 가능 (다양한 호환성 제공)

### 8-4. LangChain VS OpenAI API (p.93)

**OpenAI API:**
- 개별 API만으로도 어플리케이션 구현 가능

**LangChain 사용 시:**
- OpenAI와 오픈 모델을 포함한 대부분의 모델에서 **동일한 인터페이스로 사용 가능**
- OpenAI 내부 기능 대부분 사용 가능
- **코드가 API SDK에 비해 간결**
- 단점: 약간 느림, 원하는 기능 구현에 제약 있음

### 8-5. ChatOpenAI in LangChain (p.94–95)

#### LangChain-OpenAI 라이브러리를 통해 연결

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.5, max_tokens=1024)
```

#### invoke()를 통해 실행

- 대부분의 LangChain 구성요소는 **invoke**로 실행됨

```python
question = '''전 세계적으로 흥행한 영화에 나오는 유명한 명대사를 하나 알려주세요.
대사가 나온 배경과 의미도 설명해 주세요.'''

llm.invoke(question)
```

### 8-6. LangChain-Gemini 연결 (p.96)

#### Google AI Studio

- https://aistudio.google.com 연결
- API 키 발급 후 **환경 변수를 통해 연결**
- **LangChain-google-genAI 라이브러리** 활용

### 8-7. Ollama (p.97)

- https://ollama.com/
- **GGUF 양자화 포맷 전용 라이브러리**
- **Llama.cpp 기반** 라이브러리로 CPU 추론도 가능

### 8-8. [실습] 다양한 LLM Provider로 LangChain 첫 체인 만들기 (p.98)

- **ChatOpenAI / ChatGoogleGenerativeAI / ChatOllama** 써보기
- 구글 코랩 GPU 필요 (무료 라이센스: T4)

---

## 핵심 요약 정리

### 주요 오픈소스 LLM 비교

| 모델 | 개발사 | 주요 사이즈 | 핵심 특징 |
|------|--------|-----------|----------|
| Qwen3 | Alibaba | 0.6B~235B (MoE) | Dense/MoE 혼합, Thinking 모드 지원 |
| Mistral | Mistral AI | 7B~123B | Sliding Window Attention, MoE (Mixtral) |
| Gemma 3 | Google | 1B~27B | 멀티모달, SigLIP 기반 Vision |
| Llama 3.x | Meta | 1B~405B | 범용, 멀티모달 (3.2부터) |

### 국내 LLM 비교

| 모델 | 개발사 | 주요 사이즈 | 핵심 특징 |
|------|--------|-----------|----------|
| Kanana | 카카오 | 2.1B, 8B | 한국어 처리 우수, Apache 라이선스 |
| HyperCLOVA | 네이버 | 0.5B~14B | 한국어 리즈닝, 멀티모달 지원 |
| EXAONE | LG | 1.1B~32B | 국내 최상위권 벤치마크, 한국어 리즈닝 특화 |
| Solar | Upstage | 10.7B~31B | DUS 기법, 리더보드 1위 기록 |

### VRAM 계산 공식

```
16bit 모델: 파라미터 수(B) × 2 = 필요 VRAM(GB)
32bit 모델: 파라미터 수(B) × 4 = 필요 VRAM(GB)

안정적 추론: 기본 VRAM × 1.5배
학습(Training): 기본 VRAM × 10배 이상
```

### 양자화 기법 요약

| 기법 | 방식 | 특징 |
|------|------|------|
| INT8 (고전적) | 등간격 매핑 (0~255) | 단순하지만 성능 손실 |
| NF4 (QLoRA) | 분위수 기반 분할 | 원래 값 보존률 높음 |
| GPTQ | 중요도 기반 차등 양자화 | 앞단 오류를 뒷단에서 보정 |
| AWQ | 활성화 값 기반 양자화 | 가중치 활성화 정도 반영 |
| GGUF/GGML | C++ 기반 CPU 최적화 | 단일 파일, Ollama 지원 |
| QAT | 학습 시 양자화 고려 | Gemma3 적용 |

### LLM 앱 개발 기술 스택

```
모델 호출: OpenAI API / Anthropic API / Google AI Studio
프레임워크: LangChain (+ langchain-openai / langchain-anthropic / langchain-google-genai)
로컬 실행: Ollama (GGUF), vLLM
클라우드: Groq, Cerebras, OpenRouter
성능 보완: 프롬프트 엔지니어링, 파인 튜닝, RAG, Agent
표준화: MCP (Model Context Protocol)
```
