# 📘 Samsung SDS — 한번에 끝내는 AI 개발: Python 기초

> **교재 1 | Python 기초** — 전체 52페이지 분석 및 정리

---

## 목차

| 번호 | 섹션 | 페이지 |
|---|---|---|
| 1 | Python 개발 환경 구축 | p.1 ~ p.10 |
| 2 | 변수와 자료형 | p.11 ~ p.27 |
| 4 | 함수와 모듈 | p.28 ~ p.37 |
| 5 | 자료구조와 파일 입출력 | p.38 ~ p.44 |
| 6 | 객체지향 프로그래밍 기초 | p.45 ~ p.52 |

> ※ 섹션 3은 교재에 포함되어 있지 않으며, 번호가 2 → 4로 건너뜁니다.

---

## 1. Python 개발 환경 구축 (p.1 ~ p.10)

### 1-1. Python이란? (p.2)

- AI와 머신러닝 개발에 **가장 많이 쓰이는 프로그래밍 언어**

### 1-2. Python의 특징 (p.3)

- **문법이 간결**하고 배우기 쉬움
- **방대한 라이브러리**와 커뮤니티 지원
- **범용 언어**: 웹 개발, 데이터 분석, AI/ML, 자동화 등
  - 특히 **LLM에서 가장 많은 활용성**을 보임

### 1-3. Python & VS Code 설치 및 기본 설정 (p.4 ~ p.5)

#### Python 설치 (p.4)

- **Python.org**에서 최신 버전 다운로드 (예시: Python 3.13.7)
- 설치 시 **"Add python.exe to PATH"** 옵션 반드시 체크

#### VS Code 설치 (p.5)

- **https://code.visualstudio.com** 에서 무료 다운로드
- 오픈소스 AI 코드 에디터

### 1-4. Python 실행하기 (p.6)

#### 터미널에서 실행

- **cmd**(윈도우), **터미널**(리눅스)에서 실행

```bash
python --version
python
>>> print("Hello, Python!")
```

#### VS Code에서 실행

- 새 파일 작성 후 **Shift + F5** 또는 터미널에서 실행
- 실행 명령: `python 파일명.py`

### 1-5. 가상 환경 만들기 (p.7)

#### 프로젝트별 독립적인 환경 만들기

- 별도의 라이브러리 설치 시 버전이 다르면 **실행이 안 되거나 충돌**이 발생할 수 있음
- **프로젝트별 별도의 가상 환경**을 만들어서 개발

### 1-6. .py 파일과 .ipynb 파일 이해하기 (p.8 ~ p.9)

#### .py (Python Script) — p.8

- 일반 Python 코드 파일
- **실제 개발에 적합**

#### .ipynb (Jupyter Notebook) — p.9

- **JSON 형식**으로 저장된 노트북 파일
- **코드 + 결과 + 문서(Markdown)** 함께 관리 가능
- **데이터 분석, 교육, 실험**에 적합

### 1-7. 실습: Python 시작하기 (p.10)

- Python과 VS Code 설치
- 프로젝트 폴더와 가상 환경(선택) 구성
- ipynb 파일과 py 파일 만들고 실행하기
- 첫 번째 실습파일: **'파이썬 개발 환경 설정하기'**

---

## 2. 변수와 자료형 (p.11 ~ p.27)

### 2-1. 변수(Variable) (p.12 ~ p.14)

#### 변수란? (p.12)

- **데이터를 저장하는 메모리 공간** (데이터를 담는 상자 혹은 그릇)
- Java/C 언어의 변수 정의 예시:
  ```c
  int x = 1;  // 정수 타입 명시, 값 할당
  ```

#### Python의 동적 타이핑 (p.13 ~ p.14)

- **실행 시점에 타입이 결정됨**
- Python의 변수 정의:
  ```python
  x = 'Python'  # 타입 선언 없이 바로 할당
  ```
- `var`, `let`, `const` 등의 키워드 **불필요**
- **타입 변경 가능** (같은 변수에 다른 타입의 값 재할당 가능)

### 2-2. PEP8: 변수 명명 규칙 (p.15)

#### 의미 있는 이름 사용

| 예시 | 평가 |
|---|---|
| `a`, `b` | ❌ (의미 불명확) |
| `User_name` | ✅ (의미 명확) |

#### 규칙 요약

- **예약어 사용 금지**: `def`, `if` 등
- 특수문자는 **언더바(`_`)만 허용**
- **상수는 대문자로 작성**: `MAX_RETRY = 3`

### 2-3. Python의 기본 자료형 (p.16 ~ p.21)

#### 숫자형 — int (정수형) (p.16)

| 항목 | 내용 |
|---|---|
| 예시 | `1234` |
| 특징 | 메모리 제한 없는 큰 수 표현 가능 |
| 용도 | 카운터, 인덱스, ID 값 |

#### 숫자형 — float (실수형) (p.17)

| 항목 | 내용 |
|---|---|
| 예시 | `0.568` |
| 특징 | 부동소수점 방식 (정밀도 한계 존재) |
| 용도 | 비율, 측정값, 과학 계산 |

#### 문자형 — str (문자열) (p.18)

| 항목 | 내용 |
|---|---|
| 예시 | `'abcd'` |
| 특징 | 불변(immutable) 시퀀스 |
| 용도 | 텍스트 데이터, 로그 메시지 |

#### bool (불리언) (p.19)

| 항목 | 내용 |
|---|---|
| 값 | `True` / `False` |
| 내부 표현 | 1 / 0 |
| 특징 | 논리연산자 |
| 용도 | 플래그, 조건 판단 |

#### None (널 타입) (p.19)

| 항목 | 내용 |
|---|---|
| 값 | `None` |
| 의미 | 값이 없음을 나타냄 |
| 용도 | 초기값, 실패 표시 |

#### List (p.20 ~ p.21)

- `[1, 'a', '', 'B']` 처럼 값을 나열하는 방식
- **대괄호**로 감싸기
- 안에 들어가는 것은 **제한이 없음** (int, float, str, bool, none, list 등 모두 가능)
- **순서가 있고, 값 변경 가능** (mutable)

#### Tuple (p.21)

- **소괄호** 안에 콤마 나열: `(23, 92)`
- **순서가 있고, 값 변경 불가** (immutable)

#### Dictionary (p.21)

- **{키:값}** 형태: `{'name':'서두석', 'age':27}`
- **키로 값을 찾는 구조**

### 2-4. 타입과 형변환(Casting) (p.22)

#### 타입 확인 함수

| 함수 | 설명 | 인자 |
|---|---|---|
| `type()` | 변수의 타입 확인 | 변수 |
| `isinstance()` | 특정 타입인지 확인 | 변수, 타입 |

#### 형변환 예시

| 변환 | 결과 |
|---|---|
| `int("123")` | `123` (정수) |
| `str(100)` | `'100'` (문자열) |

### 2-5. 조건문(Conditional) (p.23 ~ p.25)

#### if-elif-else 구문 (p.23)

- **True/False**로 처리되는 조건식을 이용해 프로그램 진행의 **분기 저장**

```python
if 조건1:
    # 조건1이 참일 때
elif 조건2:
    # 조건1은 거짓, 조건2가 참일 때
else:
    # 모든 조건이 거짓일 때
```

#### 논리연산자 (p.24)

| 연산자 | 의미 |
|---|---|
| `and` | 모든 조건이 참 |
| `or` | 하나라도 참 |
| `not` | 조건 반전 |

#### Truthy vs Falsy (p.25)

**Falsy 값들** (거짓으로 평가):
- `None`
- `False`
- `0`, `0.0`
- `""`, `''`, `""""""` (빈 문자열)
- `[]`, `()`, `{}` (빈 컨테이너)

**Truthy 값들** (참으로 평가):
- 위 Falsy를 **제외한 모든 값**

### 2-6. For 반복문(Loop) (p.26)

- 특정 시퀀스를 순회하며, **같은 코드 작업을 반복**하는 구문

```python
for 변수 in 이터러블:
    # 반복 실행할 코드
```

#### 함께 쓰이는 내장 함수

| 함수 | 설명 |
|---|---|
| `range()` | 숫자 시퀀스 만들기 |
| `enumerate()` | 인덱스와 값 동시 순회 |
| `zip()` | 여러 시퀀스 동시 순회 |

### 2-7. While 반복문(Loop) (p.27)

- **조건이 True인 동안** 작업을 반복하는 구문
- ⚠️ **조건 변경 로직 필수!** (무한 루프 방지)

```python
while 조건:
    # 조건이 참인 동안 반복
    # 조건 변경 로직 필수!
```

#### 반복 제어 키워드

| 키워드 | 설명 |
|---|---|
| `break` | 반복문 즉시 종료 |
| `continue` | 다음 반복으로 건너뛰기 |
| `pass` | 아무 동작 안 함 (자리 표시) |
| `else` | 정상 종료 시 실행 (break 없이) |

---

## 4. 함수와 모듈 (p.28 ~ p.37)

> ※ 섹션 번호가 교재 원본 기준 4번입니다 (3번 섹션은 교재에 미포함).

### 4-1. 함수(Function)란? (p.29)

- 특정 코드나 기능을 반복 사용하기 위해
  **입력 → 처리 → 출력을 반환하는 코드 블록**

### 4-2. 효율적인 코드 설계 원칙 (p.30 ~ p.32)

- **DRY (Don't Repeat Yourself)**: 같은 코드 반복 피하기
- 코드는 **재사용성**과 **가독성**을 고려하여 설계되어야 함

#### Bad vs Good 예시

```python
# Bad: 같은 로직 반복
user_input1 = input1.strip().lower().replace(" ", "")
user_input2 = input2.strip().lower().replace(" ", "")

# Good: 함수로 추상화
def normalize_text(text):
    return text.strip().lower().replace(" ", "")

user_input1 = normalize_text(input1)
user_input2 = normalize_text(input2)
```

→ 하나의 함수를 만들고 호출하는 방식으로 **2개의 반복 코드를 해결**

### 4-3. 함수의 구성 요소 (p.33)

```python
def function_name(parameter1, parameter2):  # 시그니처
    """함수 설명 (docstring)"""              # 문서화
    # 함수 본문
    result = parameter1 + parameter2         # 로직
    return result                            # 반환
```

- **반환값**: 함수가 실행된 후 결과로 돌려주는 값

### 4-4. 함수의 매개변수와 반환값 설정하기 (p.34)

#### 매개변수(argument) 종류

| 종류 | 설명 | 예시 |
|---|---|---|
| **위치 인자** | 순서에 따라 값이 전달 | `greet(name, age)` |
| **키워드 매개변수** | 이름을 지정하여 순서 무관 | `greet(age=25, name="김철수")` |
| **기본값 매개변수** | 기본값이 있으면 생략 가능 | `def connect(host="localhost", port=3306)` |
| **가변 매개변수** | 개수가 정해지지 않은 인자 | `def log(*args, **kwargs)` |

- `*args`: 위치 인자 → **튜플**로 전달
- `**kwargs`: 키워드 인자 → **딕셔너리**로 전달

### 4-5. Python Module (p.35)

- Python에서 **py 파일 하나**는 모듈을 의미
- 단독으로도 실행할 수 있지만, **import 구문을 통해 다른 코드에서 사용 가능**
- 다른 모듈의 함수와 변수를 **불러와 재사용 가능**

```python
# math_utils.py (모듈)
def add(a, b):
    return a + b

# main.py
import math_utils
print(math_utils.add(2, 3))  # 5
```

### 4-6. 패키지(Package)와 라이브러리(Library) (p.36)

#### 패키지(Package)

- 하나의 폴더에 **여러 모듈을 묶어서 저장**하는 형식

```
my_package/
    __init__.py
    math_utils.py
    string_utils.py
```

```python
from my_package import math_utils
```

#### 라이브러리(Library)

- **모듈과 패키지를 모두 포함**하는 재사용 가능한 코드 묶음
- 예: Numpy, Pandas, Transformers 등
- 기본 라이브러리가 아닌 경우 **설치 필요**: `pip install numpy`

### 4-7. 실습: 다양한 함수와 모듈 연결하기 (p.37)

- `def`를 이용한 다양한 함수 만들기
- 함수의 매개변수와 Return 방식 이해하기
- 모듈과 패키지, 라이브러리 구조 이해하기
- 구성한 함수 모듈화하여 활용하기

---

## 5. 자료구조와 파일 입출력 (p.38 ~ p.44)

### 5-1. List Comprehension: 반복문의 특이한 형태 (p.39)

- **코드를 간결하게** 만드는 방식

#### 기존 방식

```python
result = []
for x in range(10):
    if x % 2 == 0:
        result.append(x**2)
```

#### List Comprehension

```python
result = [x**2 for x in range(10) if x % 2 == 0]
```

→ 4줄의 코드가 **1줄**로 축약

### 5-2. 파일 입출력 (p.40)

#### open()을 통해 파일 읽기/쓰기

| 모드 | 설명 |
|---|---|
| `"r"` | 읽기 |
| `"w"` | 쓰기 |
| `"a"` | 추가 |
| `"b"` | 바이너리 |

```python
# 파일 읽기
with open("example.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 파일 쓰기
with open("example.txt", "w", encoding="utf-8") as f:
    f.write("Hello, File!")
```

### 5-3. JSON (JavaScript Object Notation) (p.41)

#### JSON이란?

- **경량 데이터 교환 형식** (사람도 읽기 쉽고, 기계도 처리하기 쉬움)
- 파이썬의 **dict와 유사한 구조**

```python
import json

# dict → JSON 문자열
data = {"name": "철수", "age": 25}
json_str = json.dumps(data, ensure_ascii=False, indent=2)

# JSON 문자열 → dict
data = json.loads(json_str)
```

### 5-4. 예외 처리 (Exception Handling) (p.42 ~ p.43)

#### ZeroDivisionError 예시 (p.42)

```python
today = 100
yesterday = 0

change_rate = (today - yesterday) / yesterday * 100  # ← 에러!
print(change_rate)

# ZeroDivisionError: division by zero
```

- if문으로 미리 방어하기보다, **실행 중 예외가 생긴 시점에 포착해 해결하는 것**이 파이썬의 코딩 스타일

#### try-except 구문 (p.43)

```python
try:
    # 실행할 코드
    result = 10 / 0
except ZeroDivisionError:
    print("0으로 나눌 수 없습니다.")
except Exception as e:
    print("예상치 못한 오류:", e)
else:
    print("에러 없이 실행됨")
finally:
    print("항상 실행되는 코드")
```

| 블록 | 역할 |
|---|---|
| `try` | 실행할 코드 |
| `except 특정에러` | 해당 에러 발생 시 처리 |
| `except Exception as e` | 기타 모든 에러 처리 |
| `else` | 에러 없이 정상 실행 시 |
| `finally` | 에러 유무와 상관없이 항상 실행 |

### 5-5. 실습: 자료구조와 파일 입출력 (p.44)

- List Comprehension 이해하기
- File Read/Write 패턴 활용하기
- JSON 방식의 데이터 처리 방법 이해하기
- Try-Except 방식 구문의 목적과 원리 이해하기

---

## 6. 객체지향 프로그래밍 기초 (p.45 ~ p.52)

### 6-1. Class (p.46 ~ p.48)

#### 객체지향 프로그래밍(OOP)의 기본 단위 (p.46)

Class는 두 가지를 포함:

| 구성 | 설명 |
|---|---|
| **데이터** (속성) | 객체가 가지는 정보 |
| **기능** (메서드) | 객체가 수행하는 동작 |

#### Class 구성 및 인스턴스 생성 (p.47 ~ p.48)

- **생성자 `__init__()`** 을 사용하여 인스턴스 초기화

```python
class Person:
    def __init__(self, name):
        self.name = name

    def greet(self):
        return f"안녕하세요. 저는 {self.name}입니다."

p = Person("철수")
print(p.greet())
# 출력: 안녕하세요. 저는 철수입니다.
```

- `Person("철수")` → 철수라는 이름을 가진 Person 인스턴스가 만들어짐
- `self.name`에 저장된 값이 `greet()` 메서드에서 참조됨

### 6-2. Class 변수 vs Instance 변수 (p.49)

#### Class 변수

- **모든 Instance가 공유**하는 변수
- **Class명**으로 접근

#### Instance 변수

- **각 객체마다 독립적**으로 존재
- `self.(변수명)`으로 정의

```python
class Counter:
    total = 0              # 클래스 변수

    def __init__(self):
        self.count = 0     # 인스턴스 변수
        Counter.total += 1
```

| 구분 | Class 변수 | Instance 변수 |
|---|---|---|
| 공유 범위 | 모든 인스턴스 공유 | 각 객체 독립 |
| 접근 방법 | `Class명.변수명` | `self.변수명` |
| 예시 | `Counter.total` | `self.count` |

### 6-3. Class 상속 (p.50)

- 기존 Class(**부모**)를 확장해 새로운 Class(**자식**) 정의
- **코드 재사용성** 향상

```python
class Animal:                          # Parent class
    def __init__(self, name):
        self.name = name

    def eat(self):
        return f"{self.name}가 먹고 있습니다."

# 재사용 + 기능 확장
class Dog(Animal):                     # Child class
    def bark(self):
        return f"{self.name}: 멍멍!"

class Cat(Animal):                     # Child class
    def meow(self):
        return f"{self.name}: 야옹~"
```

→ Dog, Cat 클래스는 Animal의 `__init__`과 `eat`을 **상속**받고, 각자 고유한 메서드를 추가

### 6-4. 오버라이딩과 super() (p.51)

#### 오버라이딩(Overriding)

- **Parent Method를 Child가 재정의**
- Parent의 기능을 그대로 가지고 옴

#### super()

- **직전 Method의 기능 불러오기**
- Parent Method 호출 → **기능 확장**

```python
class Parent:
    def greet(self):
        return "부모 인사"

class Child(Parent):
    def greet(self):
        return super().greet() + " 자식 인사"
```

→ `Child().greet()` 호출 시: `"부모 인사 자식 인사"` 반환

### 6-5. 실습: Python의 Class 이해하기 (p.52)

- Class의 기본 구성과 생성자, Method 구성하기
- 다양한 Class의 구성 요소 활용하기
- Class 상속 구조를 이용한 코드의 재사용성 이해하기

---

## 📋 전체 요약

| 섹션 | 핵심 키워드 | 요약 |
|---|---|---|
| **1. 개발 환경 구축** | Python, VS Code, 가상환경, .py, .ipynb | Python과 VS Code를 설치하고, 가상환경을 통해 프로젝트별 독립 환경 구성 |
| **2. 변수와 자료형** | 동적 타이핑, int/float/str/bool/None, List/Tuple/Dict, 조건문, 반복문 | Python은 동적 타이핑 언어이며, 다양한 기본/컬렉션 자료형과 제어문을 지원 |
| **4. 함수와 모듈** | def, DRY, 매개변수, return, import, Package, Library | 함수로 코드를 추상화하고, 모듈/패키지/라이브러리로 재사용성 확보 |
| **5. 자료구조와 파일 입출력** | List Comprehension, open(), JSON, try-except | 간결한 리스트 생성, 파일 I/O, JSON 데이터 처리, 예외 처리 패턴 |
| **6. 객체지향 프로그래밍** | Class, Instance, 상속, 오버라이딩, super() | OOP의 기본 단위인 Class를 통해 데이터와 기능을 캡슐화하고, 상속으로 코드 재사용 |
