# System One Model PoC

로컬 Laya와 System One 계열 외부 API를 같은 판정 UI에서 선택·호출하는 작은 PoC입니다.
일반 생성형 챗이 아니라 상태(`state`)에 대한 정형화된 질문(`questions`)을 실행하고,
선택지 확률·참/거짓 확률·순서형 점수를 비교하는 용도입니다.

## 지원 모델

| UI 표시 | 실행 경로 | 모델 ID |
|---|---|---|
| `(local) Laya Multilingual` | 이 Mac의 MPS 또는 CPU | `multilingual` |
| `(decisions-api.org) Jev (1.13)` | Decisions API | `typesafe/jev-1.13` |
| `(typesafe.ai) Jev (latest)` | TypeSafe API | `jev-latest` |

두 Jev 항목은 모델 계열은 같지만 서빙 provider와 API 키가 다릅니다. 하나는 Decisions API가
`typesafe/jev-1.13`을 서빙하고, 다른 하나는 TypeSafe API를 `jev-latest`로 직접 호출합니다.

## 설치

Python 3.13 가상환경을 기준으로 합니다.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

로컬 Laya는 Hugging Face 캐시의 모델 가중치를 사용합니다. 첫 실행에서 필요한 가중치가 없으면
Laya/Hugging Face 런타임이 자동으로 받습니다.

## 외부 API 키

프로젝트 루트의 `.env`에 사용할 provider의 API 키를 설정합니다.

```dotenv
DECISION_ORG_API_KEY=replace-with-your-key
TYPESAFE_API_KEY=replace-with-your-key
```

`app.py`는 서버 시작 시 `.env`를 한 번 읽습니다. 이미 process environment에
같은 이름의 키가 있으면 `.env`가 덮어쓰지 않습니다. 키는 서버의
`Authorization: Bearer ...` 헤더에만 쓰이고 브라우저·health 응답·판정 결과에 포함되지 않습니다.

각 키가 없으면 해당 provider의 Jev 항목만 비활성화되고, 다른 provider와 로컬 Laya는 계속 사용할 수 있습니다.

## 실행

```bash
.venv/bin/python app.py
```

기본 주소는 `http://127.0.0.1:8766`입니다.

```bash
# CPU로 로컬 Laya 실행
.venv/bin/python app.py --device cpu

# 다른 포트 사용
.venv/bin/python app.py --port 9000
```

서버는 외부 API를 자동 재시도하지 않습니다. Jev를 선택하면 입력·질문·기준이 선택한
Decisions 또는 TypeSafe API로 전송되고, 성공한 요청은 credit을 사용할 수 있습니다.

## 판정 형식

- `choice`: 라벨별 확률과 최고 확률 선택지
- `noul`: 참/거짓 확률과 UI threshold 적용 결과
- `score`: 순서가 있는 등급별 확률과 기댓값

선택지와 점수 등급은 한 줄에 `라벨: 기준` 형식으로 입력합니다.

```text
결제: 결제, 중복 청구, 취소 또는 환불 요청
계정: 로그인, 계정 접근, 비밀번호 또는 인증 문제
기술지원: 오류, 장애, 기능 동작 또는 서비스 연동 문제
```

UI에는 choice·noul·score의 차이를 바로 확인할 수 있는 업무 예제 프리셋이 있습니다.

## Decisions API 계약

서버가 보내는 요청은 다음 구조입니다.

```json
{
  "model": "typesafe/jev-1.13",
  "state": {"input": "입력 텍스트"},
  "questions": {
    "judgement": {
      "type": "choice",
      "instructions": "판정 질문",
      "criteria": {"label": "criterion"}
    }
  }
}
```

원격 요청 헤더는 `Authorization`과 `Content-Type` 두 개입니다. 2026-10-01 직접 curl 실행에서
`typesafe/jev-1.13`이 `code: 0`과 `answers.judgement`를 반환하는 것을 확인했습니다.

## TypeSafe API 계약

TypeSafe 직접 호출은 `POST https://api.typesafe.ai/v1/systemone`에 같은 `state`와 `questions`
구조를 보내되 모델 ID로 `jev-latest`를 사용합니다. 응답은 Decisions의 `code`/`data.result`
래퍼 없이 `model`, `answers`, `usage`가 최상위에 오는 구조이며, 서버에서 공통 결과 형식으로
정규화합니다. 자세한 필드 정의는 [TypeSafe API 문서](https://docs.typesafe.ai/api)를 따릅니다.

## 검증

네트워크·크레딧·모델 로드 없이 요청 변환과 UI 계약을 검사합니다.

```bash
.venv/bin/python -m py_compile app.py
.venv/bin/python app.py --self-test
```

self-test는 임시 dotenv, fake Decisions/TypeSafe transport, 응답 정규화, 키 비노출,
모델 catalog·HTML route를 검사하며 실제 외부 API를 호출하지 않습니다.

## 구조

```text
.
├── app.py
├── requirements.txt
├── README.md
└── .env                 # local only, git ignored
```
