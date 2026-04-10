# Stage 2 Organization

`stage_2/`는 "핵심 실행 흐름"과 "task별 보조 스크립트"를 분리하는 기준으로 정리한다.

## 분류 기준

루트(`stage_2/`)에는 Stage 2의 주요 코드만 둔다.

- 데이터셋 정의
- 핵심 모델 모듈
- 핵심 패치 로직
- 메인 forward / smoke test 코드
- Stage 2의 기본 입출력 포맷을 직접 다루는 대표 스크립트

즉, 다른 코드가 자주 import하거나 Stage 2의 기본 흐름을 설명할 때 바로 언급되는 파일은 루트에 남긴다.

하위 폴더에는 task 성격이 강한 보조 코드를 둔다.

- 특정 데이터 준비 작업용 스크립트
- 일회성 또는 배치 실행용 생성기
- 분석/점검/추적용 스크립트
- 특정 실험 단계에서만 쓰는 유틸리티

즉, "있으면 편하지만 Stage 2의 중심 축은 아닌 코드"는 task 또는 analysis 폴더로 내린다.

## 현재 구조

### 루트

아래 파일들은 Stage 2의 주요 흐름을 구성하므로 루트에 둔다.

- `stage2_dataset.py`
- `user_adapter.py`
- `patch_stage_c.py`
- `forward_only_stage2.py`
- `prepare_stage_c_latents.py`
- `build_latent_manifest.py`

### `tasks/`

`tasks/`는 결과물을 만들기 위한 작업 단위별 보조 스크립트를 둔다.

- `tasks/hf_manifest/`
  - Hugging Face 데이터셋에서 UID manifest를 만드는 작업
- `tasks/pair_assignment/`
  - pair pool 생성
  - Stage 2 pair assignment 생성
  - shard 단위 batch 실행

### `analysis/`

`analysis/`는 모델 구조 확인, 경로 추적, 샘플 점검처럼 분석 목적의 스크립트를 둔다.

- `analysis/stage_c/`
  - Stage C 구조 분석
  - target path 추적
  - clean target 관련 조사
- `analysis/user/`
  - user preference 샘플 확인

## 정리 원칙

새 파일을 추가할 때는 아래 기준을 따른다.

1. 다른 Stage 2 코드가 직접 import하는 핵심 모듈이면 루트에 둔다.
2. 특정 작업을 수행하는 생성기나 배치 실행기면 `tasks/` 아래에 둔다.
3. 결과를 만들기보다 확인, 추적, 진단이 목적이면 `analysis/` 아래에 둔다.
4. task 폴더는 "기술명"보다 "업무 단위" 기준으로 나눈다.

이 기준으로도 애매하면, 일단 루트에 두지 말고 적절한 task 폴더를 먼저 만드는 쪽을 기본값으로 한다.
