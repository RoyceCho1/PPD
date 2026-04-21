# Stage 2

`stage_2/`는 "유저 임베딩을 Stage C prior에 주입해서 preference pair를 다루는 실험 코드"와, 그 실험에 필요한 manifest/assignment/latent 준비 스크립트를 모아둔 폴더다.

핵심적으로는 아래 흐름으로 이해하면 된다.

1. Hugging Face preference 데이터셋에서 이미지 UID 기준 lookup manifest를 만든다.
2. pair assignment 파일에서 실제로 필요한 UID만 추린다.
3. 해당 UID의 raw image를 Stage C clean latent 후보로 변환한다.
4. 저장된 latent들을 다시 manifest로 인덱싱한다.
5. Stage 2 dataset으로 user embedding + preference pair를 읽는다.
6. Stage C attention block에 user-conditioning branch를 patch해서 forward smoke test를 돌린다.

## 핵심 파일

### [stage2_dataset.py](/data/roycecho/PPD/stage_2/stage2_dataset.py)

Stage 2의 메인 데이터셋 정의다.

- user 단위 JSON을 읽어서 pair 단위 sample로 flatten한다.
- `preferred_image_uid_{i}`, `dispreferred_image_uid_{i}`, `caption_{i}` 형태의 필드를 파싱한다.
- user embedding(`emb`)을 tensor로 바꾸고 shape 검증을 한다.
- 선택적으로 `uid_to_path.json`, `uid_to_meta.json`을 붙여서 이미지 경로나 UID 메타데이터를 함께 제공한다.
- `collate_fn`까지 포함한 학습/실험 입력 포맷의 기준점 역할을 한다.

즉, Stage 2에서 "무엇을 한 샘플로 볼 것인가"를 정의하는 파일이다.

### [user_adapter.py](/data/roycecho/PPD/stage_2/user_adapter.py)

유저 임베딩을 Stage C attention에 넣기 위한 모듈이다.

- `UserProjection`: `[B, L, 3584]` 형태의 user embedding을 cross-attention token 공간으로 projection한다.
- `UserCrossAttentionAdapter`: diffusion hidden state를 query로, user token을 key/value로 써서 별도 cross-attention residual을 만든다.

즉, "유저 임베딩을 어떤 방식으로 조건(condition)으로 넣을지"를 담당한다.

### [patch_stage_c.py](/data/roycecho/PPD/stage_2/patch_stage_c.py)

Stable Cascade Stage C의 attention block에 user-conditioning branch를 삽입하는 패치 유틸이다.

- 기존 `SDCascadeAttnBlock`를 감싸는 `PatchedSDCascadeAttnBlock`을 제공한다.
- 원래 text-conditioning 경로는 유지한 채, user branch residual을 병렬로 더한다.
- 특정 block path만 골라 patch할 수 있다.
- backbone freeze와 trainable parameter summary 유틸도 같이 제공한다.

즉, "기존 Stage C를 어디까지 유지하고 어떤 지점에 user branch를 덧붙일지"를 담당한다.

### [forward_only_stage2.py](/data/roycecho/PPD/stage_2/forward_only_stage2.py)

Stage 2 통합 smoke test 스크립트다.

- `Stage2PreferenceDataset`에서 batch를 하나 뽑는다.
- Stable Cascade prior pipeline과 text encoder를 로드한다.
- Stage C에 user adapter patch를 적용한다.
- text conditioning + user conditioning을 함께 넣어 forward만 실행한다.

이 스크립트는 학습 코드가 아니라, 데이터셋과 patch가 실제로 연결되는지 확인하는 데 목적이 있다.

## Latent / Manifest 준비

### [tasks/hf_manifest/build_uid_manifest_from_hf.py](/data/roycecho/PPD/stage_2/tasks/hf_manifest/build_uid_manifest_from_hf.py)

Hugging Face preference 데이터셋을 스캔해서 UID 기준 lookup 파일을 만든다.

- `uid_to_path.json`: `image_uid -> image path`
- `uid_to_meta.json`: `image_uid -> 집계 메타데이터`

수집하는 메타데이터에는 대략 아래가 포함된다.

- 등장 횟수
- best / non-best 등장 횟수
- split별 카운트
- caption 샘플
- partner UID 샘플
- source path 샘플

필요하면 이미지 바이너리를 로컬 디렉토리에 UID 이름으로 저장하는 역할도 한다.

즉, HF 데이터셋을 Stage 2에서 바로 쓰기보다, UID 중심 manifest로 한 번 정리하는 전처리 단계다.

### [extract_needed_uids_from_assignments.py](/data/roycecho/PPD/stage_2/extract_needed_uids_from_assignments.py)

`stage2_pair_assignments*.jsonl`에서 실제로 참조되는 이미지 UID만 추출한다.

- 기본적으로 `query_pairs`의 UID를 모은다.
- 옵션으로 `support_pairs`까지 포함할 수 있다.
- 결과를 txt/json으로 저장한다.

즉, "manifest에 있는 전체 UID"가 아니라 "현재 assignment가 실제로 쓰는 UID 부분집합"을 뽑는 스크립트다.

### [image_to_latents.py](/data/roycecho/PPD/stage_2/image_to_latents.py)

raw image를 Stage C clean latent 후보로 바꾸는 스크립트다.

- local Wuerstchen example의 `EfficientNetEncoder`를 불러온다.
- image transform을 적용한 뒤 encoder forward를 수행한다.
- 필요 시 scaling rule까지 적용한다.
- 각 UID/image마다 latent tensor와 metadata를 저장한다.
- UID 목록 + `uid_to_path.json`을 이용해 필요한 이미지 subset만 처리할 수 있다.

이 스크립트는 학습을 하지 않고, latent artifact를 준비하는 데만 집중한다.

### [build_latent_manifest.py](/data/roycecho/PPD/stage_2/build_latent_manifest.py)

이미 저장된 latent `.pt` 파일을 다시 스캔해서 `latent_manifest.jsonl`을 만든다.

- latent를 새로 생성하지는 않는다.
- 각 `.pt` 파일에서 UID, tensor shape, dtype, 통계값을 읽는다.
- 필요하면 `uid_to_path.json`, `uid_to_meta.json`을 붙여 manifest를 enrich한다.

즉, latent 폴더를 모델 입력용 lookup manifest로 인덱싱하는 역할이다.

## Pair Assignment 작업

### [tasks/pair_assignment/build_pair_pool_from_hf.py](/data/roycecho/PPD/stage_2/tasks/pair_assignment/build_pair_pool_from_hf.py)

HF preference 데이터셋에서 Stage 2 pair assignment의 재료가 되는 pair pool을 만드는 스크립트다.

### [tasks/pair_assignment/build_stage2_pair_assignments.py](/data/roycecho/PPD/stage_2/tasks/pair_assignment/build_stage2_pair_assignments.py)

pair pool을 바탕으로 Stage 2에서 사용할 assignment JSONL을 생성하는 스크립트다.

### [tasks/pair_assignment/run_build_stage2_pair_assignments_all.py](/data/roycecho/PPD/stage_2/tasks/pair_assignment/run_build_stage2_pair_assignments_all.py)

assignment 생성 작업을 shard 단위로 여러 번 실행하는 배치 runner다.

## Analysis 스크립트

`analysis/` 아래 파일들은 결과를 생산하는 메인 파이프라인보다는 확인과 추적에 가깝다.

### `analysis/stage_c/`

- `inspect_stage_c.py`: Stage C 구조를 확인한다.
- `inspect_stage_c_target_path.py`: patch 후보 path를 확인한다.
- `trace_stage_c_clean_target.py`: clean target 관련 경로를 추적한다.

### `analysis/user/`

- `inspect_user_preferences.py`: user preference 샘플과 데이터 형태를 점검한다.

## 권장 이해 순서

처음 코드를 읽을 때는 아래 순서가 가장 자연스럽다.

1. `stage2_dataset.py`
2. `user_adapter.py`
3. `patch_stage_c.py`
4. `forward_only_stage2.py`
5. `tasks/hf_manifest/build_uid_manifest_from_hf.py`
6. `extract_needed_uids_from_assignments.py`
7. `image_to_latents.py`
8. `build_latent_manifest.py`

앞의 1~4는 "모델 쪽 핵심 흐름", 뒤의 5~8은 "입력 데이터와 latent artifact 준비 흐름"이다.

## 한 줄로 요약하면

`stage_2/`는 "user preference 기반 Stage 2 실험을 위한 모델 패치 코드"와 "그 실험에 필요한 UID manifest / pair assignment / latent 준비 스크립트"를 함께 모아둔 폴더다.
