# PPD Stage 2 Experiment Chat Summary

이 문서는 이 채팅에서 진행한 PPD Stage 2 Stable Cascade 실험 흐름, 주요 판단, 코드/명령어 설계, 남은 작업을 한 파일로 정리한 것이다.

## 1. 초기 Stable Cascade Inference 점검

- `stage_2/infer_stage2.py`에서 user embedding conditioning을 제외하고 기존 Stable Cascade가 정상적으로 생성되는지 먼저 확인했다.
- `CONDITIONS = ("base", "branch_off", "zero_user", "zero_user_zero_mask", "real_user")`를 argument로 고를 수 있게 사용하는 흐름을 확인했다.
- `--condition base`로 vanilla 또는 patched-base 상태를 테스트했다.
- 512x512 생성에서는 astronaut가 잘 보이지 않거나 이미지가 zoom-in/crop된 것처럼 보였고, 1024x1024에서는 상대적으로 안정적이었다.
- 결론적으로 Stable Cascade Stage C/decoder path는 1024 기준, 즉 Stage C latent `[16,24,24]` 쪽이 더 canonical하다고 판단했다.

대표 base 생성 명령:

```bash
conda run -n ppd_stage2 python stage_2/infer_stage2.py image \
  --condition base \
  --no-default-extra-prompts \
  --prompt "An astronaut" \
  --seed 0 \
  --seed 42 \
  --height 1024 \
  --width 1024 \
  --run-name stable_cascade_base_astronaut_1024
```

## 2. 기존 12x12 Latent 정리

기존 512/768 이미지 기반 latent는 `[1,16,12,12]`로 만들어져 있었고, 새 24x24 latent와 구분하기 위해 정리했다.

- 기존 실파일 위치:
  - `/Data_Storage/roycecho/PPD/stage2_latents_v512_train`
- 기존 repo 링크:
  - `/data/roycecho/PPD/stage2_latents`
- 목표 위치:
  - `/Data_Storage/roycecho/PPD/latents/latents_12x12`
  - `/data/roycecho/PPD/latents/latents_12x12`
- manifest는 다시 생성하면 되므로 latent 파일 중심으로 이동했다.

12x12 manifest 생성 예:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n ppd_stage2 python stage_2/tasks/latent/build_latent_manifest.py \
  --latent-root /Data_Storage/roycecho/PPD/latents/latents_12x12 \
  --output-jsonl /Data_Storage/roycecho/PPD/latents/latents_12x12/latent_manifest.jsonl \
  --summary-json /Data_Storage/roycecho/PPD/latents/latents_12x12/latent_manifest_summary.json \
  --progress-every 5000
```

## 3. Diffusers 기반 24x24 Latent 시도와 실패

처음에는 기존 `image_to_latents.py`를 사용해 `--target-image-size 1024x1024`로 latent shape를 `[1,16,24,24]`에 맞췄다.

중요 sidecar 필드로 남기기로 한 값:

```text
target_image_size: [1024, 1024]
effnet_input_size: [768, 768]
preprocess_resolution_mode: "auto"
effnet_preprocess_resolution: [768, 768]
encoder_class
scaling_applied
scaling_expression
expected_latent_shape: [1,16,24,24]
actual_latent_shape: [1,16,24,24]
```

하지만 decoder reconstruction sanity check에서 semantic 정보가 거의 유지되지 않았다. 색감과 큰 구조는 일부 남았지만 객체/장면 의미가 무너졌다.

scaling ablation도 수행했다.

```text
A. 현재 scaled latent
B. unscaled latent
C. scaled latent x 42
D. scaled latent x 10
E. scaled latent x 2
F. scaled latent / 2
```

결과는 A_scaled가 가장 나았고, 단순 scaling mismatch만으로 설명되지는 않았다.

## 4. Stability-AI 원본 StableCascade `train_c.py` 확인

원본 코드 기준 Stage C latent 생성 경로를 확인했다.

파일:

```text
/data/roycecho/StableCascade/train/train_c.py
```

핵심 코드:

```python
images = batch["images"].to(self.device)
return models.effnet(extras.effnet_preprocess(images))
```

원본 흐름:

```text
raw jpg/png
-> PIL RGB
-> ToTensor(): [3,H,W], float [0,1]
-> Resize(image_size)
-> SmartCrop 또는 deterministic crop
-> ImageNet Normalize
-> EfficientNetV2-S features
-> 1x1 Conv: 1280 -> 16
-> BatchNorm2d affine=False
-> Stage C latent
```

중요한 결론:

- `train_c.py`의 Stage C latent는 Stable Diffusion VAE latent가 아니다.
- raw image를 EfficientNetEncoder로 압축한 16-channel semantic feature map이다.
- `image_size=768`이면 latent shape는 `[B,16,24,24]`가 된다.
- Diffusers Wuerstchen 예시의 `add(1).div(42)` scaling은 원본 `train_c.py` 학습 경로와 맞지 않는 것으로 판단했다.

## 5. Canonical 24x24 Stability Raw Latent 생성

새 canonical latent root:

```text
/Data_Storage/roycecho/PPD/latents/latents_24x24_stability_raw
/data/roycecho/PPD/latents/latents_24x24_stability_raw
```

메인 생성 스크립트:

```text
stage_2/tasks/latent/stability_raw_image_to_latents.py
```

설정:

```text
image_size: 768
expected_latent_shape: [1,16,24,24]
scaling_applied: false
scaling_expression: null
latent_semantics: stability_train_c_raw_effnet
crop_policy: deterministic_center_crop
normalization: ImageNet mean/std
encoder_source: Stability-AI/StableCascade/modules/effnet.py
```

train 생성 명령 예:

```bash
CUDA_VISIBLE_DEVICES=0 python stage_2/tasks/latent/stability_raw_image_to_latents.py \
  --image-dir /Data_Storage/roycecho/PPD/repo_data/pickapic_uid_images/train \
  --output-dir /Data_Storage/roycecho/PPD/latents/latents_24x24_stability_raw/train \
  --image-size 768 \
  --effnet-checkpoint /Data_Storage/roycecho/PPD/checkpoints/stable_cascade/effnet_encoder.safetensors \
  --device cuda \
  --batch-size 128 \
  --shape-policy strict \
  --skip-existing \
  --summary-json /Data_Storage/roycecho/PPD/latents/latents_24x24_stability_raw/train_summary.json \
  --progress-every-batches 100
```

sanity check 스크립트:

```text
stage_2/tasks/latent/decode_stability_raw_latent_sanity.py
```

이 스크립트는 원본 이미지, StableCascade previewer reconstruction, diffusers decoder reconstruction을 비교하는 용도다.

## 6. Manifest와 Dataset Join 확인

새 latent manifest 기준 확인 항목:

```text
missing latents == 0
latent shape == (16,24,24)
scaled == false
latent_semantics == stability_train_c_raw_effnet
dataset length == assignment query pair count
```

collate 후 기대 shape:

```text
user_emb: [B,L,3584]
preferred_latent: [B,16,24,24]
dispreferred_latent: [B,16,24,24]
caption: list[str]
```

forward-only smoke에서 확인할 것:

```text
prior input shape mismatch 없음
patched prior forward 정상
NaN/Inf 없음
base / zero / real 비교 경로 정상
```

train-step smoke에서 확인할 것:

```text
loss finite
backward 정상
gradient가 user adapter에 들어감
frozen backbone/reference 유지
NaN/Inf 없음
```

## 7. Stage 2 Training 설계

메인 학습 스크립트:

```text
stage_2/train_stage2_full.py
```

구조:

- Stable Cascade prior를 train prior와 frozen reference prior 두 개로 둔다.
- train prior에는 user-conditioning branch를 patch한다.
- reference prior는 base/frozen으로 유지한다.
- preferred/dispreferred latent에 같은 noise/timestep을 적용한다.
- train prior와 reference prior의 MSE 차이로 DPO-style objective를 계산한다.

loss 개념:

```text
train_pref_err
train_dispref_err
ref_pref_err
ref_dispref_err
score = beta * ((train_dispref_err - train_pref_err) - (ref_dispref_err - ref_pref_err))
loss = -logsigmoid(score)
```

업데이트 범위는 user branch만으로 제한한다.

대표 trainable markers:

```text
.user_projection.
.user_adapter.k_proj.
.user_adapter.v_proj.
.user_adapter.out_proj.   # 옵션에 따라 freeze 가능
.user_scale               # 옵션에 따라 freeze 가능
```

## 8. Patch Block과 Scope 실험

초기 default patch block:

```text
down_blocks.0.2
down_blocks.0.5
```

4-block 후보:

```text
down_blocks.0.2
down_blocks.0.5
down_blocks.0.8
down_blocks.0.11
```

전체 attention block:

```text
--patch-all-attention-blocks
```

중요 실험 축:

```text
2-block baseline
4-block patch
6~8 block patch
all attention block patch
```

실험 판단 기준:

```text
zero_user가 base와 얼마나 가까운가
real_user가 base에서 의미 있게 달라지는가
latent norm이 언제 튀는가
decoded image가 언제 무너지는가
```

## 9. 주요 Training 옵션 의미

요청했던 설정:

```text
patch_path:
- down_blocks.0.2
- down_blocks.0.5

trainable:
- .user_projection.
- .user_adapter.k_proj.
- .user_adapter.v_proj.

frozen:
- .user_adapter.out_proj.

user_adapter_zero_init_out: false
user_projection_bias: true
user_projection_norm_affine: true
user_adapter_projection_bias: true
trainable_user_scale: false
user_scale: 1.0
```

명령어 핵심 옵션:

```text
--user-scale 1.0
--no-trainable-user-scale
--user-projection-bias
--user-projection-norm-affine
--user-adapter-projection-bias
--no-user-adapter-zero-init-out
--no-train-user-adapter-out-proj
```

## 10. Effective Batch와 Step 수

데이터:

```text
train query pairs ~= 196,644
batch_size=2 -> micro_steps ~= 98,322
```

1000 optimization steps 근처를 맞추기 위한 설정:

```text
batch_size = 2
gradient_accumulation_steps = 98
effective batch ~= 196
optimizer steps ~= 1004
```

768 effective batch를 그대로 유지하면 1 epoch에서 optimizer step 수가 너무 작아진다. 따라서 먼저 1000 optimization step에 가까운 설정을 우선 비교하기로 했다.

## 11. 대표 Training Commands

### 2-block, out_proj freeze, user_scale fixed

```bash
CUDA_VISIBLE_DEVICES=0 python stage_2/train_stage2_full.py \
  --train-embedding-json-paths "data/user_emb_7b_full/train_shard*.json" \
  --train-assignment-jsonl-paths "artifacts/pair_assignments/train/stage2_pair_assignments_train_shard*.jsonl" \
  --train-latent-manifest-jsonl-path /Data_Storage/roycecho/PPD/latents/latents_24x24_stability_raw/latent_manifest_train.jsonl \
  --train-uid-to-path-json-path data/train_uid_to_path.json \
  --val-embedding-json-paths "data/user_emb_7b_full/validation_shard*.json" \
  --val-assignment-jsonl-paths "artifacts/pair_assignments/validation/stage2_pair_assignments_validation_shard*.jsonl" \
  --val-latent-manifest-jsonl-path /Data_Storage/roycecho/PPD/latents/latents_24x24_stability_raw/latent_manifest_validation.jsonl \
  --val-uid-to-path-json-path data/validation_uid_to_path.json \
  --device cuda \
  --reference-device cuda \
  --torch-dtype bfloat16 \
  --local-files-only \
  --patch-path down_blocks.0.2 \
  --patch-path down_blocks.0.5 \
  --batch-size 2 \
  --gradient-accumulation-steps 98 \
  --learning-rate 1e-5 \
  --num-epochs 1 \
  --user-scale 1.0 \
  --no-trainable-user-scale \
  --user-projection-bias \
  --user-projection-norm-affine \
  --user-adapter-projection-bias \
  --no-user-adapter-zero-init-out \
  --no-train-user-adapter-out-proj \
  --user-dropout-prob 0.1 \
  --log-every 10 \
  --val-every-steps 100 \
  --max-val-batches 20 \
  --latest-checkpoint-every-steps 0 \
  --checkpoint-every-steps 100 \
  --keep-last-checkpoints 9999 \
  --output-dir artifacts/stage2_train_full \
  --wandb-mode online \
  --wandb-run-name sharedQ_2block_scopeNoOutProj_us1fixed_biasOn_normAffineOn_noZeroOut_eff196
```

### 4-block version

2-block 명령어에서 patch path만 아래처럼 확장:

```text
--patch-path down_blocks.0.2
--patch-path down_blocks.0.5
--patch-path down_blocks.0.8
--patch-path down_blocks.0.11
```

### all-block, out_proj freeze, user_scale fixed

```text
--patch-all-attention-blocks
--no-train-user-adapter-out-proj
--no-trainable-user-scale
```

## 12. Inference / Evaluation Grid

평가 스크립트:

```text
stage_2/tasks/eval/generate_stage2_user_grid.py
```

조건:

```text
base
zero_user
zero_user_zero_mask
real_user
```

inference scale:

```bash
--inference-user-scale 0.03
```

scale sweep:

```bash
--inference-user-scale-sweep 0.0 0.01 0.03 0.1 0.3 1.0
```

보고 싶은 metrics:

```text
prior output L2 norm
base 대비 prior cosine
CLIP image cosine
pixel pairwise metric
decoded image grid
```

checkpoint sweep 목표:

```text
row: checkpoint step 100, 200, 400, 800
column: base, zero_user, real_user_0.01, real_user_0.03, real_user_0.1, real_user_0.3, real_user_1.0
```

목표:

```text
언제부터 latent norm이 튀기 시작하는지
언제부터 decoded image가 무너지는지
zero_user drift와 real_user effect가 어떻게 변하는지
```

## 13. base / zero_user / real_user 해석

`base`:

- user branch를 사용하지 않는 baseline path.
- user-conditioned branch 영향 없이 현재 prior가 prompt/seed에서 만드는 결과.

`zero_user`:

- user branch는 켜지만 user embedding을 0으로 넣는 조건.
- zero-user no-op 여부를 확인하는 핵심 조건.

`real_user`:

- 실제 user embedding을 넣은 personalized condition.

해석:

```text
base 정상, zero_user 정상, real_user만 붕괴
-> user embedding 방향 또는 user branch magnitude 문제

base 정상, zero_user부터 붕괴
-> branch가 켜지는 것 자체가 generic drift를 만듦

base부터 붕괴
-> checkpoint/inference path/prompt/decoder 문제 가능성
```

## 14. zero-user no-op 관련 실험

zero-user drift 후보:

```text
projection bias
LayerNorm affine
adapter projection bias
out_proj nonzero init
out_proj trainable scope
```

zero-user no-op 복구 실험:

```text
user_projection_bias = false
user_projection_norm_affine = false
user_adapter_projection_bias = false
user_adapter_zero_init_out = true
```

scope 축소 실험:

```text
train:
- user_projection
- k_proj
- v_proj

freeze:
- out_proj
```

이유:

- out_proj는 branch output을 hidden residual로 바꾸는 마지막 projection이라 generic drift를 키울 수 있다.
- k_proj/v_proj는 user-conditioned signal에 더 직접적이다.

## 15. 2-GPU Reference Split Plan

OOM 상황:

```text
train prior + reference prior가 GPU 0에 같이 올라감
GPU 0 memory almost full
20MB allocation도 실패
```

현재 문제:

- `--reference-device cuda`는 별도 GPU가 아니라 train device와 같은 GPU를 의미한다.
- 따라서 GPU 2장이 있어도 reference prior가 GPU 1로 가지 않는다.

목표:

```text
train_prior -> cuda:0
reference_prior -> cuda:1
train forward/backward tensors -> cuda:0
reference forward tensors -> cuda:1
reference output/error만 detach 후 cuda:0으로 이동
```

수정 필요:

```text
1. train_smoke_stage2.py::_resolve_reference_device()가 cuda:N 지원
2. train_stage2_full.py parser에서 --reference-device choices 제거
3. train_smoke_stage2.py parser에서 --reference-device choices 제거
4. train/reference device별 memory logging 추가
5. py_compile 문법 체크
```

수정 후 smoke command:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python stage_2/train_stage2_full.py \
  --train-embedding-json-paths "data/user_emb_7b_full/train_shard*.json" \
  --train-assignment-jsonl-paths "artifacts/pair_assignments/train/stage2_pair_assignments_train_shard*.jsonl" \
  --train-latent-manifest-jsonl-path /Data_Storage/roycecho/PPD/latents/latents_24x24_stability_raw/latent_manifest_train.jsonl \
  --train-uid-to-path-json-path data/train_uid_to_path.json \
  --val-embedding-json-paths "data/user_emb_7b_full/validation_shard*.json" \
  --val-assignment-jsonl-paths "artifacts/pair_assignments/validation/stage2_pair_assignments_validation_shard*.jsonl" \
  --val-latent-manifest-jsonl-path /Data_Storage/roycecho/PPD/latents/latents_24x24_stability_raw/latent_manifest_validation.jsonl \
  --val-uid-to-path-json-path data/validation_uid_to_path.json \
  --device cuda:0 \
  --reference-device cuda:1 \
  --torch-dtype bfloat16 \
  --local-files-only \
  --patch-all-attention-blocks \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --num-epochs 1 \
  --max-train-steps 2 \
  --user-scale 1.0 \
  --no-trainable-user-scale \
  --user-projection-bias \
  --user-projection-norm-affine \
  --user-adapter-projection-bias \
  --no-user-adapter-zero-init-out \
  --no-train-user-adapter-out-proj \
  --user-dropout-prob 0.1 \
  --log-every 1 \
  --val-every-steps 0 \
  --latest-checkpoint-every-steps 0 \
  --checkpoint-every-steps 0 \
  --output-dir artifacts/stage2_train_full \
  --wandb-mode disabled
```

주의:

- `max-train-steps=2`, `gradient_accumulation_steps=4`는 optimizer step까지 가지 않는다.
- 2 micro step forward/backward가 cuda:0/cuda:1 split에서 OOM 없이 되는지만 확인한다.
- optimizer step까지 확인하려면 `--max-train-steps 4`가 필요하다.

## 16. 현재 남은 우선순위

```text
1. 2-GPU reference split 코드 수정
2. py_compile 문법 체크
3. max-train-steps 2 smoke run
4. nvidia-smi로 GPU 0/1 memory 분산 확인
5. all-block full run 재시작
6. checkpoint step별 scale sweep으로 norm/cosine/grid 비교
```

## 17. Codex / VS Code Auto-Review 및 Reconnecting 장애 기록

2026-05-08부터 Codex auto-review가 `auto-review timed out`으로 끝나고, 일반 채팅도 `thinking` 중 `reconnecting`이 여러 번 뜬 뒤 응답이 오는 증상이 있었다.

확인한 사실:

- repo 내부 hook/script 문제는 아니었다.
- auto-review 세션은 원래 `codex-auto-review` guardian subagent로 `read-only` sandbox, `approval_policy=never`로 뜨는 것이 정상이다.
- `~/.codex/models_cache.json`이 이전 client version `0.128.0` 기준으로 남아 있었고, VS Code 확장/CLI는 `0.129.0-alpha.15` 계열로 올라가면서 모델 캐시 miss가 발생했다.
- 이때 로그에 `failed to refresh available models: timeout waiting for child process to exit`가 반복됐다.

모델 캐시 쪽 조치:

- 기존 `~/.codex/models_cache.json`은 `~/.codex/models_cache.json.bak`로 보존했다.
- `.bak`의 구조를 유지한 채 `client_version`을 `0.129.0`으로 맞춘 새 `models_cache.json`을 만들었다.
- VS Code reload 후 로그에서 `models cache: cache hit`, `models cache: cache entry applied models_count=6`, `using cached models for OnlineIfUncached`를 확인했다.
- 따라서 auto-review 모델 캐시 문제는 해결된 것으로 판단했다.

남은 reconnecting 원인:

- 단순 질의(`today's date`)와 low intelligence에서도 매번 `reconnecting`이 반복되어 reasoning latency 문제가 아니라고 판단했다.
- 최근 `remoteexthost.log`에서 OpenAI 확장 쪽 `Error: not-connected`가 연속으로 찍혔다.
- stack은 `openai.chatgpt-26.506.21252-linux-x64/out/extension.js`의 `sendBroadcast`에서 발생했다.
- 더 직접적인 원인은 OpenAI 확장이 내부 IPC socket을 열지 못하는 것이었다.

핵심 에러:

```text
Error: listen EACCES: permission denied /tmp/codex-ipc/ipc-1007.sock
Error: not-connected
```

권한 상태:

```text
current user: uid=1007(roycecho)
/tmp/codex-ipc: owner=jaewoong group=jaewoong mode=775
/tmp/codex-ipc/ipc-1001.sock: owner=jaewoong group=jaewoong
```

해석:

- OpenAI VS Code 확장은 Linux에서 `os.tmpdir()/codex-ipc/ipc-<uid>.sock` 형태의 socket을 만든다.
- 현재 `os.tmpdir()`이 `/tmp`이고, `/tmp/codex-ipc`가 다른 사용자 소유 `775`라서 `roycecho`가 `ipc-1007.sock`을 만들 수 없다.
- 그래서 확장의 IPC router가 실패하고, webview broadcast channel이 `not-connected`가 되며, UI에서는 `reconnecting`으로 보인다.
- `~/.codex` cache/memory 삭제만으로는 `/tmp/codex-ipc` 권한 문제가 해결되지 않는다.

관리자 권한이 있을 때의 직접 해결:

```bash
sudo chmod 1777 /tmp/codex-ipc
```

또는:

```bash
sudo rm -rf /tmp/codex-ipc
```

관리자 권한 없이 시도할 수 있는 우회책:

```text
1. user-writable TMPDIR를 만들기
2. VS Code Remote Server가 시작될 때 TMPDIR=/data/roycecho/.tmp 를 갖도록 설정
3. VS Code Remote Server를 완전히 재시작
4. OpenAI 확장이 /data/roycecho/.tmp/codex-ipc/ipc-1007.sock을 쓰는지 확인
```

주의:

- `~/.codex/auth.json`, `~/.codex/config.toml`, `~/.codex/plugins`, `~/.codex/skills`, `~/.codex/sessions`는 삭제하면 로그인/설정/플러그인/세션 정보가 날아갈 수 있다.
- 전체 삭제보다 먼저 `/tmp/codex-ipc` 권한 문제를 우회하는 것이 맞다.
- 정리 삭제가 필요하면 `~/.codex/cache`, `~/.codex/models_cache.json`, `~/.codex/logs_*.sqlite*`, `~/.codex/state_*.sqlite*`, `~/.codex/.tmp`, `~/.codex/tmp`, `~/.codex/shell_snapshots` 정도만 백업 후 제한적으로 삭제하는 편이 안전하다.
