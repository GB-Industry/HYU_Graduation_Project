# Bouldering Adaptive Control — Prototype v1

`Prototype v0`의 2D route proof-of-concept 다음 단계로, 승인된 프로젝트 설계의 **M0 Physics Foundation** 일부를 코드 형태로 만든 프로토타입입니다.

## Validated Foundation

The current physical foundation validates model integrity, unilateral frictional
foot support, bounded hand point grasps, and deterministic static torque control.
Controller gains are physical `Nm/rad` and `Nm*s/rad`; motor strength changes torque
ceilings, not impedance. Physical readiness requires a sustained contact/motion
window, not the historical instantaneous thresholds below.

- Stage 1 model evidence: `docs/verification/validated-foundation-stage1.md`.
- Stage 2 contact evidence: `docs/verification/validated-foundation-stage2.md`.
- Stage 3 static-control evidence: `docs/verification/validated-foundation-stage3.md`.
- Stage 4 single-hand transition evidence: `docs/verification/validated-foundation-stage4.md`.
- Stage 5 transfer-family evidence: `docs/verification/validated-foundation-stage5.md`.

Stage 4 adds one measured release/reach/strict-capture/new-readiness transition
at both 2 ms and 1 ms. It preserves all Stage 1-3 physical/controller bounds and
does not certify the old synthetic multi-move route or generalized route planning.
Same-run transition evidence is available with
`MUJOCO_GL=egl python scripts/validate_transition.py --suite all --render --summary`.

Stage 5 extends that explicit contract to both hands, a native-contact foot
transfer, and a no-reset two-hand sequence. Endpoint-settled hand acquisition
reports margin inside the unchanged 1 mm gate; foot touchdown is not counted as
support until native compression/friction and persistence are verified.
Run `MUJOCO_GL=egl python scripts/validate_transfers.py --suite all --render --summary`.
These are bounded movement contracts, not generalized route planning or RL.

Stage 5 visual inspection is separate from numerical acceptance. The legacy
640x480 telemetry videos are not accepted as usable full-body visual evidence.
`MUJOCO_GL=egl python scripts/render_clean_transfers.py` renders the saved native
trajectories at 1920x1080 from three cameras with an off-body essentials panel.
See `docs/verification/validated-foundation-stage5-visual.md`: layout passes, but
the hand/sequence whole-body motion demonstration remains inadequate.

Stage 5.1 adds explicitly declared whole-body fixtures without changing physical
gains, grip/friction limits, strict capture or readiness gates. The native matrix
accepts eight positive and eight negative cases at 2 ms and 1 ms. Independent
inspection of all twelve native 1 ms decoded views, with representative 2 ms
comparisons, shows pre-release knee/hip extension and torso reorientation, a real
foot step, and cumulative no-reset sequence posture. The historical Stage 5
visual failure above is preserved; its deficiency is resolved only for these new
fixtures, not for generalized naturalism or pressure inferred from geometry.
Full regression passes (445 passed, one skipped), as do all Stage 1-5 preservation
checks and complete decoding of all 24 new MP4s. Stage 6 has not started.

See [Stage 5.1 evidence and limits](docs/verification/validated-foundation-stage5.1.md)
and [the native 16-case report](outputs/whole-body-stage5.1/report.json).
From the authorized worktree, after activating the canonical environment below:

```bash
MUJOCO_GL=egl python scripts/validate_whole_body.py --suite all --summary --output outputs/whole-body-stage5.1
MUJOCO_GL=egl python scripts/render_clean_transfers.py --fixture whole_body --input outputs/whole-body-stage5.1 --dt .002 --output outputs/whole-body-clean-2ms
MUJOCO_GL=egl python scripts/render_clean_transfers.py --fixture whole_body --input outputs/whole-body-stage5.1 --dt .001 --output outputs/whole-body-clean-1ms
```

Stage 5.2 completes a bounded fixed-scene morphology/target study: 44 accepted
records at 2 ms and 1 ms, with 30 physical successes, ten locally ROM-limited
searches, and four scoped geometric/path counterexamples. All 22 timestep pairs
agree; local search failure is not global physical impossibility. Full regression
passes with 488 passed and one skipped. Stage 6 has not started.
See [Stage 5.2 evidence, measured responses, and limits](docs/verification/validated-foundation-stage5.2.md)
and [the native 44-case report](outputs/morphology-envelope-stage5.2/report.json).

Stage 5.3 adds one native RH-up / LF-up / LH-up climbing chain, with exact
state/reference continuation and about 89 mm pelvis / 84 mm COM vertical gain.
Baseline and longer-limb profiles pass at 2 ms and 1 ms; the shorter profile
remains honestly unresolved under three fixed reference candidates. Clean
scene-only videos are verified at 1.0x native time. Full regression passes
(529 passed, one skipped), and Stage 1-5.2 preservation passes. RL has not started.
See [Stage 5.3 evidence, timing, videos and limits](docs/verification/validated-foundation-stage5.3.md).

### Interactive Stage 5.3 Playback

From this authorized worktree, on a graphical desktop:

```bash
/home/yuchan/Desktop/project/boulder_prototype/.venv/bin/python scripts/view_ascent.py
```

This passive MuJoCo viewer opens **paused on the recorded climbing source pose**,
not the legacy synthetic stance. It loads the certified native RH-up / LF-up /
LH-up sequence from `outputs/ascent-stage5.3-final`, checks evidence/code/model
hashes and exact move boundaries, and never steps physics or regenerates motion.
Mouse rotation, zoom and pan use MuJoCo's normal camera controls.

- `Space`: play/pause; `R` or `Home`: restart at the first climbing pose, paused.
- `1`, `2`, `3`: 0.5x, 1x, 2x playback; arrows or `,` / `.`: step backward/forward one native timestamp.
- `C`: toggle recorded contact markers. Blue connects follow saved hand equality activation; foot points and displayed loads are saved native observations.
- `--profile longer`: the +5% limb geometry; `--dt .001`: the authoritative 1 ms recording.
- `--autoplay`: start playing; `--check`: validate the saved recording without opening a window.

Playback copies literal saved pose/velocity/control/equality state into an
isolated display model/data pair and refreshes kinematics only. It does not
interpolate poses, simulate mouse perturbations, or recompute native forces.
At capture, the pre-activation sample remains exactly as recorded; activation
appears on the first saved post-capture sample (within one native interval).
Full integration vectors are checked at move boundaries; per-step warmstarts
are not recorded, so this display stream is not a dynamics-restart checkpoint.
Missing or mismatched evidence is an error, with no standing-pose fallback or
automatic recapture. The source recordings are offline/ignored artifacts and
must be present. A desktop display is required for the GUI; headless use is
limited to `--check`.

Run from the authorized worktree with the canonical project environment:

```bash
source /home/yuchan/Desktop/project/boulder_prototype/.venv/bin/activate
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
MUJOCO_GL=egl python scripts/validate_controller.py --render --summary
```

This validates static deterministic control, not generalized climbing movement or
complete climbing feasibility. The old physical route initializer is still rejected
by strict capture gates. Historical v1.6 claims below are archival, not current
physical certification; `idealized_debug` and archived test-controller fixtures are
explicitly NONPHYSICAL.

## 이번 버전 (v1.6: Generalized Multi-Limb Transition & Sequential Execution)이 검증하는 것

- `BoulderScene`을 simulation/planning 공통 scene 형식으로 사용
- **일반화된 다중 사지 전이 API (`TransitionRequest`, `TransitionResult`, `execute_transition`)**:
  - 특정 사지나 홀드에 종속되지 않는 범용 사지 전이 프리미티브 구현
  - 양손(`RIGHT_HAND`, `LEFT_HAND`) 및 양발(`LEFT_FOOT`, `RIGHT_FOOT`) 완전 지원
  - 명시적이고 세분화된 전이 상태 코드(`TransitionStatus`): `SUCCESS`, `INVALID_REQUEST`, `SOURCE_NOT_ATTACHED`, `INELIGIBLE_TARGET`, `SUPPORT_FAILURE`, `GRIP_FAILURE`, `REACH_FAILURE`, `ATTACH_FAILURE`, `UNSTABLE_FINAL_STATE`
- **양방향 대칭 손 전이 (Bilateral Hand Transitions)**:
  - 동일한 일반화 전이 엔진을 통해 오른손(RH: H4 -> H5, $\Delta p = 0.166\,\text{m}$) 및 왼손(LH: H3 -> H7, $\Delta p = 0.256\,\text{m}$) 실제 개별 홀드 전이 완수
  - 사지 엔드포인트 순간이동(Teleport) 없이 연속적이고 부드러운 전신 협응 궤적 생성
- **발 재배치 물리 기반 (Foot Reposition Foundation)**:
  - 왼발 발디딤 전이(LF: H1 -> H6, $\Delta p = 0.164\,\text{m}$, 0.10m 상방 스텝업) 구현
  - 기존 지지 해제, 나머지 3개 사지만으로 전신 중력 지지 유지, 연속적 이동 및 상단 스텝 사이트(`site_step_H6`) 물리적 안착
  - 생체역학적 발 방향 벡터(+Y 토우 방향) 및 기립 피치 바이어스를 통한 무릎 가동 범위 충돌 방지
- **시뮬레이션 리셋 없는 다중 전이 순차 실행 (`execute_transition_sequence`)**:
  - **시뮬레이션 상태 완전 보존**: 이전 전이에서 도출된 실제 물리 상태(`qpos`, `qvel`, 활성 구속조건, 실제 루트 위치)로부터 다음 전이를 즉시 연속 실행
  - 전이 간 `mj_resetData`, 루트 상태 덮어쓰기, 좌표 텔레포트 일체 배제
  - 3단계 연속 클라이밍 시퀀스 검증: Move 1 (RH H4->H5) $\to$ Move 2 (LF H1->H6) $\to$ Move 3 (LH H3->H7)
  - 단조 증가하는 물리 타임스탬프 및 연속적인 루트 고도($z \approx 1.21\,\text{m} \to 1.23\,\text{m} \to 1.23\,\text{m}$) 유지
- **상태 기반 실용적 안정화 판정 기준 (`check_stabilization_readiness`)**:
  - 임의의 고정 딜레이가 아닌 물리 상태 기반 판정: 루트 선속도($<0.60\,\text{m/s}$), 각속도($<3.50\,\text{rad/s}$), 관절 속도 노름, 최소 지지 수($\ge 3$), 루트 최저 높이($\ge 0.85\,\text{m}$) 종합 평가
  - 격렬한 흔들림/추락 상태와 다음 동작 수용 가능한 안정 상태를 확실히 판별
- **시퀀스 실패 전파 제어**:
  - 시퀀스 도중 한 전이가 실패할 경우 즉시 시뮬레이션을 중단하고 후속 전이 실행 배제, 실패 원인과 시점의 물리 상태를 명시적으로 반환
- **다양한 클라이머 신체 프로파일 검증**:
  - `base`, `compact_strong`, `long_reach_lower_grip` 3가지 프로파일 모두 3단계 전이 시퀀스 100% 성공
- **오프스크린 시퀀스 렌더링 및 2x2 비교 몽타주 (`sequence_montage.png`)**:
  - 375프레임 30fps 고해상도 시퀀스 비디오(`sequence_base.mp4`) 및 애니메이션 GIF(`sequence_base.gif`)
  - 실시간 HUD 오버레이: Move 회차, 이동 사지, 출발/도착 홀드, 현재 페이즈, 접촉 상태, 물리 시뮬레이션 시간, 루트 z 고도 표시

## 아직 하지 않는 것

- RL training (PPO, 정책 학습 등)
- 임의 경로에 대한 전역 모션 플래닝 (global pathfinding over arbitrary routes)
- DEADPOINT / DYNO
- 3D reconstruction
- 피로도(Fatigue) 모델링

따라서 이 버전은 **학습 결과나 완성된 climbing policy가 아니라**, 다중 사지 전이 프리미티브와 상태 리셋 없는 순차 실행 물리 엔진 기초를 검증하는 단계입니다.

## 실행 환경 (Canonical Environment: Ubuntu Linux 24.04)

### 가상환경 활성화 및 환경변수 설정
```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```
*(Windows PowerShell 사용 시: `$env:PYTHONPATH="src;.venv\Lib\site-packages;."`)*

### 테스트 실행 (Headless / SSH 지원)
```bash
python -m unittest discover -s tests -v
```

### 데모 스크립트 실행
```bash
python scripts/run_demo.py
```

생성물:
- `outputs/scene.json`
- `outputs/model_long_reach_lower_grip.xml`
- `outputs/model_compact_strong.xml`
- `outputs/profile_binding.png`
- `outputs/grip_demo.json`

### 오프스크린 시각 검증 렌더러 (Headless / EGL)
SSH 원격 터미널 환경 또는 로컬 GUI 디스플레이가 없는 환경에서 MuJoCo EGL Offscreen Renderer API를 사용하여 정적 4점 스탠스 검증 이미지와 단일 사지 전이 비디오/키프레임을 생성할 수 있습니다:
```bash
# EGL 기반 헤드리스 렌더링 (NVIDIA GPU / Linux)
MUJOCO_GL=egl python scripts/render_demo.py --mode all

# 정적 4점 지지 스탠스 검증 이미지 생성
MUJOCO_GL=egl python scripts/render_demo.py --mode stance --profile compact_strong

# 단일 사지 리치 및 재부착 전이 비디오 생성
MUJOCO_GL=egl python scripts/render_demo.py --mode transition --profile base
```
*(Windows 환경에서는 `python scripts/render_demo.py --mode all` 실행)*

생성물 (`outputs/visual/`):
- `outputs/visual/stance_compact_strong.png` (1280x720 고해상도 정적 스탠스 검증 이미지)
- `outputs/visual/transition_base.mp4` (1280x720 @ 30fps 단일 사지 리치 및 재부착 전이 비디오)
- `outputs/visual/transition_base.gif` (30fps 애니메이션 GIF)
- `outputs/visual/transition_00_initial.png` (초기 4점 평형 스탠스)
- `outputs/visual/transition_01_release.png` (구속 해제 이벤트)
- `outputs/visual/transition_02_support.png` (3점 자유 루트 벽면 지지 유지)
- `outputs/visual/transition_03_reach.png` (탐색적 리치 궤적)
- `outputs/visual/transition_04_attach.png` (홀드 자격 판정 및 재부착)
- `outputs/visual/transition_05_stabilized.png` (안정화된 새 스탠스 안착)

### 3D 씬 뷰어 실행 (대화형 뷰어)
로컬 데스크톱 디스플레이 환경(X11/Wayland 또는 Windows)에서 대화형 GLFW 뷰어 실행 (주의: 대화형 로컬 렌더링 시에는 EGL 환경변수를 설정하지 않고 실행합니다):
```bash
# 단일 사지 리치 전이 모드 (결정론적 전이 시퀀스 시각화)
python scripts/view_scene.py --profile base --mode transition

# 정적 4점 지지 클라이밍 스탠스 모드 (양손 홀드 결합, 양발 디딤, 자유 루트 전신 중력 물리)
python scripts/view_scene.py --profile compact_strong --mode stance

# 헤드리스 무결성 검증 (SSH 환경용)
python scripts/view_scene.py --headless --mode stance
python scripts/view_scene.py --headless --mode transition
```

> **주의:** `--mode`는 MuJoCo GUI 창 안에서 변경하는 것이 아니라, **CLI 커맨드라인 옵션(`--mode transition`, `--mode stance`, `--mode pose`, `--mode free`)**으로 지정하여 실행합니다.

모드 설명:
- `--mode transition`: 단일 사지 해제, 3점 지지 유지, 탐색적 리치, 홀드 자격 판정 및 재부착, 안정화 스탠스 안착의 6단계 전이 과정을 시각화합니다.
- `--mode stance` (기본값): MuJoCo connect 구속을 통해 양손을 홀드(H3, H4)에 부착하고 양발을 홀드(H1, H2)에 지지한 상태에서, 임시 루트 지지 없이 순수 자유 루트와 중력 하에서 정적 벽면 지지 상태를 관찰합니다.
- `--mode pose`: 25개 관절 액추에이터 구동 및 엔드이펙터 사이트 시각적 디버깅을 위한 모드입니다. 자유 직립 밸런스 제어기가 아니므로, 시각 검사를 위해 임시 루트 지지가 적용됩니다.
- `--mode free` (또는 `passive`): 어떠한 루트 지지나 홀드 결합도 없는 실제 자유 루트 물리 모드입니다. 중력에 의해 바닥으로 낙하합니다.

기타 옵션:
- `--profile {base, compact_strong, long_reach_lower_grip}`
- `--headless` (GUI 창 없이 모델/물리/사이트/스탠스/전이 무결성 검증, SSH 환경용)
- `--duration SECONDS` (지정 시간 후 자동 종료)

## 설계상 다음 단계

Prototype v1.6:
1. Sequential hold-to-hold progression (multiple transitions along a route)
2. Feasibility evaluation & motion envelope analysis
3. Dynamic momentum transfer (deadpoint preparatory mechanics)
