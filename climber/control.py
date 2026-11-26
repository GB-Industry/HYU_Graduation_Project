"""Simple pose-hold controller (baseline, not the research controller).

[한국어 설명]
관절마다 "목표 각도를 유지해" 하는 가장 단순한 PD 제어기.
    토크 = kp × (목표각 - 현재각) - kd × 관절속도   (모터 최대 토크에서 잘림)
- 벽에 매달린 시작 자세를 그대로 유지시키는 용도 (뷰어 hang 모드, 테스트).
- 팔은 약하게(kp 100), 몸통·다리는 강하게(kp 400). 팔을 뻣뻣하게 하면 손 고정점과 싸워서
  손에 걸리는 힘 방향이 흔들리고 결국 놓친다 (실험으로 확인: 전부 400이면 0.3초 만에 놓침).
- 나중에 강화학습 결과와 비교할 '스크립트 제어' 기준선이 될 수 있다. 이 제어기는 이동은 못 한다.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_KEYS = ("shoulder", "elbow", "twist", "wrist")


class PoseHold:
    def __init__(self, model: mujoco.MjModel, target_qpos: np.ndarray,
                 kp_arm: float = 100.0, kd_arm: float = 10.0,
                 kp_body: float = 400.0, kd_body: float = 25.0):
        self.target = np.asarray(target_qpos, dtype=float).copy()
        self.gear = model.actuator_gear[:, 0].copy()
        joints = [model.actuator_trnid[a, 0] for a in range(model.nu)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.vadr = np.array([model.jnt_dofadr[j] for j in joints])
        arm = np.array([any(k in model.joint(j).name for k in ARM_KEYS) for j in joints])
        self.kp = np.where(arm, kp_arm, kp_body)
        self.kd = np.where(arm, kd_arm, kd_body)

    def __call__(self, data: mujoco.MjData) -> np.ndarray:
        """data.ctrl에 넣을 모터 명령 (-1 ~ 1)."""
        torque = self.kp * (self.target[self.qadr] - data.qpos[self.qadr]) - self.kd * data.qvel[self.vadr]
        return np.clip(torque / self.gear, -1.0, 1.0)
