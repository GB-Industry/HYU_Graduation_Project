"""Measure a built climber: standing dimensions and per-segment masses.

[한국어 설명]
만들어진 캐릭터를 실제로 T자 자세로 세워 놓고 키·팔 너비·질량 등을 재는 파일.
실험 결과를 정리할 때 "이 체형은 키 몇 cm" 같은 수치의 근거로 쓴다.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .poses import t_pose

SEGMENTS = ("pelvis", "abdomen", "chest", "head", "left_upper_arm", "left_forearm", "left_hand",
            "left_thigh", "left_shin", "left_foot")


def body_stats(model: mujoco.MjModel) -> dict[str, float]:
    """Standing height, joint heights, arm span (fingertip to fingertip) and total mass.

    반환: height(발바닥~머리 꼭대기), shoulder_height, hip_joint_height,
          arm_span(손끝~손끝), total_mass  (단위 m, kg)
    """
    d = mujoco.MjData(model)
    d.qpos[:] = t_pose(model)
    mujoco.mj_forward(model, d)
    head = model.body("head").id
    top = max(d.geom_xpos[i][2] + model.geom_rbound[i] for i in range(model.ngeom)
              if model.geom_bodyid[i] == head and model.geom_contype[i])

    def tip(side: str) -> float:
        hand = [i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body(f"{side}_hand").id
                and model.geom_contype[i]][0]
        x = d.geom_xpos[hand][0]
        axis = d.geom_xmat[hand].reshape(3, 3)[:, 2]  # paddle long axis
        return float(x + np.sign(x) * abs(axis[0]) * model.geom_size[hand][2])  # paddle tip

    root = model.body("pelvis").id
    return {
        "height": float(top),
        "shoulder_height": float(d.xpos[model.body("left_upper_arm").id][2]),
        "hip_joint_height": float(d.xpos[model.body("left_thigh").id][2]),
        "arm_span": abs(tip("left") - tip("right")),
        "total_mass": float(sum(model.body_mass[b] for b in range(model.nbody) if model.body_rootid[b] == root)),
    }


def segment_masses(model: mujoco.MjModel) -> dict[str, float]:
    """Mass of each body segment (limbs: left side; right side is identical).  부위별 질량 (kg)"""
    return {s: float(model.body_mass[model.body(s).id]) for s in SEGMENTS}
