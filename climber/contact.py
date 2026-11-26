"""Hand grip and foot support rules.

[한국어 설명]
손·발이 홀드에 "붙어 있는지"를 판정하는 파일. 매 시뮬레이션 스텝마다 update(data)를 호출한다.

손 (잡기)
  - 잡기 시작: 손바닥이 홀드에 닿아 있고, 손바닥 방향과 그 접촉점의 홀드 법선(홀드 → 손)이
    서로 정면으로 마주 볼 때 (사이 각도 ≤ grasp_angle_deg, 기본 20°).
    → 실제로 닿은 접촉점에서 손과 홀드를 고정점(connect)으로 연결한다. 그래서 손이 홀드를 파고들지 않는다.
  - 놓치기: 손이 홀드에 거는 힘(하중)이
      (1) 홀드가 버틸 수 있는 방향을 벗어나거나: 하중 방향과 홀드의 pull_dir 사이 각도 > pull_cone_deg
          (홀드마다 다름. 저그는 아래 방향, 기본 ±40°. scene.Hold 참고)
      (2) 그립 한계를 넘으면: |하중| > capacity (기본 BodyProfile.grip_capacity)
    가 release_patience 스텝(기본 10 = 0.02초) 연속으로 이어질 때.
    (하중이 cone_min_load(30N)보다 작으면 방향이 의미 없으므로 방향 검사는 하지 않는다.)
    놓친 손은 regrip_cooldown 스텝(기본 150 = 0.3초) 동안 다시 잡지 못한다 (미끄러져 떨어지는 시간).
  - 악력 피로 (fatigue_step): 손마다 악력(capacity)을 따로 갖는다.
      잡고 있으면 줄어든다: 하중이 악력과 같으면 최저치 위의 악력이 60초마다 절반 (하중이 작으면 그만큼 느리게)
      놓고 있으면 회복된다: 약 30초면 95% 회복
      아무리 지쳐도 처음 악력의 30% 밑으로는 안 내려간다
    값은 BodyProfile(grip_fatigue_halflife, grip_recovery_time, grip_min_fraction)에서 온다.

발 (지지)
  - 고정하지 않는다. 발은 누르기만 하고 당기지 못하므로 MuJoCo의 마찰 접촉에 그대로 맡긴다.
  - '지지 중' 판정만 한다: 발바닥이 무언가(홀드·벽·바닥)를 normal_force_min(10N) 이상 누르고,
    발바닥 방향과 그 면의 법선이 grasp_angle_deg 이내로 마주 보며, 미끄러지는 중이 아닐 때.
  - 판정 결과는 강화학습의 관찰값·보상에 쓴다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

SIDES = ("left", "right")


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(a @ b / (na * nb), -1.0, 1.0))))


@dataclass
class HandState:
    capacity: float                       # 현재 그립 한계 (N)
    hold: str | None = None               # 잡고 있는 홀드 이름 (없으면 None)
    touching: str | None = None           # 닿아 있는 홀드 (잡지 않아도)
    palm_angle_deg: float | None = None   # 손바닥 방향 vs 홀드 법선 각도 (닿아 있을 때)
    load: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 손이 홀드에 거는 힘 (N, 월드 좌표)
    load_angle_deg: float | None = None   # 하중 방향 vs 홀드의 버틸 수 있는 방향(pull_dir) 각도
    last_release: str | None = None       # 마지막으로 놓친 이유
    grip_time: float = 0.0                # 현재 홀드를 잡고 있는 시간 (s)


@dataclass
class FootState:
    supported: bool = False
    surface: str | None = None            # 밟고 있는 면 (홀드 이름 / wall / floor)
    normal_force: float = 0.0             # 누르는 힘 (N)
    sole_angle_deg: float | None = None   # 발바닥 방향 vs 면 법선 각도
    slipping: bool = False


def fatigue_step(capacity: float, base: float, load: float, dt: float, gripping: bool,
                 halflife: float = 60.0, recovery_time: float = 30.0, min_fraction: float = 0.3) -> float:
    """악력 한 스텝 갱신.

    잡고 있을 때: 최저치(base × min_fraction) 위의 악력이 줄어든다. 줄어드는 빠르기 ∝ 하중/현재악력.
        하중이 악력과 같으면 halflife초마다 절반, 하중이 악력의 1/4이면 4배 느리게.
    놓고 있을 때: 처음 악력(base)을 향해 회복. recovery_time초면 95% 회복 (시간상수 = recovery_time / 3).
    """
    floor = base * min_fraction
    if gripping:
        ratio = min(max(load / max(capacity, 1e-6), 0.0), 1.0)
        rate = math.log(2.0) / halflife * ratio
        return max(floor, capacity - rate * (capacity - floor) * dt)
    return min(base, capacity + (base - capacity) * dt / (recovery_time / 3.0))


class ContactModel:
    def __init__(self, model: mujoco.MjModel, grip_capacity: float = 850.0,
                 grasp_angle_deg: float = 20.0, release_patience: int = 10,
                 regrip_cooldown: int = 150, normal_force_min: float = 10.0,
                 cone_min_load: float = 30.0, grip_fatigue_halflife: float = 60.0,
                 grip_recovery_time: float = 30.0, grip_min_fraction: float = 0.3,
                 profile=None):
        """profile(BodyProfile)을 주면 악력·피로 값을 그 사람 것으로 쓴다 (권장)."""
        if profile is not None:
            grip_capacity = profile.grip_capacity
            grip_fatigue_halflife = profile.grip_fatigue_halflife
            grip_recovery_time = profile.grip_recovery_time
            grip_min_fraction = profile.grip_min_fraction
        self.m = model
        self.grasp_angle_deg = grasp_angle_deg
        self.release_patience = release_patience
        self.regrip_cooldown = regrip_cooldown
        self.normal_force_min = normal_force_min
        self.cone_min_load = cone_min_load  # 이보다 작은 하중은 방향이 의미 없으므로 방향 검사 안 함
        self.grip_capacity = grip_capacity  # 지치기 전 악력 (손마다)
        self.fatigue = dict(halflife=grip_fatigue_halflife, recovery_time=grip_recovery_time,
                            min_fraction=grip_min_fraction)

        def seg_geoms(body: str) -> set[int]:
            bid = model.body(body).id
            return {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid and model.geom_contype[g]}

        self.hand_geoms = {s: seg_geoms(f"{s}_hand") for s in SIDES}
        self.foot_geoms = {s: seg_geoms(f"{s}_foot") for s in SIDES}
        self.palm_site = {s: model.site(f"{s}_palm").id for s in SIDES}
        self.sole_site = {s: model.site(f"{s}_sole").id for s in SIDES}
        self.hold_of_geom: dict[int, str] = {}
        for g in range(model.ngeom):
            name = model.geom(g).name
            if name.startswith("hold_"):
                self.hold_of_geom[g] = name[len("hold_"):]
        # 홀드별 버틸 수 있는 방향과 허용 각도 (model.py가 <custom>에 저장)
        self.pull = {}
        for h in set(self.hold_of_geom.values()):
            dx, dy, dz, cone = model.numeric(f"hold_{h}_pull").data
            v = np.array([dx, dy, dz], dtype=float)
            self.pull[h] = (v / np.linalg.norm(v), float(cone))
        self.eq = {(s, h): model.equality(f"grasp_{s}_{h}").id
                   for s in SIDES for h in set(self.hold_of_geom.values())}
        climber_root = model.body("pelvis").id
        self.climber_geoms = {g for g in range(model.ngeom) if model.body_rootid[model.geom_bodyid[g]] == climber_root}
        self.reset()

    # ------------------------------------------------------------------ 상태 초기화
    def reset(self, data: mujoco.MjData | None = None) -> None:
        self.hands = {s: HandState(capacity=self.grip_capacity) for s in SIDES}
        self.feet = {s: FootState() for s in SIDES}
        self._violation = {s: 0 for s in SIDES}
        self._cooldown = {s: 0 for s in SIDES}
        self._grip_normal_local = {s: None for s in SIDES}  # 잡은 순간 홀드 법선 (홀드 body 좌표)
        if data is not None:
            for eid in self.eq.values():
                data.eq_active[eid] = 0

    # ------------------------------------------------------------------ 매 스텝 호출
    def update(self, data: mujoco.MjData) -> "ContactModel":
        """mj_step(또는 mj_forward) 직후에 호출. 손 잡기/놓치기를 처리하고 발 지지를 판정한다."""
        touches = self._hand_touches(data)
        for s in SIDES:
            if self.hands[s].hold is None:
                self._try_grip(data, s, touches.get(s))
                # 쉬는 손은 악력 회복
                st = self.hands[s]
                st.capacity = fatigue_step(st.capacity, self.grip_capacity, 0.0, self.m.opt.timestep,
                                           gripping=False, **self.fatigue)
            else:
                self._check_release(data, s)
            if self._cooldown[s] > 0:
                self._cooldown[s] -= 1
        for s in SIDES:
            self._update_foot(data, s)
        return self

    # ------------------------------------------------------------------ 손
    def _hand_touches(self, data):
        """손-홀드 접촉 중 손마다 가장 깊은 것: (홀드 이름, 접촉점, 홀드→손 법선)."""
        best = {}
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = c.geom1, c.geom2
            for s in SIDES:
                if g1 in self.hand_geoms[s] and g2 in self.hold_of_geom:
                    hold, normal = self.hold_of_geom[g2], -c.frame[:3]
                elif g2 in self.hand_geoms[s] and g1 in self.hold_of_geom:
                    hold, normal = self.hold_of_geom[g1], c.frame[:3].copy()
                else:
                    continue
                if s not in best or c.dist < best[s][3]:
                    best[s] = (hold, c.pos.copy(), np.array(normal), c.dist)
        return best

    def _try_grip(self, data, s, touch) -> None:
        st = self.hands[s]
        st.load = np.zeros(3)
        st.load_angle_deg = None
        st.grip_time = 0.0
        if touch is None:
            st.touching, st.palm_angle_deg = None, None
            return
        hold, point, n_hold, _ = touch
        palm_dir = data.site_xmat[self.palm_site[s]].reshape(3, 3)[:, 2]
        st.touching = hold
        st.palm_angle_deg = _angle_deg(palm_dir, -n_hold)   # 손바닥이 홀드 쪽(-법선)을 향하면 0°
        if self._cooldown[s] > 0 or st.palm_angle_deg > self.grasp_angle_deg:
            return
        # 잡기: 실제 접촉점을 손 좌표 / 홀드 좌표로 각각 바꿔 고정점으로 설정
        eid = self.eq[(s, hold)]
        hb, ob = self.m.body(f"{s}_hand").id, self.m.body(f"hold_{hold}").id
        hR, oR = data.xmat[hb].reshape(3, 3), data.xmat[ob].reshape(3, 3)
        self.m.eq_data[eid][0:3] = hR.T @ (point - data.xpos[hb])
        self.m.eq_data[eid][3:6] = oR.T @ (point - data.xpos[ob])
        data.eq_active[eid] = 1
        self._grip_normal_local[s] = oR.T @ n_hold
        self._violation[s] = 0
        st.hold = hold

    def _grip_force_on_hand(self, data, eid) -> np.ndarray:
        rows = [i for i in range(data.nefc)
                if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and data.efc_id[i] == eid]
        return data.efc_force[rows].copy() if len(rows) == 3 else np.zeros(3)

    def _contact_force_on_hand(self, data, s, hold) -> np.ndarray:
        """그 홀드가 손을 직접 미는 접촉력 합 (월드 좌표)."""
        total, f6 = np.zeros(3), np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in self.hand_geoms[s] and self.hold_of_geom.get(c.geom2) == hold:
                sign = -1.0   # 접촉력은 geom1 → geom2 방향으로 geom2를 민다. 손이 geom1이면 반대
            elif c.geom2 in self.hand_geoms[s] and self.hold_of_geom.get(c.geom1) == hold:
                sign = 1.0
            else:
                continue
            mujoco.mj_contactForce(self.m, data, i, f6)
            total += sign * (c.frame.reshape(3, 3).T @ f6[:3])
        return total

    def _check_release(self, data, s) -> None:
        st = self.hands[s]
        hold = st.hold
        eid = self.eq[(s, hold)]
        # 손이 홀드에 거는 힘 = -(고정점이 손을 당기는 힘 + 홀드 표면이 손을 미는 접촉력)
        # 둘 중 하나만 보면, 손이 홀드에 살짝 눌린 상태에서 두 힘이 서로 상쇄되는 걸 큰 하중으로 착각한다
        st.load = -(self._grip_force_on_hand(data, eid) + self._contact_force_on_hand(data, s, hold))
        mag = float(np.linalg.norm(st.load))
        pull_dir, cone = self.pull[hold]
        st.load_angle_deg = _angle_deg(st.load, pull_dir) if mag > 1.0 else 0.0
        st.grip_time += self.m.opt.timestep
        # 악력 피로: 잡고 있는 동안, 하중이 클수록 빨리 줄어든다
        st.capacity = fatigue_step(st.capacity, self.grip_capacity, mag, self.m.opt.timestep,
                                   gripping=True, **self.fatigue)
        reason = None
        if mag > st.capacity:
            reason = f"grip limit ({mag:.0f} N > {st.capacity:.0f} N)"
        elif mag > self.cone_min_load and st.load_angle_deg > cone:
            reason = f"outside pull direction ({st.load_angle_deg:.0f}° > {cone:.0f}°)"
        self._violation[s] = self._violation[s] + 1 if reason else 0
        if self._violation[s] >= self.release_patience:
            self.release(data, s, reason)

    def release(self, data, s, reason: str = "released") -> None:
        """손을 놓는다 (강화학습 정책이 일부러 손을 뗄 때도 호출)."""
        st = self.hands[s]
        if st.hold is not None:
            data.eq_active[self.eq[(s, st.hold)]] = 0
        st.hold, st.last_release = None, reason
        st.grip_time = 0.0
        self._violation[s] = 0
        self._cooldown[s] = self.regrip_cooldown

    # ------------------------------------------------------------------ 발
    def _update_foot(self, data, s) -> None:
        fs = self.feet[s]
        fs.supported, fs.surface, fs.normal_force, fs.sole_angle_deg, fs.slipping = False, None, 0.0, None, False
        sole_dir = data.site_xmat[self.sole_site[s]].reshape(3, 3)[:, 2]
        f6 = np.zeros(6)
        best = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in self.foot_geoms[s] and c.geom2 not in self.climber_geoms:
                other, n_surface = c.geom2, -c.frame[:3]
            elif c.geom2 in self.foot_geoms[s] and c.geom1 not in self.climber_geoms:
                other, n_surface = c.geom1, c.frame[:3].copy()
            else:
                continue
            mujoco.mj_contactForce(self.m, data, i, f6)
            normal, tangential = float(f6[0]), float(np.linalg.norm(f6[1:3]))
            if normal <= best:
                continue
            best = normal
            name = self.m.geom(other).name
            fs.surface = self.hold_of_geom.get(other, name or "world")
            fs.normal_force = normal
            fs.sole_angle_deg = _angle_deg(sole_dir, -n_surface)
            fs.slipping = tangential >= 0.95 * float(c.friction[0]) * normal
        fs.supported = (fs.normal_force >= self.normal_force_min and fs.sole_angle_deg is not None
                        and fs.sole_angle_deg <= self.grasp_angle_deg and not fs.slipping)

    # ------------------------------------------------------------------ 요약
    def summary(self) -> dict:
        return {
            "hands": {s: {"hold": h.hold, "touching": h.touching,
                          "palm_angle": None if h.palm_angle_deg is None else round(h.palm_angle_deg, 1),
                          "load_N": round(float(np.linalg.norm(h.load)), 1),
                          "capacity_N": round(h.capacity, 1),
                          "load_angle": None if h.load_angle_deg is None else round(h.load_angle_deg, 1),
                          "last_release": h.last_release} for s, h in self.hands.items()},
            "feet": {s: {"supported": f.supported, "surface": f.surface, "normal_N": round(f.normal_force, 1),
                         "sole_angle": None if f.sole_angle_deg is None else round(f.sole_angle_deg, 1),
                         "slipping": f.slipping} for s, f in self.feet.items()},
        }
