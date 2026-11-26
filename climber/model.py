"""MJCF generator for the mannequin climber.

The look (light-orange wooden drawing mannequin) and the physics body are the same
primitives: segment shapes carry mass and collide with the world; joint balls are
visual only. No meshes, so the model stays cheap to simulate.

Joint layout, ranges and torque ceilings follow the first physics prototype
(free root + 25 hinges), plus a forearm twist per arm so the palm can face a hold: waist 3, each arm 6, each leg 6 = 27.

[한국어 설명]
BodyProfile(신체 사양서)과 Wall(벽·홀드)을 받아 MuJoCo 모델(MJCF XML)을 만드는 생성기. 이 프로젝트의 핵심 파일.
- 몸: 골반(자유 관절) → 허리 → 가슴 → 머리/팔, 골반 → 다리 의 트리 구조
- 관절 27개, 관절마다 토크 모터 1개 (강화학습 정책은 ctrl 27개를 -1~1로 출력)
  (팀원 프로토타입 25개 + 손바닥을 홀드 쪽으로 돌리는 아래팔 회전 2개)
- 보이는 도형이 그대로 질량을 갖고 벽·바닥과 충돌. 관절 볼(구)은 장식이라 질량·충돌 없음
- 몸끼리는 충돌하지 않음 (자기 충돌 없음)
- 손 그립: 손과 홀드를 붙이는 connect 제약을 꺼진 상태로 만들어 두고, 켜면 "잡기"
보통은 build_model(profile)만 호출하면 된다.
"""
from __future__ import annotations

from xml.sax.saxutils import escape

import mujoco

from .profile import BASELINE, BodyProfile
from .scene import Wall, default_wall

# 색상 (R G B 투명도)
SKIN = "0.98 0.74 0.50 1"        # 몸: 연한 주황
JOINT_BALL = "0.94 0.65 0.41 1"  # 관절 볼: 조금 더 진한 주황
HOLD_RGBA = "0.30 0.62 0.68 1"   # 홀드: 청록
SIDES = ("left", "right")

# 관절 사양표. 이름: (회전축, 최소각°, 최대각°, 감쇠, 최대토크 N·m)
#   회전축  : "1 0 0" = 좌우축(앞뒤로 숙이기/들기), "0 1 0" = 앞뒤축(옆으로 기울이기/벌리기), "0 0 1" = 세로축(비틀기)
#   최소/최대각: 가동범위. 실제 적용값 = 이 값 × BodyProfile.rom_scale (유연성)
#   감쇠    : 빨리 돌수록 생기는 저항 (덜렁거림 방지)
#   최대토크 : 그 관절의 최대 힘 (근력). 실제 적용값 = 이 값 × BodyProfile.strength_scale
# name: (axis, lo_deg, hi_deg, damping, max_torque_Nm)
JOINTS = {
    "waist_yaw": ("0 0 1", -35, 35, 4.0, 90),
    "waist_pitch": ("1 0 0", -25, 45, 4.0, 110),
    "waist_roll": ("0 1 0", -25, 25, 4.0, 90),
    "shoulder_pitch": ("1 0 0", -120, 160, 2.0, 85),
    "shoulder_roll": ("0 1 0", -80, 80, 2.0, 85),
    "shoulder_yaw": ("0 0 1", -80, 80, 2.0, 75),
    "elbow": ("1 0 0", 0, 145, 1.5, 65),
    "forearm_twist": ("0 0 1", -85, 85, 1.0, 20),   # 아래팔 돌리기 (손바닥 엎기/뒤집기)
    "wrist": ("1 0 0", -60, 60, 1.0, 35),
    "hip_pitch": ("1 0 0", -110, 120, 3.0, 140),
    "hip_yaw": ("0 0 1", -45, 45, 3.0, 80),
    "knee": ("-1 0 0", 0, 150, 2.5, 130),
    "ankle_pitch": ("1 0 0", -45, 35, 2.0, 55),
    "ankle_roll": ("0 1 0", -25, 25, 2.0, 40),
}
# hip roll is asymmetric so each leg abducts outward more than it crosses inward
# 고관절 옆 벌림만 좌우 따로: 다리는 바깥으로 벌리는 범위가 안쪽으로 모으는 범위보다 넓다
HIP_ROLL = {"left": ("0 1 0", -50, 60, 3.0, 110), "right": ("0 1 0", -60, 50, 3.0, 110)}

# Fraction of total body mass per segment, de Leva (1996) adult male.
# Trunk = upper (chest, split rib cage / shoulder yoke) + middle (abdomen) + lower (pelvis).
# Limb entries are per side. Sums to 1.0.
# 부위별 질량 비율 (실제 성인 남성 통계). 실제 질량 = 비율 × total_mass. 팔다리는 한쪽 기준.
MASS_FRACTION = {"head": 0.0694, "chest": 0.1246, "yoke": 0.0350, "abdomen": 0.1633, "pelvis": 0.1117,
                 "upper_arm": 0.0271, "forearm": 0.0162, "hand": 0.0061,
                 "thigh": 0.1416, "shin": 0.0433, "foot": 0.0137}
SOLE_DEPTH = 0.06  # ankle joint height above the sole, ~0.035 H   발목 관절 ~ 발바닥 높이 (m)


def _f(v: float) -> str:
    """숫자를 XML에 넣기 좋은 짧은 문자열로 바꾼다."""
    return f"{v:.5g}"


def actuated_joints() -> list[str]:
    """관절(=모터) 이름 27개를 모터 순서대로 돌려준다. data.ctrl[i]가 어느 관절인지 알 때 사용.
    순서: 허리 3 → 왼팔 6 → 오른팔 6 → 왼다리 6 → 오른다리 6"""
    names = ["waist_yaw", "waist_pitch", "waist_roll"]
    for s in SIDES:
        names += [f"{s}_{j}" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "forearm_twist", "wrist")]
    for s in SIDES:
        names += [f"{s}_{j}" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    return names


def _joint_spec(name: str):
    """"left_elbow" 같은 이름으로 위 사양표에서 해당 관절 사양을 찾는다."""
    side, _, base = name.partition("_")
    if base == "hip_roll":
        return HIP_ROLL[side]
    return JOINTS[base if side in SIDES else name]


def _joint(name: str, p: BodyProfile) -> str:
    """관절 XML 한 줄. 가동범위에 rom_scale(유연성)을 곱해 넣는다."""
    axis, lo, hi, damping, _ = _joint_spec(name)
    lo, hi = lo * p.rom_scale, hi * p.rom_scale
    return f'<joint name="{name}" axis="{axis}" range="{_f(lo)} {_f(hi)}" damping="{_f(damping)}"/>'


def _seg(kind: str, attrs: str, mass: float | None = None) -> str:
    """A body segment: collides with the world and carries mass.  물리 도형: 질량 O, 벽·바닥과 충돌 O"""
    return f'<geom class="seg" type="{kind}" {attrs} mass="{_f(mass)}"/>'


def _ball(pos: str, r: float) -> str:
    """A visual-only joint ball.  관절 볼(장식): 질량 X, 충돌 X"""
    return f'<geom class="vis" type="sphere" pos="{pos}" size="{_f(r)}" rgba="{JOINT_BALL}"/>'


def _arm(side: str, p: BodyProfile, mass) -> str:
    """팔 하나: 위팔(어깨 관절 3) → 아래팔(팔꿈치 1) → 손(손목 1).
    각 body의 원점 = 관절 위치, 팔은 아래(-z) 방향으로 뻗는다. 손 끝의 site는 그립 기준점."""
    sx = p.shoulder_width / 2 * (-1 if side == "left" else 1)
    palm = 1.0 if side == "left" else -1.0  # 손바닥 방향: 왼손은 +x(몸 안쪽), 오른손은 -x
    chest_h = 0.25 * p.torso_length / 0.55
    ua, fa = p.upper_arm_length, p.forearm_length
    j = lambda n: _joint(f"{side}_{n}", p)
    return f'''
          <body name="{side}_upper_arm" pos="{_f(sx)} 0 {_f(chest_h * 0.32)}">
            {j("shoulder_pitch")}{j("shoulder_roll")}{j("shoulder_yaw")}
            {_ball("0 0 0", 0.045)}
            {_seg("capsule", f'fromto="0 0 -0.05 0 0 {_f(-(ua - 0.05))}" size="0.036"', mass("upper_arm"))}
            <body name="{side}_forearm" pos="0 0 {_f(-ua)}">
              {j("elbow")}{j("forearm_twist")}
              {_ball("0 0 0", 0.033)}
              {_seg("capsule", f'fromto="0 0 -0.04 0 0 {_f(-(fa - 0.04))}" size="0.030"', mass("forearm"))}
              <body name="{side}_hand" pos="0 0 {_f(-fa)}">
                {j("wrist")}
                {_ball("0 0 0", 0.024)}
                <!-- 손: 5mm 이내로 가까워지면 '닿음'으로 감지만 하고(margin), 실제로 미는 힘은 맞닿은 뒤부터(gap) -->
                {_seg("ellipsoid", 'pos="0 0.004 -0.058" size="0.020 0.034 0.046" margin="0.005" gap="0.005"', mass("hand"))}
                <!-- 손바닥: 팔을 내렸을 때 몸 안쪽을 향하는 넓적한 면. z축 = 손바닥이 향하는 방향 -->
                <site name="{side}_palm" pos="{_f(0.020 * palm)} 0.004 -0.058" zaxis="{_f(palm)} 0 0" size="0.015"/>
              </body>
            </body>
          </body>'''


def _leg(side: str, p: BodyProfile, mass) -> str:
    """다리 하나: 허벅지(고관절 3) → 정강이(무릎 1) → 발(발목 2).
    발은 보이지 않는 상자(실제로 밟는 부분, 마찰 1.8) + 보이는 타원체(겉모습)로 나뉜다.
    발목 관절 2개는 같은 점에 겹쳐 있다 (한 점에서 두 방향으로 도는 관절)."""
    hip_x = p.hip_width / 2
    hx = 0.85 * hip_x * (-1 if side == "left" else 1)
    pelvis_h = 0.14 * p.torso_length / 0.55
    th, sh = p.thigh_length, p.shin_length
    j = lambda n: _joint(f"{side}_{n}", p)
    return f'''
      <body name="{side}_thigh" pos="{_f(hx)} 0 {_f(-pelvis_h * 0.35)}">
        {j("hip_pitch")}{j("hip_roll")}{j("hip_yaw")}
        {_ball("0 0 0", 0.048)}
        {_seg("capsule", f'fromto="0 0 -0.06 0 0 {_f(-(th - 0.06))}" size="0.052"', mass("thigh"))}
        <body name="{side}_shin" pos="0 0 {_f(-th)}">
          {j("knee")}
          {_ball("0 0 0", 0.045)}
          {_seg("capsule", f'fromto="0 0 -0.05 0 0 {_f(-(sh - 0.05))}" size="0.040"', mass("shin"))}
          <body name="{side}_foot" pos="0 0.02 {_f(-sh)}">
            {j("ankle_pitch")}{j("ankle_roll")}
            {_ball("0 0 0", 0.032)}
            <geom class="seg" type="box" pos="0 0.035 {_f(-(SOLE_DEPTH - 0.018))}" size="0.040 0.095 0.018" mass="{_f(mass("foot"))}" friction="1.8 0.05 0.005" rgba="0 0 0 0"/>
            <geom class="vis" type="ellipsoid" pos="0 0.04 {_f(-(SOLE_DEPTH - 0.024))}" size="0.040 0.100 0.024" rgba="{SKIN}"/>
            <geom class="vis" type="capsule" fromto="0 0 0 0 0.01 {_f(-(SOLE_DEPTH - 0.03))}" size="0.026" rgba="{SKIN}"/>
            <!-- 발바닥 중심. z축 = 발바닥이 향하는 방향(아래) -->
            <site name="{side}_sole" pos="0 0.035 {_f(-SOLE_DEPTH)}" zaxis="0 0 -1" size="0.015"/>
          </body>
        </body>
      </body>'''


def _climber(p: BodyProfile) -> str:
    """몸 전체 조립. 골반에 freejoint(자유 관절, 모터 없음)가 있어 몸이 공중에서 자유롭게 움직인다.
    허리 관절 3개는 골반-허리 사이 한 점에 겹쳐 있고, 가슴은 허리에 고정되어 따라 움직인다."""
    mass = lambda seg: MASS_FRACTION[seg] * p.total_mass
    ts = p.torso_length / 0.55
    pelvis_h, abdomen_h, chest_h = 0.14 * ts, 0.16 * ts, 0.25 * ts
    hip_x, shoulder_x = p.hip_width / 2, p.shoulder_width / 2
    # 몸통 세로 배치 (골반 원점 기준, m)
    pelvis_half = 0.085 * ts                 # 골반 타원체 세로 반지름
    pelvis_top = -0.01 + pelvis_half         # 골반 윗면 높이
    waist_z = pelvis_h * 0.5                 # 허리 관절 위치 (골반 윗부분 안쪽)
    abd_off = abdomen_h * 0.45               # 허리 관절 → 허리 마디 중심
    abd_half = waist_z + abd_off - (pelvis_top - 0.01)  # 허리 마디 아랫면이 골반 윗면과 1cm 겹치도록
    arms = "".join(_arm(s, p, mass) for s in SIDES)
    legs = "".join(_leg(s, p, mass) for s in SIDES)
    return f'''
    <body name="pelvis" pos="0 -0.46 1.25" childclass="climber">
      <freejoint name="root"/>
      {_seg("ellipsoid", f'pos="0 0 -0.01" size="{_f(hip_x * 0.78)} 0.085 {_f(pelvis_half)}"', mass("pelvis"))}
      <!-- 허리 관절은 골반 윗부분(요추 아래쪽)에 있다. 허리 마디는 그 위에 얹혀 있어서
           숙이거나 비틀어도 골반과 붙은 채로 꺾인다 -->
      <body name="abdomen" pos="0 0 {_f(waist_z)}">
        {_joint("waist_yaw", p)}{_joint("waist_pitch", p)}{_joint("waist_roll", p)}
        {_seg("ellipsoid", f'pos="0 0 {_f(abd_off)}" size="0.085 0.068 {_f(abd_half)}"', mass("abdomen"))}
        <body name="chest" pos="0 0 {_f(abd_off + abdomen_h * 0.5 + chest_h * 0.45)}">
          {_seg("ellipsoid", f'size="{_f(shoulder_x * 0.6)} 0.085 {_f(0.15 * ts)}"', mass("chest"))}
          {_seg("ellipsoid", f'pos="0 0 {_f(0.075 * ts)}" size="{_f(shoulder_x * 0.81)} 0.075 0.060"', mass("yoke"))}
          <body name="head" pos="0 0 {_f(chest_h * 0.52)}">
            {_ball("0 0 0.012", 0.022)}
            {_seg("sphere", 'pos="0 0.005 0.128" size="0.100"', mass("head"))}
          </body>{arms}
        </body>
      </body>{legs}
    </body>'''


def _wall(w: Wall) -> str:
    """벽(큰 상자 1개)과 홀드(공)들. 홀드 표면 바깥쪽에 '잡는 점' site를 하나씩 둔다."""
    out = [f'<geom name="wall" type="box" pos="0 {_f(w.thickness / 2)} {_f(w.height / 2)}" '
           f'size="{_f(w.width / 2)} {_f(w.thickness / 2)} {_f(w.height / 2)}" '
           f'friction="{_f(w.friction)} 0.05 0.005" material="wallmat"/>']
    for h in w.holds:
        n = escape(h.name)
        # hold centre sits slightly proud of the wall; site marks the outward grasp point
        out.append(
            f'<body name="hold_{n}" pos="{_f(h.x)} {_f(-0.02)} {_f(h.z)}">'
            f'<geom name="hold_{n}" type="sphere" size="{_f(h.radius)}" friction="{_f(h.friction)} 0.05 0.005" '
            f'material="holdmat" rgba="{HOLD_RGBA}"/>'
            f'<site name="hold_{n}_site" pos="0 {_f(-(h.radius + 0.006))} 0" size="0.016" rgba="0 0 0 0"/>'
            f'</body>')
    return "\n    ".join(out)


def _hold_info(w: Wall) -> str:
    """홀드별 '버틸 수 있는 방향'을 모델 안에 저장 (contact.py가 읽는다): x y z 허용각도"""
    rows = []
    for h in w.holds:
        dx, dy, dz = h.pull_dir
        rows.append(f'<numeric name="hold_{escape(h.name)}_pull" data="{_f(dx)} {_f(dy)} {_f(dz)} {_f(h.pull_cone_deg)}"/>')
    return "\n    ".join(rows)


def _actuators(p: BodyProfile) -> str:
    """관절마다 토크 모터 1개. 실제 토크 = ctrl(-1~1) × gear(최대토크 × strength_scale)."""
    rows = []
    for name in actuated_joints():
        gear = _joint_spec(name)[4] * p.strength_scale
        rows.append(f'<motor name="act_{name}" joint="{name}" ctrlrange="-1 1" gear="{_f(gear)}"/>')
    return "\n    ".join(rows)


def _grasps(w: Wall) -> str:
    """One inactive hand-to-hold point link per (hand, hold); contact.py places and activates it.

    (손 2개 × 홀드 수)만큼 '손-홀드 고정점' 제약을 꺼진 상태로 만든다.
    잡는 위치(앵커)는 고정이 아니라, contact.py가 실제로 닿은 접촉점으로 그때그때 설정한 뒤 켠다.
    그래서 손이 홀드를 파고들지 않는다. 직접 켜지 말고 contact.ContactModel을 쓸 것.
    """
    rows = []
    for s in SIDES:
        for h in w.holds:
            rows.append(f'<connect name="grasp_{s}_{h.name}" body1="{s}_hand" body2="hold_{h.name}" anchor="0 0 0" '
                        f'active="false" solref="0.004 1" solimp="0.99 0.99 0.001"/>')
    return "\n    ".join(rows)


def build_xml(profile: BodyProfile = BASELINE, wall: Wall | None = None) -> str:
    """전체 MJCF XML 문자열을 만든다: 시뮬레이션 설정(0.002초 간격, 중력) + 조명/재질 + 벽 + 몸 + 모터 + 그립.
    충돌 규칙: 몸(contype 2, conaffinity 1) vs 세상(1, 1) → 몸-세상은 충돌, 몸-몸은 충돌 안 함."""
    wall = wall or default_wall()
    return f'''<mujoco model="mannequin_climber_{escape(profile.name)}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <custom>
    {_hold_info(wall)}
  </custom>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096" numslices="40" numstacks="24"/>
    <headlight ambient="0.38 0.38 0.38" diffuse="0.45 0.45 0.45" specular="0.08 0.08 0.08"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.30 0.30 0.31" rgb2="0.20 0.20 0.21" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.24 0.24 0.25" rgb2="0.24 0.24 0.25" mark="edge" markrgb="0.33 0.33 0.34" width="512" height="512"/>
    <material name="floormat" texture="grid" texrepeat="10 10"/>
    <texture name="walltex" type="2d" builtin="checker" rgb1="0.42 0.42 0.44" rgb2="0.42 0.42 0.44" mark="edge" markrgb="0.48 0.48 0.50" width="512" height="512"/>
    <material name="wallmat" texture="walltex" texrepeat="5 5"/>
    <material name="wood" specular="0.25" shininess="0.35"/>
    <material name="holdmat" specular="0.25" shininess="0.35"/>
  </asset>
  <default>
    <geom contype="1" conaffinity="1" solref="0.01 1"/>
    <default class="climber">
      <joint type="hinge" limited="true" armature="0.01"/>
      <site rgba="0 0 0 0"/>
      <!-- climber geoms touch the world but not each other (contype 2 vs conaffinity 1) -->
      <default class="seg">
        <geom contype="2" conaffinity="1" material="wood" rgba="{SKIN}"/>
      </default>
      <default class="vis">
        <geom contype="0" conaffinity="0" mass="0" material="wood"/>
      </default>
    </default>
  </default>
  <worldbody>
    <light pos="-1.5 -3.5 4" dir="0.3 0.6 -0.75" diffuse="0.55 0.55 0.55" castshadow="true"/>
    <geom name="floor" type="plane" size="4 4 0.1" friction="1.0 0.05 0.005" material="floormat"/>
    {_wall(wall)}
    <camera name="back" pos="0 -3.2 1.5" xyaxes="1 0 0 0 0.2 1" mode="fixed"/>
    {_climber(profile)}
  </worldbody>
  <actuator>
    {_actuators(profile)}
  </actuator>
  <equality>
    {_grasps(wall)}
  </equality>
</mujoco>
'''


def build_model(profile: BodyProfile = BASELINE, wall: Wall | None = None) -> mujoco.MjModel:
    """build_xml()로 만든 XML을 바로 MuJoCo 모델 객체로 바꿔 준다. 보통은 이것만 쓰면 된다."""
    return mujoco.MjModel.from_xml_string(build_xml(profile, wall))
