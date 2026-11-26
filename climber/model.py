"""MJCF generator for the mannequin climber.

The look (light-orange wooden drawing mannequin) and the physics body are the same
primitives: segment shapes carry mass and collide with the world; joint balls are
visual only. No meshes, so the model stays cheap to simulate.

Joint layout, ranges and torque ceilings follow the first physics prototype
(free root + 25 hinges): waist 3, each arm 5, each leg 6.
"""
from __future__ import annotations

from xml.sax.saxutils import escape

import mujoco

from .profile import BASELINE, BodyProfile
from .scene import Wall, default_wall

SKIN = "0.98 0.74 0.50 1"
JOINT_BALL = "0.94 0.65 0.41 1"
HOLD_RGBA = "0.30 0.62 0.68 1"
SIDES = ("left", "right")

# name: (axis, lo_deg, hi_deg, damping, max_torque_Nm)
JOINTS = {
    "waist_yaw": ("0 0 1", -35, 35, 4.0, 90),
    "waist_pitch": ("1 0 0", -25, 45, 4.0, 110),
    "waist_roll": ("0 1 0", -25, 25, 4.0, 90),
    "shoulder_pitch": ("1 0 0", -120, 160, 2.0, 85),
    "shoulder_roll": ("0 1 0", -80, 80, 2.0, 85),
    "shoulder_yaw": ("0 0 1", -80, 80, 2.0, 75),
    "elbow": ("1 0 0", 0, 145, 1.5, 65),
    "wrist": ("1 0 0", -60, 60, 1.0, 35),
    "hip_pitch": ("1 0 0", -110, 120, 3.0, 140),
    "hip_yaw": ("0 0 1", -45, 45, 3.0, 80),
    "knee": ("-1 0 0", 0, 150, 2.5, 130),
    "ankle_pitch": ("1 0 0", -45, 35, 2.0, 55),
    "ankle_roll": ("0 1 0", -25, 25, 2.0, 40),
}
# hip roll is asymmetric so each leg abducts outward more than it crosses inward
HIP_ROLL = {"left": ("0 1 0", -50, 60, 3.0, 110), "right": ("0 1 0", -60, 50, 3.0, 110)}

# Fraction of total body mass per segment, de Leva (1996) adult male.
# Trunk = upper (chest, split rib cage / shoulder yoke) + middle (abdomen) + lower (pelvis).
# Limb entries are per side. Sums to 1.0.
MASS_FRACTION = {"head": 0.0694, "chest": 0.1246, "yoke": 0.0350, "abdomen": 0.1633, "pelvis": 0.1117,
                 "upper_arm": 0.0271, "forearm": 0.0162, "hand": 0.0061,
                 "thigh": 0.1416, "shin": 0.0433, "foot": 0.0137}
SOLE_DEPTH = 0.06  # ankle joint height above the sole, ~0.035 H


def _f(v: float) -> str:
    return f"{v:.5g}"


def actuated_joints() -> list[str]:
    names = ["waist_yaw", "waist_pitch", "waist_roll"]
    for s in SIDES:
        names += [f"{s}_{j}" for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist")]
    for s in SIDES:
        names += [f"{s}_{j}" for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    return names


def _joint_spec(name: str):
    side, _, base = name.partition("_")
    if base == "hip_roll":
        return HIP_ROLL[side]
    return JOINTS[base if side in SIDES else name]


def _joint(name: str, p: BodyProfile) -> str:
    axis, lo, hi, damping, _ = _joint_spec(name)
    lo, hi = lo * p.rom_scale, hi * p.rom_scale
    return f'<joint name="{name}" axis="{axis}" range="{_f(lo)} {_f(hi)}" damping="{_f(damping)}"/>'


def _seg(kind: str, attrs: str, mass: float | None = None) -> str:
    """A body segment: collides with the world and carries mass."""
    return f'<geom class="seg" type="{kind}" {attrs} mass="{_f(mass)}"/>'


def _ball(pos: str, r: float) -> str:
    """A visual-only joint ball."""
    return f'<geom class="vis" type="sphere" pos="{pos}" size="{_f(r)}" rgba="{JOINT_BALL}"/>'


def _arm(side: str, p: BodyProfile, mass) -> str:
    sx = p.shoulder_width / 2 * (-1 if side == "left" else 1)
    chest_h = 0.25 * p.torso_length / 0.55
    ua, fa = p.upper_arm_length, p.forearm_length
    j = lambda n: _joint(f"{side}_{n}", p)
    return f'''
          <body name="{side}_upper_arm" pos="{_f(sx)} 0 {_f(chest_h * 0.32)}">
            {j("shoulder_pitch")}{j("shoulder_roll")}{j("shoulder_yaw")}
            {_ball("0 0 0", 0.045)}
            {_seg("capsule", f'fromto="0 0 -0.05 0 0 {_f(-(ua - 0.05))}" size="0.036"', mass("upper_arm"))}
            <body name="{side}_forearm" pos="0 0 {_f(-ua)}">
              {j("elbow")}
              {_ball("0 0 0", 0.033)}
              {_seg("capsule", f'fromto="0 0 -0.04 0 0 {_f(-(fa - 0.04))}" size="0.030"', mass("forearm"))}
              <body name="{side}_hand" pos="0 0 {_f(-fa)}">
                {j("wrist")}
                {_ball("0 0 0", 0.024)}
                {_seg("ellipsoid", 'pos="0 0.004 -0.058" size="0.020 0.034 0.046"', mass("hand"))}
                <site name="{side}_hand_site" pos="0 0 -0.06" size="0.02"/>
              </body>
            </body>
          </body>'''


def _leg(side: str, p: BodyProfile, mass) -> str:
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
            <site name="{side}_foot_site" pos="0 0.09 {_f(-SOLE_DEPTH)}" size="0.02"/>
          </body>
        </body>
      </body>'''


def _climber(p: BodyProfile) -> str:
    mass = lambda seg: MASS_FRACTION[seg] * p.total_mass
    ts = p.torso_length / 0.55
    pelvis_h, abdomen_h, chest_h = 0.14 * ts, 0.16 * ts, 0.25 * ts
    hip_x, shoulder_x = p.hip_width / 2, p.shoulder_width / 2
    arms = "".join(_arm(s, p, mass) for s in SIDES)
    legs = "".join(_leg(s, p, mass) for s in SIDES)
    return f'''
    <body name="pelvis" pos="0 -0.46 1.25" childclass="climber">
      <freejoint name="root"/>
      {_seg("ellipsoid", f'pos="0 0 -0.01" size="{_f(hip_x * 0.78)} 0.085 0.085"', mass("pelvis"))}
      <body name="abdomen" pos="0 0 {_f(pelvis_h * 0.5 + abdomen_h * 0.45)}">
        {_joint("waist_yaw", p)}{_joint("waist_pitch", p)}{_joint("waist_roll", p)}
        {_seg("ellipsoid", 'size="0.085 0.068 0.062"', mass("abdomen"))}
        <body name="chest" pos="0 0 {_f(abdomen_h * 0.5 + chest_h * 0.45)}">
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


def _actuators(p: BodyProfile) -> str:
    rows = []
    for name in actuated_joints():
        gear = _joint_spec(name)[4] * p.strength_scale
        rows.append(f'<motor name="act_{name}" joint="{name}" ctrlrange="-1 1" gear="{_f(gear)}"/>')
    return "\n    ".join(rows)


def _grasps(w: Wall) -> str:
    """One inactive hand-to-hold weld point per (hand, hold); activate to 'grip'."""
    rows = []
    for s in SIDES:
        for h in w.holds:
            rows.append(f'<connect name="grasp_{s}_{h.name}" site1="{s}_hand_site" site2="hold_{h.name}_site" '
                        f'active="false" solref="0.004 1" solimp="0.99 0.99 0.001"/>')
    return "\n    ".join(rows)


def build_xml(profile: BodyProfile = BASELINE, wall: Wall | None = None) -> str:
    wall = wall or default_wall()
    return f'''<mujoco model="mannequin_climber_{escape(profile.name)}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
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
    return mujoco.MjModel.from_xml_string(build_xml(profile, wall))
