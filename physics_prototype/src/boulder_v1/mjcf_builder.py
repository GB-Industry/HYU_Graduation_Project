from __future__ import annotations

from html import escape
import math

from .contact_geometry import ContactMode, Frame, SHOE_FRICTION, canonical_geometry
from .schema import Affordance, BoulderScene, ClimberProfile, Limb, SourceType

END_EFFECTOR_SITES: dict[Limb, str] = {
    Limb.LEFT_HAND: "left_hand_site",
    Limb.RIGHT_HAND: "right_hand_site",
    Limb.LEFT_FOOT: "left_foot_site",
    Limb.RIGHT_FOOT: "right_foot_site",
}
END_EFFECTOR_SITE_NAMES: tuple[str, ...] = tuple(END_EFFECTOR_SITES.values())


def get_grasp_equality_name(limb: Limb, region_id: str) -> str:
    return f"grasp_{limb.value.lower()}_{region_id}"


def _f(v: float) -> str:
    if not math.isfinite(v):
        raise ValueError("Nonfinite derived MJCF parameter")
    return repr(float(v)) if v else "0"


def _frame_attributes(frame: Frame, parent: Frame | None = None) -> str:
    position, rotation = frame.position, frame.rotation
    if parent is not None:
        delta = tuple(position[i] - parent.position[i] for i in range(3))
        position = tuple(sum(parent.rotation[j][i] * delta[j] for j in range(3)) for i in range(3))
        rotation = tuple(tuple(sum(parent.rotation[k][i] * rotation[k][j] for k in range(3))
                               for j in range(3)) for i in range(3))
    xyaxes = tuple(rotation[i][j] for j in (0, 1) for i in range(3))
    return f'pos="{" ".join(_f(v) for v in position)}" xyaxes="{" ".join(_f(v) for v in xyaxes)}"'


def _range(lo: float, hi: float, scale: float = 1.0, cap: float | None = None) -> str:
    r_lo, r_hi = lo * scale, hi * scale
    if cap is not None:
        r_lo, r_hi = max(-cap, r_lo), min(cap, r_hi)
    if not math.isfinite(r_lo) or not math.isfinite(r_hi) or not r_lo < r_hi:
        raise ValueError("ROM scaling produces invalid joint limits")
    return f"{_f(r_lo)} {_f(r_hi)}"


def _validate_inertia(mass: float, inertia: tuple[float, float, float]) -> None:
    if any(not math.isfinite(v) or v <= 1e-15 for v in (mass, *inertia)):
        raise ValueError("Segment mass/inertia is nonfinite or below MuJoCo's numerical minimum")


def _ellipsoid_inertial(mass: float, a: float, b: float, c: float) -> str:
    inertia = (mass * (b * b + c * c) / 5,
               mass * (a * a + c * c) / 5,
               mass * (a * a + b * b) / 5)
    _validate_inertia(mass, inertia)
    return f'<inertial pos="0 0 0" mass="{_f(mass)}" diaginertia="{" ".join(_f(v) for v in inertia)}"/>'


def build_mjcf(
    scene: BoulderScene,
    profile: ClimberProfile,
    contact_mode: ContactMode = ContactMode.PHYSICAL,
) -> str:
    """Generate the simplified free-root, 25-hinge physical-model proxy.

    Technical coordinate ranges are not clinical anatomical envelopes:
      - Torso / Spine: 3 DoF (waist yaw, pitch, roll)
      - Arms: 5 DoF x 2 (shoulder pitch, roll, yaw; elbow pitch; wrist pitch)
      - Legs: 6 DoF x 2 (hip pitch, roll, yaw; knee pitch; ankle pitch, roll)
    """
    if not isinstance(contact_mode, ContactMode):
        raise ValueError("contact_mode must be a ContactMode")
    debug = contact_mode == ContactMode.IDEALIZED_DEBUG
    p = profile
    if scene.scale != 1.0:
        raise ValueError("This MJCF backend supports scene.scale == 1 only (coordinates are metres)")
    if not 1e-6 <= p.rom_scale <= 1.2:
        raise ValueError("Supported prototype rom_scale is [1e-6, 1.2], not a validated human ROM envelope")
    # Conservative numerical support domain, not a human-population model.
    if any(not .05 <= v <= 2.0 for v in (p.torso_length, p.shoulder_width, p.hip_width,
                                        p.upper_arm_length, p.forearm_length, p.thigh_length, p.shin_length)):
        raise ValueError("Supported humanoid dimensions are [0.05, 2.0] metres")
    if not .1 <= p.mass_scale <= 10:
        raise ValueError("Supported mass_scale is [0.1, 10]")
    if not 1e-6 <= p.strength_scale <= 10:
        raise ValueError("Supported strength_scale is [1e-6, 10]")
    for wall in scene.walls:
        if wall.normal != (0.0, -1.0, 0.0):
            raise ValueError(f"Wall {wall.id!r}: only axis-aligned normal (0, -1, 0) is supported")
        if max(abs(v) for v in wall.center) > 1e3 or any(not 1e-6 <= v <= 100 for v in wall.size):
            raise ValueError(f"Wall {wall.id!r}: coordinates/sizes exceed the supported numerical domain")
        if not 1e-6 <= wall.friction <= 10:
            raise ValueError(f"Wall {wall.id!r}: supported friction is [1e-6, 10]")
    for region in scene.contact_regions:
        if region.source_type not in (SourceType.HOLD, SourceType.WALL):
            raise ValueError(f"Region {region.id!r}: VOLUME/EDGE geometry is not implemented")
        if region.source_type == SourceType.WALL and not region.affordances <= {Affordance.PRESS, Affordance.SMEAR}:
            raise ValueError(f"Wall region {region.id!r}: GRASP/STEP attachment geometry is not implemented")
        if region.source_type == SourceType.WALL and region.half_size is not None:
            raise ValueError(f"Wall region {region.id!r}: box geometry is not implemented for metadata-only patches")
        if max(abs(v) for v in region.position) > 1e3 or not 1e-6 <= region.radius <= 100:
            raise ValueError(f"Region {region.id!r}: coordinates/radius exceed the supported numerical domain")
        if region.half_size is not None and any(not 1e-6 <= v <= 100 for v in region.half_size):
            raise ValueError(f"Region {region.id!r}: box half_size exceeds the supported numerical domain")
        if not 1e-6 <= region.friction <= 10:
            raise ValueError(f"Region {region.id!r}: supported friction is [1e-6, 10]")
    for limb, hold_id in scene.start_configuration.items():
        region = scene.region(hold_id)
        mode = Affordance.GRASP if limb.is_hand else Affordance.STEP
        if region.source_type != SourceType.HOLD or mode not in region.affordances:
            raise ValueError(f"Start {limb.value}:{hold_id} cannot be represented as an attachment by this backend")

    wall_xml = []
    for wall in scene.walls:
        x, y, z = wall.center
        sx, sy, sz = wall.size
        wall_xml.append(
            f'<body name="{escape(wall.id)}" pos="{_f(x)} {_f(y)} {_f(z)}">'
            f'<geom name="{escape(wall.id)}_geom" type="box" size="{_f(sx)} {_f(sy)} {_f(sz)}" '
            f'friction="{_f(wall.friction)} 0.05 0.005" rgba="0.65 0.65 0.68 1"/>'
            f'</body>'
        )

    contact_xml = []
    for region in scene.contact_regions:
        if region.source_type == SourceType.WALL:
            continue
        geometry = canonical_geometry(region)
        step_site_xml = ""
        if Affordance.STEP in region.affordances:
            step_site_xml = (
                f'<site name="site_step_{escape(region.id)}" '
                f'{_frame_attributes(geometry.foot_frame, geometry.body_frame)} '
                f'size="0.016" rgba="0.2 0.8 0.4 0.9"/>'
            )
        contact_xml.append(
            f'<body name="contact_{escape(region.id)}" {_frame_attributes(geometry.body_frame)}>'
            f'<geom name="geom_{escape(region.id)}" type="{geometry.shape}" size="{" ".join(_f(v) for v in geometry.size)}" '
            f'friction="{_f(region.friction)} 0.05 0.005" rgba="0.25 0.45 0.8 1"/>'
            f'<site name="site_{escape(region.id)}" {_frame_attributes(geometry.hand_frame, geometry.body_frame)} '
            f'size="0.016" rgba="1.0 0.82 0.1 0.9"/>'
            f'{step_site_xml}'
            f'</body>'
        )

    # Anthropometric dimensions derived from ClimberProfile
    shoulder_x = p.shoulder_width / 2.0
    hip_x = p.hip_width / 2.0
    ua, fa = p.upper_arm_length, p.forearm_length
    th, sh = p.thigh_length, p.shin_length
    m_scale = p.mass_scale
    s_scale = p.strength_scale
    rom = p.rom_scale
    if not math.isfinite(78.3 * m_scale * 9.81):
        raise ValueError("Climber mass produces a nonfinite gravitational force")
    # Check inferred limb proxies too; finite inputs can overflow or underflow
    # primitive inertia calculations before MuJoCo sees their XML.
    for length, radius, mass in ((ua, .038, 2.4 * m_scale), (fa, .031, 1.5 * m_scale),
                                 (th, .052, 7.5 * m_scale), (sh, .038, 3.5 * m_scale),
                                 (.055, .036, 1.5 * m_scale)):
        cylinder_mass = mass * length / (length + 4 * radius / 3)
        sphere_mass = mass - cylinder_mass
        axial = cylinder_mass * radius * radius / 2 + sphere_mass * 2 * radius * radius / 5
        transverse = (cylinder_mass * (3 * radius * radius + length * length) / 12
                      + sphere_mass * (2 * radius * radius / 5 + length * length / 4 + 3 * length * radius / 8))
        _validate_inertia(mass, (transverse, transverse, axial))
    for mass, a, b, c in ((.55 * m_scale, .026, .038, .016), (1.2 * m_scale, .040, .095, .018)):
        _validate_inertia(mass, (mass * (b*b + c*c) / 3,
                                mass * (a*a + c*c) / 3, mass * (a*a + b*b) / 3))

    # Torso split: pelvis, abdomen, chest based on p.torso_length (default 0.55)
    torso_scale = p.torso_length / 0.55
    pelvis_h = 0.14 * torso_scale
    abdomen_h = 0.16 * torso_scale
    chest_h = 0.25 * torso_scale
    # Fixed segment mass fractions; morphology deforms the inertia proxy, not mass.
    # The former 1 kg root contribution is explicitly folded into the rigid pelvis.
    pelvis_inertial = _ellipsoid_inertial(10.0 * m_scale, hip_x * .90, .095, pelvis_h * .48)
    abdomen_inertial = _ellipsoid_inertial(12.0 * m_scale, hip_x * .82, .088, abdomen_h * .48)
    chest_inertial = _ellipsoid_inertial(18.0 * m_scale, shoulder_x * .88, .115, chest_h * .48)

    # 25-DoF Humanoid Character XML
    # Existing visual styling is unchanged; collision envelopes below are separate.
    c_pants = "0.20 0.24 0.34 1"        # Dark slate athletic climbing pants
    c_pants_accent = "0.25 0.30 0.42 1" # Contour shading on quadriceps/calves
    c_shirt = "0.85 0.36 0.20 1"        # Vibrant rust/coral athletic climbing shirt
    c_shirt_accent = "0.75 0.30 0.16 1" # Latissimus dorsi back contours
    c_skin = "0.86 0.72 0.60 1"         # Warm natural athletic skin tone
    c_hair = "0.18 0.15 0.14 1"         # Dark hair / climbing beanie cap
    c_shoe_rubber = "0.10 0.10 0.12 1"  # High-friction Vibram rubber sole & rand
    c_shoe_upper = "0.15 0.75 0.65 1"   # Vibrant cyan climbing shoe upper

    character_xml = f'''
    <body name="climber_root" pos="0 -0.46 1.25">
      <freejoint name="root"/>
      <body name="pelvis" pos="0 0 0">
        {pelvis_inertial}
        <!-- Anatomical athletic pelvis in climbing pants -->
        <geom name="pelvis_geom" type="ellipsoid" size="{_f(hip_x * 0.90)} 0.095 {_f(pelvis_h * 0.48)}" pos="0 0 0" mass="{_f(10.0 * m_scale)}" rgba="{c_pants}"/>
        <geom name="pelvis_hip_l" type="sphere" size="0.054" pos="{_f(-hip_x * 0.85)} 0 -0.02" mass="0" rgba="{c_pants}"/>
        <geom name="pelvis_hip_r" type="sphere" size="0.054" pos="{_f(hip_x * 0.85)} 0 -0.02" mass="0" rgba="{c_pants}"/>
        <!-- Gluteal / seat anatomical contours -->
        <geom name="glute_l" type="ellipsoid" size="{_f(hip_x * 0.40)} 0.065 0.055" pos="{_f(-hip_x * 0.45)} -0.04 -0.015" contype="0" conaffinity="0" mass="0" rgba="{c_pants}"/>
        <geom name="glute_r" type="ellipsoid" size="{_f(hip_x * 0.40)} 0.065 0.055" pos="{_f(hip_x * 0.45)} -0.04 -0.015" contype="0" conaffinity="0" mass="0" rgba="{c_pants}"/>
        <!-- Continuous visual waist bridge connecting pelvis seamlessly upward -->
        <geom name="pelvis_waist_bridge" type="capsule" fromto="0 -0.008 0 0 -0.008 {_f(pelvis_h * 0.52)}" size="{_f(hip_x * 0.78)}" contype="0" conaffinity="0" mass="0" rgba="{c_pants}"/>

        <!-- Spine / Torso chain -->
        <body name="abdomen" pos="0 0 {_f(pelvis_h * 0.5 + abdomen_h * 0.45)}">
          {abdomen_inertial}
          <joint name="waist_yaw" type="hinge" axis="0 0 1" range="{_range(-35, 35, rom)}" damping="4"/>
          <joint name="waist_pitch" type="hinge" axis="1 0 0" range="{_range(-25, 45, rom)}" damping="4"/>
          <joint name="waist_roll" type="hinge" axis="0 1 0" range="{_range(-25, 25, rom)}" damping="4"/>
          <!-- Seamless athletic waist connecting pelvis to chest -->
          <geom name="abdomen_geom" type="ellipsoid" size="{_f(hip_x * 0.82)} 0.088 {_f(abdomen_h * 0.48)}" pos="0 0 0" mass="{_f(12.0 * m_scale)}" rgba="{c_shirt}"/>
          <geom name="abdomen_bridge" type="capsule" fromto="0 -0.005 {_f(-abdomen_h * 0.48)} 0 -0.005 {_f(abdomen_h * 0.48)}" size="{_f(hip_x * 0.76)}" contype="0" conaffinity="0" mass="0" rgba="{c_shirt}"/>

          <body name="chest" pos="0 0 {_f(abdomen_h * 0.5 + chest_h * 0.45)}">
            {chest_inertial}
            <!-- Broad athletic V-tapered ribcage -->
            <geom name="chest_geom" type="ellipsoid" size="{_f(shoulder_x * 0.88)} 0.115 {_f(chest_h * 0.48)}" pos="0 0 0" mass="{_f(18.0 * m_scale)}" rgba="{c_shirt}"/>
            <geom name="chest_bridge" type="capsule" fromto="0 -0.005 {_f(-chest_h * 0.45)} 0 -0.005 {_f(chest_h * 0.28)}" size="{_f(shoulder_x * 0.76)}" contype="0" conaffinity="0" mass="0" rgba="{c_shirt}"/>
            <!-- Latissimus dorsi back flares (climber V-taper) -->
            <geom name="lat_l" type="capsule" fromto="{_f(-shoulder_x * 0.82)} -0.02 {_f(chest_h * 0.20)} {_f(-hip_x * 0.72)} -0.015 {_f(-chest_h * 0.40)}" size="0.040" contype="0" conaffinity="0" mass="0" rgba="{c_shirt_accent}"/>
            <geom name="lat_r" type="capsule" fromto="{_f(shoulder_x * 0.82)} -0.02 {_f(chest_h * 0.20)} {_f(hip_x * 0.72)} -0.015 {_f(-chest_h * 0.40)}" size="0.040" contype="0" conaffinity="0" mass="0" rgba="{c_shirt_accent}"/>
            <!-- Continuous shoulder girdle / clavicle spanning across upper chest -->
            <geom name="shoulder_girdle" type="capsule" fromto="{_f(-shoulder_x)} 0 {_f(chest_h * 0.32)} {_f(shoulder_x)} 0 {_f(chest_h * 0.32)}" size="0.046" contype="0" conaffinity="0" mass="0" rgba="{c_shirt}"/>
            <geom name="deltoid_l" type="sphere" size="0.054" pos="{_f(-shoulder_x)} 0 {_f(chest_h * 0.32)}" mass="0" rgba="{c_shirt}"/>
            <geom name="deltoid_r" type="sphere" size="0.054" pos="{_f(shoulder_x)} 0 {_f(chest_h * 0.32)}" mass="0" rgba="{c_shirt}"/>

            <!-- Neck and Head with Hair & Facial Contour -->
            <body name="head" pos="0 0 {_f(chest_h * 0.52)}">
              <!-- Visual trapezius slope / neck base -->
              <geom name="neck_base" type="capsule" fromto="0 -0.008 -0.015 0 -0.008 0.040" size="0.036" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
              <geom name="neck_geom" type="capsule" fromto="0 0 0 0 0 0.055" size="0.036" mass="{_f(1.5 * m_scale)}" rgba="{c_skin}"/>
              <geom name="head_geom" type="ellipsoid" pos="0 0.01 0.135" size="0.075 0.092 0.105" mass="{_f(3.5 * m_scale)}" rgba="{c_skin}"/>
              <!-- Cranial hair / beanie cap -->
              <geom name="head_hair" type="ellipsoid" pos="0 0.005 0.155" size="0.076 0.090 0.082" contype="0" conaffinity="0" mass="0" rgba="{c_hair}"/>
              <!-- Jawline contour -->
              <geom name="head_jaw" type="capsule" fromto="0 0.025 0.080 0 0.045 0.115" size="0.042" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
            </body>

            <!-- Left Arm -->
            <body name="left_upper_arm" pos="{_f(-shoulder_x)} 0 {_f(chest_h * 0.32)}">
              <joint name="left_shoulder_pitch" type="hinge" axis="1 0 0" range="{_range(-120, 160, rom)}" damping="2"/>
              <joint name="left_shoulder_roll" type="hinge" axis="0 1 0" range="{_range(-80, 80, rom, cap=85)}" damping="2"/>
              <joint name="left_shoulder_yaw" type="hinge" axis="0 0 1" range="{_range(-80, 80, rom)}" damping="2"/>
              <!-- Shoulder joint cap overlapping into torso -->
              <geom name="left_shoulder_cap" type="sphere" size="0.046" pos="0 0 0" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
              <geom name="left_upper_arm_geom" type="capsule" fromto="0 0 0 0 0 {_f(-ua)}" size="0.038" mass="{_f(2.4 * m_scale)}" rgba="{c_skin}"/>
              <!-- Biceps/triceps anatomical contour -->
              <geom name="left_biceps" type="capsule" fromto="0 0.006 {_f(-ua * 0.25)} 0 0.006 {_f(-ua * 0.72)}" size="0.042" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>

              <body name="left_forearm" pos="0 0 {_f(-ua)}">
                <joint name="left_elbow" type="hinge" axis="1 0 0" range="{_range(0, 145, rom)}" damping="1.5"/>
                <!-- Elbow joint cap -->
                <geom name="left_elbow_cap" type="sphere" size="0.035" pos="0 0 0" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
                <geom name="left_forearm_geom" type="capsule" fromto="0 0 0 0 0 {_f(-fa)}" size="0.031" mass="{_f(1.5 * m_scale)}" contype="0" conaffinity="0" rgba="{c_skin}"/>
                <geom name="left_forearm_collider" type="capsule" fromto="0 0 0 0 0 {_f(-fa)}" size="0.028" mass="0" rgba="0 0 0 0"/>
                <!-- Climber forearm muscle belly (flexor/extensor mass tapering to wrist) -->
                <geom name="left_forearm_muscle" type="capsule" fromto="0 0.006 0 0 0.006 {_f(-fa * 0.65)}" size="0.037" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>

                <body name="left_hand" pos="0 0 {_f(-fa)}">
                  <joint name="left_wrist" type="hinge" axis="1 0 0" range="{_range(-60, 60, rom)}" damping="1.0"/>
                  <!-- Wrist cap & anatomical hand -->
                  <geom name="left_wrist_cap" type="sphere" size="0.026" pos="0 0 0" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
                  <geom name="left_hand_geom" type="box" size="0.026 0.038 0.016" pos="0 0 -0.03" mass="{_f(0.55 * m_scale)}" rgba="{c_skin}"/>
                  <!-- Curled functional climbing fingers & thumb -->
                  <geom name="left_fingers_geom" type="box" size="0.022 0.032 0.010" pos="0 0.008 -0.048" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
                  <geom name="left_thumb_geom" type="capsule" fromto="-0.024 0.012 -0.018 -0.026 0.025 -0.038" size="0.009" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
                  <site name="left_hand_site" pos="0 0 -0.04" size="0.024" rgba="0.92 0.22 0.20 0.9" type="sphere"/>
                  <site name="left_grasp_site" pos="0 0 -0.04" size="0.024" rgba="0.92 0.22 0.20 0.9" type="sphere"/>
                </body>
              </body>
            </body>

            <!-- Right Arm -->
            <body name="right_upper_arm" pos="{_f(shoulder_x)} 0 {_f(chest_h * 0.32)}">
              <joint name="right_shoulder_pitch" type="hinge" axis="1 0 0" range="{_range(-120, 160, rom)}" damping="2"/>
              <joint name="right_shoulder_roll" type="hinge" axis="0 1 0" range="{_range(-80, 80, rom, cap=85)}" damping="2"/>
              <joint name="right_shoulder_yaw" type="hinge" axis="0 0 1" range="{_range(-80, 80, rom)}" damping="2"/>
              <!-- Shoulder joint cap overlapping into torso -->
              <geom name="right_shoulder_cap" type="sphere" size="0.046" pos="0 0 0" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
              <geom name="right_upper_arm_geom" type="capsule" fromto="0 0 0 0 0 {_f(-ua)}" size="0.038" mass="{_f(2.4 * m_scale)}" rgba="{c_skin}"/>
              <geom name="right_biceps" type="capsule" fromto="0 0.006 {_f(-ua * 0.25)} 0 0.006 {_f(-ua * 0.72)}" size="0.042" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>

              <body name="right_forearm" pos="0 0 {_f(-ua)}">
                <joint name="right_elbow" type="hinge" axis="1 0 0" range="{_range(0, 145, rom)}" damping="1.5"/>
                <!-- Elbow joint cap -->
                <geom name="right_elbow_cap" type="sphere" size="0.035" pos="0 0 0" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
                <geom name="right_forearm_geom" type="capsule" fromto="0 0 0 0 0 {_f(-fa)}" size="0.031" mass="{_f(1.5 * m_scale)}" contype="0" conaffinity="0" rgba="{c_skin}"/>
                <geom name="right_forearm_collider" type="capsule" fromto="0 0 0 0 0 {_f(-fa)}" size="0.028" mass="0" rgba="0 0 0 0"/>
                <geom name="right_forearm_muscle" type="capsule" fromto="0 0.006 0 0 0.006 {_f(-fa * 0.65)}" size="0.037" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>

                <body name="right_hand" pos="0 0 {_f(-fa)}">
                  <joint name="right_wrist" type="hinge" axis="1 0 0" range="{_range(-60, 60, rom)}" damping="1.0"/>
                  <!-- Wrist cap & anatomical hand -->
                  <geom name="right_wrist_cap" type="sphere" size="0.026" pos="0 0 0" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
                  <geom name="right_hand_geom" type="box" size="0.026 0.038 0.016" pos="0 0 -0.03" mass="{_f(0.55 * m_scale)}" rgba="{c_skin}"/>
                  <geom name="right_fingers_geom" type="box" size="0.022 0.032 0.010" pos="0 0.008 -0.048" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
                  <geom name="right_thumb_geom" type="capsule" fromto="0.024 0.012 -0.018 0.026 0.025 -0.038" size="0.009" contype="0" conaffinity="0" mass="0" rgba="{c_skin}"/>
                  <site name="right_hand_site" pos="0 0 -0.04" size="0.024" rgba="0.20 0.50 0.92 0.9" type="sphere"/>
                  <site name="right_grasp_site" pos="0 0 -0.04" size="0.024" rgba="0.20 0.50 0.92 0.9" type="sphere"/>
                </body>
              </body>
            </body>
          </body>
        </body>

        <!-- Left Leg in Cohesive Climbing Pants -->
        <body name="left_thigh" pos="{_f(-hip_x * 0.85)} 0 {_f(-pelvis_h * 0.35)}">
          <joint name="left_hip_pitch" type="hinge" axis="1 0 0" range="{_range(-110, 120, rom)}" damping="3"/>
          <joint name="left_hip_roll" type="hinge" axis="0 1 0" range="{_range(-50, 60, rom)}" damping="3"/>
          <joint name="left_hip_yaw" type="hinge" axis="0 0 1" range="{_range(-45, 45, rom)}" damping="3"/>
          <!-- Hip cap & thigh in matching climbing pants -->
          <geom name="left_hip_cap" type="sphere" size="0.052" pos="0 0 0" contype="0" conaffinity="0" mass="0" rgba="{c_pants}"/>
          <geom name="left_thigh_geom" type="capsule" fromto="0 0 0 0 0 {_f(-th)}" size="0.052" mass="{_f(7.5 * m_scale)}" rgba="{c_pants}"/>
          <!-- Quadriceps muscle contour -->
          <geom name="left_quad" type="capsule" fromto="0 0.008 {_f(-th * 0.18)} 0 0.008 {_f(-th * 0.75)}" size="0.057" contype="0" conaffinity="0" mass="0" rgba="{c_pants_accent}"/>

          <body name="left_shin" pos="0 0 {_f(-th)}">
            <joint name="left_knee" type="hinge" axis="-1 0 0" range="{_range(0, 150, rom)}" damping="2.5"/>
            <!-- Knee joint cap in matching pants -->
            <geom name="left_knee_cap" type="sphere" size="0.046" pos="0 0.008 0" contype="0" conaffinity="0" mass="0" rgba="{c_pants}"/>
            <geom name="left_shin_geom" type="capsule" fromto="0 0 0 0 0 {_f(-sh)}" size="0.038" mass="{_f(3.5 * m_scale)}" contype="0" conaffinity="0" rgba="{c_pants}"/>
            <geom name="left_shin_collider" type="capsule" fromto="0 0 0 0 0 {_f(-sh + min(0.003, sh * 0.1))}" size="0.038" mass="0" rgba="0 0 0 0"/>
            <!-- Gastrocnemius calf muscle tapering to ankle -->
            <geom name="left_calf" type="capsule" fromto="0 -0.012 {_f(-sh * 0.15)} 0 -0.012 {_f(-sh * 0.65)}" size="0.046" contype="0" conaffinity="0" mass="0" rgba="{c_pants_accent}"/>

            <body name="left_foot" pos="0 0.02 {_f(-sh)}">
              <joint name="left_ankle_pitch" type="hinge" axis="1 0 0" range="{_range(-45, 35, rom)}" damping="2.0"/>
              <joint name="left_ankle_roll" type="hinge" axis="0 1 0" range="{_range(-25, 25, rom)}" damping="2.0"/>
              <!-- Sleek climbing shoe: Dark high-friction sticky rubber sole contacting holds -->
              <geom name="left_foot_geom" type="box" size="0.040 0.095 0.018" pos="0 0.035 -0.018" mass="{_f(1.2 * m_scale)}" friction="1.8 0.05 0.005" rgba="{c_shoe_rubber}"/>
              <!-- Shoe upper (visual): vibrant climbing shoe body -->
              <geom name="left_shoe_upper" type="box" size="0.038 0.090 0.014" pos="0 0.035 0.004" contype="0" conaffinity="0" mass="0" rgba="{c_shoe_upper}"/>
              <!-- Downturned toe & heel tension rand -->
              <geom name="left_shoe_toe" type="ellipsoid" size="0.034 0.038 0.018" pos="0 0.105 -0.014" contype="0" conaffinity="0" mass="0" rgba="{c_shoe_rubber}"/>
              <geom name="left_shoe_heel" type="capsule" fromto="0 -0.02 0.006 0 0.02 0.006" size="0.028" contype="0" conaffinity="0" mass="0" rgba="{c_shoe_rubber}"/>
              <site name="left_foot_site" pos="0 0.09 -0.025" size="0.024" rgba="0.92 0.58 0.12 0.9" type="sphere"/>
            </body>
          </body>
        </body>

        <!-- Right Leg in Cohesive Climbing Pants -->
        <body name="right_thigh" pos="{_f(hip_x * 0.85)} 0 {_f(-pelvis_h * 0.35)}">
          <joint name="right_hip_pitch" type="hinge" axis="1 0 0" range="{_range(-110, 120, rom)}" damping="3"/>
          <joint name="right_hip_roll" type="hinge" axis="0 1 0" range="{_range(-60, 50, rom)}" damping="3"/>
          <joint name="right_hip_yaw" type="hinge" axis="0 0 1" range="{_range(-45, 45, rom)}" damping="3"/>
          <!-- Hip cap & thigh in matching climbing pants -->
          <geom name="right_hip_cap" type="sphere" size="0.052" pos="0 0 0" contype="0" conaffinity="0" mass="0" rgba="{c_pants}"/>
          <geom name="right_thigh_geom" type="capsule" fromto="0 0 0 0 0 {_f(-th)}" size="0.052" mass="{_f(7.5 * m_scale)}" rgba="{c_pants}"/>
          <geom name="right_quad" type="capsule" fromto="0 0.008 {_f(-th * 0.18)} 0 0.008 {_f(-th * 0.75)}" size="0.057" contype="0" conaffinity="0" mass="0" rgba="{c_pants_accent}"/>

          <body name="right_shin" pos="0 0 {_f(-th)}">
            <joint name="right_knee" type="hinge" axis="-1 0 0" range="{_range(0, 150, rom)}" damping="2.5"/>
            <!-- Knee joint cap in matching pants -->
            <geom name="right_knee_cap" type="sphere" size="0.046" pos="0 0.008 0" contype="0" conaffinity="0" mass="0" rgba="{c_pants}"/>
            <geom name="right_shin_geom" type="capsule" fromto="0 0 0 0 0 {_f(-sh)}" size="0.038" mass="{_f(3.5 * m_scale)}" contype="0" conaffinity="0" rgba="{c_pants}"/>
            <geom name="right_shin_collider" type="capsule" fromto="0 0 0 0 0 {_f(-sh + min(0.003, sh * 0.1))}" size="0.038" mass="0" rgba="0 0 0 0"/>
            <geom name="right_calf" type="capsule" fromto="0 -0.012 {_f(-sh * 0.15)} 0 -0.012 {_f(-sh * 0.65)}" size="0.046" contype="0" conaffinity="0" mass="0" rgba="{c_pants_accent}"/>

            <body name="right_foot" pos="0 0.02 {_f(-sh)}">
              <joint name="right_ankle_pitch" type="hinge" axis="1 0 0" range="{_range(-45, 35, rom)}" damping="2.0"/>
              <joint name="right_ankle_roll" type="hinge" axis="0 1 0" range="{_range(-25, 25, rom)}" damping="2.0"/>
              <!-- Sleek climbing shoe: Dark high-friction sticky rubber sole contacting holds -->
              <geom name="right_foot_geom" type="box" size="0.040 0.095 0.018" pos="0 0.035 -0.018" mass="{_f(1.2 * m_scale)}" friction="1.8 0.05 0.005" rgba="{c_shoe_rubber}"/>
              <!-- Shoe upper (visual): vibrant climbing shoe body -->
              <geom name="right_shoe_upper" type="box" size="0.038 0.090 0.014" pos="0 0.035 0.004" contype="0" conaffinity="0" mass="0" rgba="{c_shoe_upper}"/>
              <geom name="right_shoe_toe" type="ellipsoid" size="0.034 0.038 0.018" pos="0 0.105 -0.014" contype="0" conaffinity="0" mass="0" rgba="{c_shoe_rubber}"/>
              <geom name="right_shoe_heel" type="capsule" fromto="0 -0.02 0.006 0 0.02 0.006" size="0.028" contype="0" conaffinity="0" mass="0" rgba="{c_shoe_rubber}"/>
              <site name="right_foot_site" pos="0 0.09 -0.025" size="0.024" rgba="0.18 0.82 0.38 0.9" type="sphere"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    '''

    # Actuators: 25 motors scaled by p.strength_scale
    actuator_specs = [
        # Torso / Waist (3)
        ("waist_yaw", 90.0), ("waist_pitch", 110.0), ("waist_roll", 90.0),
        # Arms (5x2 = 10)
        ("left_shoulder_pitch", 85.0), ("left_shoulder_roll", 85.0), ("left_shoulder_yaw", 75.0),
        ("left_elbow", 65.0), ("left_wrist", 35.0),
        ("right_shoulder_pitch", 85.0), ("right_shoulder_roll", 85.0), ("right_shoulder_yaw", 75.0),
        ("right_elbow", 65.0), ("right_wrist", 35.0),
        # Legs (6x2 = 12)
        ("left_hip_pitch", 140.0), ("left_hip_roll", 110.0), ("left_hip_yaw", 80.0),
        ("left_knee", 130.0), ("left_ankle_pitch", 55.0), ("left_ankle_roll", 40.0),
        ("right_hip_pitch", 140.0), ("right_hip_roll", 110.0), ("right_hip_yaw", 80.0),
        ("right_knee", 130.0), ("right_ankle_pitch", 55.0), ("right_ankle_roll", 40.0),
    ]

    if any(not math.isfinite(g * s_scale) or g * s_scale <= 0 for _, g in actuator_specs):
        raise ValueError("Strength scaling produces nonfinite or zero motor gear")
    actuators = "\n".join(
        f'<motor name="act_{name}" joint="{name}" ctrllimited="true" ctrlrange="-1 1" gear="{_f(g * s_scale)}"/>'
        for name, g in actuator_specs
    )

    equality_xml = []
    for region in scene.contact_regions:
        if Affordance.GRASP in region.affordances:
            for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
                site_name = END_EFFECTOR_SITES[limb]
                eq_name = get_grasp_equality_name(limb, region.id)
                equality_xml.append(
                    f'<connect name="{escape(eq_name)}" site1="{site_name}" site2="site_{escape(region.id)}" '
                    f'active="false" solref="{_f(.02 if debug else .004)} 1" solimp="0.99 0.99 0.001"/>'
                )
        if debug and Affordance.STEP in region.affordances:
            for limb in (Limb.LEFT_FOOT, Limb.RIGHT_FOOT):
                site_name = END_EFFECTOR_SITES[limb]
                eq_name = get_grasp_equality_name(limb, region.id)
                equality_xml.append(
                    f'<connect name="{escape(eq_name)}" site1="{site_name}" site2="site_step_{escape(region.id)}" active="false"/>'
                )

    contact_excludes_xml = []
    # The legacy attachment scaffold is explicit debug behavior, not physical support.
    if debug:
        for region in scene.contact_regions:
            if region.source_type == SourceType.WALL:
                continue
            for b in ("left_hand", "left_forearm", "right_hand", "right_forearm", "left_shin", "right_shin"):
                contact_excludes_xml.append(
                    f'<exclude body1="{b}" body2="contact_{escape(region.id)}"/>'
                )
        for wall in scene.walls:
            for b in ("left_hand", "right_hand"):
                contact_excludes_xml.append(f'<exclude body1="{b}" body2="{escape(wall.id)}"/>')
    # Self-collision exclusions for adjacent torso/head/limb segments
    self_excludes = [
        ("pelvis", "abdomen"),
        ("abdomen", "chest"),
        ("chest", "head"),
        ("pelvis", "left_thigh"),
        ("pelvis", "right_thigh"),
        ("chest", "left_upper_arm"),
        ("chest", "right_upper_arm"),
        ("left_thigh", "left_shin"),
        ("right_thigh", "right_shin"),
        ("left_shin", "left_foot"),
        ("right_shin", "right_foot"),
        ("left_upper_arm", "left_forearm"),
        ("right_upper_arm", "right_forearm"),
        ("left_forearm", "left_hand"),
        ("right_forearm", "right_hand"),
    ]
    for b1, b2 in self_excludes:
        contact_excludes_xml.append(f'<exclude body1="{b1}" body2="{b2}"/>')

    contact_pairs_xml = []
    surfaces = [(f"geom_{region.id}", region.friction) for region in scene.contact_regions
                if region.source_type == SourceType.HOLD]
    surfaces += [(f"{wall.id}_geom", wall.friction) for wall in scene.walls]
    surfaces.append(("floor", 1.0))
    for shoe in ("left_foot_geom", "right_foot_geom"):
        for surface, friction in surfaces:
            mu = _f(min(SHOE_FRICTION, friction))
            contact_pairs_xml.append(
                f'<pair geom1="{shoe}" geom2="{escape(surface)}" condim="3" '
                f'friction="{mu} {mu} 0 0 0" solref="0.01 1"/>'
            )

    xml = f'''<mujoco model="boulder_prototype_v1_{escape(profile.name)}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <custom>
    <numeric name="contact_mode" data="{1 if debug else 0}"/>
  </custom>
  <visual>
    <global offwidth="1920" offheight="1080"/>
  </visual>
  <default>
    <joint limited="true" armature="0.01"/>
    <geom contype="1" conaffinity="1" solref="0.01 1"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" friction="1.0 0.05 0.005"/>
    {''.join(wall_xml)}
    {''.join(contact_xml)}
    {character_xml}
  </worldbody>
  <actuator>
    {actuators}
  </actuator>
  <equality>
    {''.join(equality_xml)}
  </equality>
  <contact>
    {''.join(contact_pairs_xml)}
    {''.join(contact_excludes_xml)}
  </contact>
</mujoco>
'''
    return xml
