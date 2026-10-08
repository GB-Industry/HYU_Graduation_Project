import re
from dataclasses import replace
import unittest
import xml.etree.ElementTree as ET

from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.contact_geometry import ContactMode
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import ClimberProfile


def motor_gear(xml: str, motor_name: str) -> float:
    root = ET.fromstring(xml)
    for motor in root.findall("./actuator/motor"):
        if motor.attrib.get("name") == motor_name:
            return float(motor.attrib["gear"])
    raise KeyError(motor_name)


def joint_range(xml: str, joint_name: str) -> tuple[float, float]:
    root = ET.fromstring(xml)
    for joint in root.findall(".//joint"):
        if joint.attrib.get("name") == joint_name:
            lo, hi = joint.attrib["range"].split()
            return float(lo), float(hi)
    raise KeyError(joint_name)


class MJCFBuilderTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_synthetic_scene()

    def test_generated_xml_is_well_formed(self):
        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)
        self.assertEqual(root.tag, "mujoco")
        self.assertIsNotNone(root.find("./worldbody/body[@name='climber_root']/body[@name='pelvis']"))

    def test_free_root_uses_rigid_pelvis_mass_budget(self):
        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)
        inertial = root.find("./worldbody/body[@name='climber_root']/inertial")
        self.assertIsNone(inertial)  # Massless frame; welded pelvis carries the budget.
        inertial = root.find("./worldbody/body[@name='climber_root']/body[@name='pelvis']/inertial")
        self.assertEqual(float(inertial.attrib["mass"]), 10.0)
        self.assertTrue(all(float(v) > 0.0 for v in inertial.attrib["diaginertia"].split()))

    def test_morphology_changes_segment_length(self):
        short = build_mjcf(self.scene, ClimberProfile(name="short", upper_arm_length=0.25))
        long = build_mjcf(self.scene, ClimberProfile(name="long", upper_arm_length=0.38))
        self.assertIn('fromto="0 0 0 0 0 -0.25"', short)
        self.assertIn('fromto="0 0 0 0 0 -0.38"', long)

    def test_strength_changes_actuator_gear(self):
        weak = build_mjcf(self.scene, ClimberProfile(name="weak", strength_scale=0.7))
        strong = build_mjcf(self.scene, ClimberProfile(name="strong", strength_scale=1.3))
        self.assertLess(motor_gear(weak, "act_left_shoulder_pitch"), motor_gear(strong, "act_left_shoulder_pitch"))

    def test_rom_changes_joint_range(self):
        tight = build_mjcf(self.scene, ClimberProfile(name="tight", rom_scale=0.8))
        mobile = build_mjcf(self.scene, ClimberProfile(name="mobile", rom_scale=1.2))
        self.assertLess(joint_range(tight, "left_shoulder_pitch")[1], joint_range(mobile, "left_shoulder_pitch")[1])

    def test_mass_scale_changes_mass(self):
        light = build_mjcf(self.scene, ClimberProfile(name="light", mass_scale=0.8))
        heavy = build_mjcf(self.scene, ClimberProfile(name="heavy", mass_scale=1.2))
        root_light = ET.fromstring(light)
        root_heavy = ET.fromstring(heavy)
        pelvis_light = float(root_light.find(".//geom[@name='pelvis_geom']").attrib["mass"])
        pelvis_heavy = float(root_heavy.find(".//geom[@name='pelvis_geom']").attrib["mass"])
        self.assertLess(pelvis_light, pelvis_heavy)
        self.assertAlmostEqual(pelvis_light, 10.0 * 0.8, places=2)
        self.assertAlmostEqual(pelvis_heavy, 10.0 * 1.2, places=2)

    def test_humanoid_topology_and_dof(self):
        """Verify 25 actuated DoF (in target range 23-27) and anthropomorphic body segments."""
        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)

        motors = root.findall("./actuator/motor")
        self.assertEqual(len(motors), 25, "Expected exactly 25 actuated motors")
        self.assertTrue(23 <= len(motors) <= 27, "Actuated DoF must be in 23-27 range")

        # Key anthropomorphic body segments
        expected_bodies = [
            "pelvis", "abdomen", "chest", "head",
            "left_upper_arm", "left_forearm", "left_hand",
            "right_upper_arm", "right_forearm", "right_hand",
            "left_thigh", "left_shin", "left_foot",
            "right_thigh", "right_shin", "right_foot",
        ]
        for b in expected_bodies:
            body = root.find(f".//body[@name='{b}']")
            self.assertIsNotNone(body, f"Missing required body segment '{b}'")

        # Bilateral symmetry in limb actuator gears
        symmetric_pairs = [
            ("act_left_shoulder_pitch", "act_right_shoulder_pitch"),
            ("act_left_shoulder_roll", "act_right_shoulder_roll"),
            ("act_left_shoulder_yaw", "act_right_shoulder_yaw"),
            ("act_left_elbow", "act_right_elbow"),
            ("act_left_wrist", "act_right_wrist"),
            ("act_left_hip_pitch", "act_right_hip_pitch"),
            ("act_left_hip_roll", "act_right_hip_roll"),
            ("act_left_hip_yaw", "act_right_hip_yaw"),
            ("act_left_knee", "act_right_knee"),
            ("act_left_ankle_pitch", "act_right_ankle_pitch"),
            ("act_left_ankle_roll", "act_right_ankle_roll"),
        ]
        for left_act, right_act in symmetric_pairs:
            self.assertEqual(motor_gear(xml, left_act), motor_gear(xml, right_act))

    def test_end_effector_sites_defined(self):
        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)
        expected_sites = {
            "left_hand": "left_hand_site",
            "right_hand": "right_hand_site",
            "left_foot": "left_foot_site",
            "right_foot": "right_foot_site",
        }
        for body_name, site_name in expected_sites.items():
            site = root.find(f".//body[@name='{body_name}']//site[@name='{site_name}']")
            self.assertIsNotNone(site, f"Missing site {site_name} in body {body_name}")
            self.assertGreater(float(site.attrib.get("size", "0")), 0.0)
            self.assertIn("rgba", site.attrib)

    def test_contact_region_sites_have_rgba(self):
        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)
        for region in self.scene.contact_regions:
            if region.source_type.value == "WALL":
                continue
            site = root.find(f".//body[@name='contact_{region.id}']//site[@name='site_{region.id}']")
            self.assertIsNotNone(site)
            self.assertIn("rgba", site.attrib)

    def test_grasp_equality_constraints_predeclared(self):
        from boulder_v1.mjcf_builder import get_grasp_equality_name
        from boulder_v1.schema import Affordance, Limb

        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)

        eq_section = root.find("./equality")
        self.assertIsNotNone(eq_section, "Missing <equality> element in MJCF")
        self.assertEqual(root.find("./custom/numeric[@name='contact_mode']").attrib["data"], "0")
        self.assertEqual(len(eq_section.findall("connect")), 16)
        self.assertFalse(any("foot" in conn.attrib["name"] for conn in eq_section))

        grasp_regions = [r for r in self.scene.contact_regions if Affordance.GRASP in r.affordances]
        self.assertEqual(len(grasp_regions), 8)  # H1 through H7 and TOP

        for region in grasp_regions:
            for limb, site_name in [
                (Limb.LEFT_HAND, "left_hand_site"),
                (Limb.RIGHT_HAND, "right_hand_site"),
            ]:
                eq_name = get_grasp_equality_name(limb, region.id)
                conn = eq_section.find(f"./connect[@name='{eq_name}']")
                self.assertIsNotNone(conn, f"Missing equality constraint {eq_name}")
                self.assertEqual(conn.attrib.get("site1"), site_name)
                self.assertEqual(conn.attrib.get("site2"), f"site_{region.id}")
                self.assertEqual(conn.attrib.get("active"), "false")

    def test_contact_exclusions_predeclared(self):
        from boulder_v1.schema import SourceType

        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)

        contact_section = root.find("./contact")
        self.assertIsNotNone(contact_section, "Missing <contact> element in MJCF")

        climber_bodies = {body.attrib["name"] for body in root.findall(".//body[@name='climber_root']//body")}
        exclusions = contact_section.findall("exclude")
        self.assertEqual(len(exclusions), 15)
        for exclusion in exclusions:
            self.assertIn(exclusion.attrib["body1"], climber_bodies)
            self.assertIn(exclusion.attrib["body2"], climber_bodies)
        pairs = contact_section.findall("pair")
        self.assertEqual(len(pairs), 20)
        frictions = {f"geom_{r.id}": r.friction for r in self.scene.contact_regions if r.source_type == SourceType.HOLD}
        frictions.update({f"{w.id}_geom": w.friction for w in self.scene.walls})
        frictions["floor"] = 1.
        self.assertEqual({(p.attrib["geom1"], p.attrib["geom2"]) for p in pairs},
                         {(shoe, surface) for shoe in ("left_foot_geom", "right_foot_geom") for surface in frictions})
        for pair in pairs:
            mu = min(1.8, frictions[pair.attrib["geom2"]])
            self.assertEqual([float(v) for v in pair.attrib["friction"].split()], [mu, mu, 0., 0., 0.])
            self.assertEqual(pair.attrib["condim"], "3")

    def test_idealized_debug_has_explicit_legacy_foot_equalities_and_environment_debt(self):
        from boulder_v1.schema import SourceType

        root = ET.fromstring(build_mjcf(self.scene, ClimberProfile(name="base"),
                                      contact_mode=ContactMode.IDEALIZED_DEBUG))
        self.assertEqual(root.find("./custom/numeric[@name='contact_mode']").attrib["data"], "1")
        equalities = root.findall("./equality/connect")
        self.assertEqual(sum("hand" in e.attrib["name"] for e in equalities), 16)
        self.assertEqual(sum("foot" in e.attrib["name"] for e in equalities), 14)
        self.assertEqual(len(root.findall("./contact/exclude")), 65)
        contact_section = root.find("./contact")

        hold_regions = [r for r in self.scene.contact_regions if r.source_type != SourceType.WALL]
        for region in hold_regions:
            for b in ("left_hand", "left_forearm", "right_hand", "right_forearm", "left_shin", "right_shin"):
                ex = contact_section.find(f"./exclude[@body1='{b}'][@body2='contact_{region.id}']")
                self.assertIsNotNone(ex, f"Missing contact exclusion for {b} and contact_{region.id}")

        for wall in self.scene.walls:
            for b in ("left_hand", "right_hand"):
                ex = contact_section.find(f"./exclude[@body1='{b}'][@body2='{wall.id}']")
                self.assertIsNotNone(ex, f"Missing contact exclusion for {b} and {wall.id}")

    def test_refined_morphology_ellipsoids_and_inertias(self):
        """Structural regression; compiled analytical tensors are tested separately."""
        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)

        # Torso segment geometries are now ellipsoids rather than blocky boxes/cylinders
        for body_name, geom_name in [
            ("pelvis", "pelvis_geom"),
            ("abdomen", "abdomen_geom"),
            ("chest", "chest_geom"),
        ]:
            geom = root.find(f".//body[@name='{body_name}']/geom[@name='geom_name']".replace("geom_name", geom_name))
            self.assertIsNotNone(geom, f"Missing {geom_name} on {body_name}")
            self.assertEqual(geom.attrib.get("type"), "ellipsoid", f"{geom_name} must be an ellipsoid")

        # Explicit simplified ellipsoid inertias, not physiological calibration.
        for body_name in ("pelvis", "abdomen", "chest"):
            inertial = root.find(f".//body[@name='{body_name}']/inertial")
            self.assertIsNotNone(inertial, f"Missing explicit <inertial> on {body_name}")
            self.assertGreater(float(inertial.attrib["mass"]), 0.0)
            iner_diag = [float(v) for v in inertial.attrib["diaginertia"].split()]
            self.assertEqual(len(iner_diag), 3)
            self.assertTrue(all(v > 0.0 for v in iner_diag))

        # Cranial ellipsoid and neck capsule
        neck = root.find(".//geom[@name='neck_geom']")
        self.assertIsNotNone(neck)
        self.assertEqual(neck.attrib.get("type"), "capsule")
        self.assertLessEqual(float(neck.attrib.get("size")), 0.04)

        head = root.find(".//geom[@name='head_geom']")
        self.assertIsNotNone(head)
        self.assertEqual(head.attrib.get("type"), "ellipsoid")

    def test_climbing_shoe_geometry(self):
        """Verify sleek climbing shoe geometry with high-friction sticky rubber."""
        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        root = ET.fromstring(xml)

        for side in ("left", "right"):
            shoe = root.find(f".//geom[@name='{side}_foot_geom']")
            self.assertIsNotNone(shoe)
            self.assertEqual(shoe.attrib.get("type"), "box")
            sizes = [float(s) for s in shoe.attrib.get("size").split()]
            self.assertEqual(len(sizes), 3)
            # Width <= 5cm half-width (10cm total), thickness <= 3cm half-height
            self.assertLessEqual(sizes[0], 0.05)
            self.assertLessEqual(sizes[2], 0.03)

            frictions = [float(f) for f in shoe.attrib.get("friction").split()]
            self.assertGreaterEqual(frictions[0], 1.5, "Shoe sliding friction must be >= 1.5 for bouldering rubber")

    def test_backend_rejects_unsupported_scene_features_before_compile(self):
        from boulder_v1.schema import Affordance, Limb, SourceType

        cases = [replace(self.scene, scale=2),
                 replace(self.scene, walls=(replace(self.scene.walls[0], normal=(1, 0, 0)),))]
        for source in (SourceType.VOLUME, SourceType.EDGE):
            cases.append(replace(self.scene, contact_regions=(replace(self.scene.contact_regions[0], source_type=source),
                                                               *self.scene.contact_regions[1:])))
        for mode in (Affordance.GRASP, Affordance.STEP):
            patch = replace(self.scene.region("WALL_PATCH"), affordances=frozenset([mode, Affordance.PRESS]))
            cases.append(replace(self.scene, contact_regions=(*self.scene.contact_regions[:-1], patch)))
        starts = dict(self.scene.start_configuration)
        starts[Limb.LEFT_HAND] = "WALL_PATCH"
        cases.append(replace(self.scene, start_configuration=starts))
        for scene in cases:
            with self.subTest(scene=scene), self.assertRaises(ValueError):
                build_mjcf(scene, ClimberProfile(name="base"))
        xml = build_mjcf(self.scene, ClimberProfile(name="base"))
        self.assertNotIn('site_WALL_PATCH', xml)  # PRESS/SMEAR patch is metadata only.

    def test_tiny_values_are_preserved_and_derived_failures_rejected(self):
        from boulder_v1.mjcf_builder import _f
        from boulder_v1.runtime import compile_model

        for value in (1e-5, -1e-5, 1e-8, 6e-5):
            self.assertEqual(float(_f(value)), value)
            self.assertNotEqual(float(_f(value)), 0)
        tiny = replace(self.scene.contact_regions[0], radius=1e-5)
        scene = replace(self.scene, contact_regions=(tiny, *self.scene.contact_regions[1:]))
        model = compile_model(build_mjcf(scene, ClimberProfile(name="tiny", strength_scale=1e-6)))
        self.assertEqual(model.geom("geom_H1").size[0], 1e-5)
        self.assertGreater(model.actuator_gear[0, 0], 0)
        for updates in ({"strength_scale": 1e308}, {"strength_scale": 1e152}, {"strength_scale": 1e-310},
                        {"mass_scale": 1e308}, {"mass_scale": 1e-300},
                        {"forearm_length": 1e308}, {"torso_length": 1e308}, {"upper_arm_length": 1e-300},
                        {"upper_arm_length": 1e8},
                        {"rom_scale": 1e-300}, {"rom_scale": 2}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                build_mjcf(self.scene, ClimberProfile(name="unsupported", **updates))
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                _f(value)
        for updates in ({"friction": 1e308}, {"friction": 1e-310}):
            wall = replace(self.scene.walls[0], **updates)
            with self.assertRaises(ValueError):
                build_mjcf(replace(self.scene, walls=(wall,)), ClimberProfile(name="base"))


if __name__ == "__main__":
    unittest.main()
