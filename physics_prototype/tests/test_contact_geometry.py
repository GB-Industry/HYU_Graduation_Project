from dataclasses import fields, replace
import json
import math
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from boulder_v1.contact_geometry import (
    ContactGeometry,
    ContactMode,
    FOOT_SITE_OFFSET,
    Frame,
    HAND_SITE_OFFSET,
    SHOE_FRICTION,
    canonical_geometry,
)
from boulder_v1.mjcf_builder import END_EFFECTOR_SITES, build_mjcf, get_grasp_equality_name
from boulder_v1.retargeter import StanceSpecification, solve_retargeted_stance
from boulder_v1.runtime import mujoco_available
from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import Affordance, ClimberProfile, ContactRegion, Limb, SourceType


class ContactGeometryTests(unittest.TestCase):
    def setUp(self):
        self.region = make_synthetic_scene().region("H1")

    def test_half_size_is_optional_final_field_and_serialized(self):
        self.assertEqual(fields(ContactRegion)[-1].name, "half_size")
        self.assertIsNone(self.region.half_size)
        self.assertIsNone(self.region.to_dict()["half_size"])
        size = [.08, .04, .06]
        box = replace(self.region, half_size=size)
        size.clear()
        self.assertEqual(box.half_size, (.08, .04, .06))
        self.assertEqual(box.radius, self.region.radius)
        payload = json.loads(json.dumps(box.to_dict(), allow_nan=False))
        self.assertEqual(payload["half_size"], [.08, .04, .06])
        self.assertEqual(payload["radius"], self.region.radius)

    def test_half_size_rejects_nonpositive_nonfinite_and_malformed_vectors(self):
        invalid = ((), (.1, .2), (.1, .2, .3, .4), "123", {1, 2, 3}, {"x": .1},
                   (.1, 0, .1), (.1, -.1, .1), (.1, True, .1), (.1, "0.1", .1),
                   (.1, None, .1), (.1, float("nan"), .1), (.1, float("inf"), .1))
        for size in invalid:
            with self.subTest(size=size), self.assertRaises(ValueError):
                replace(self.region, half_size=size)

    def test_default_sphere_frames_and_physical_offsets(self):
        g = canonical_geometry(self.region)
        self.assertIsInstance(g, ContactGeometry)
        self.assertEqual(g.shape, "sphere")
        self.assertEqual(g.size, (.075,))
        self.assertEqual(g.body_frame.position, self.region.position)
        np.testing.assert_array_equal(g.body_frame.rotation, np.eye(3))
        np.testing.assert_array_equal(g.hand_frame.rotation, [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        np.testing.assert_allclose(g.hand_frame.position, [-.33, -.101, .72], atol=1e-15)
        normal = np.array([0, -.5, math.sqrt(3) / 2])
        rotation = [[1, 0, 0], [0, math.sqrt(3) / 2, -.5], [0, .5, math.sqrt(3) / 2]]
        np.testing.assert_allclose(g.foot_surface_frame.rotation, rotation, atol=1e-15)
        np.testing.assert_allclose(g.foot_surface_frame.normal, normal, atol=1e-15)
        np.testing.assert_allclose(g.foot_surface_frame.position,
                                   np.array(self.region.position) + .075 * normal, atol=1e-15)
        np.testing.assert_allclose(g.foot_frame.position,
                                   np.array(self.region.position) + (.075 + .011) * normal, atol=1e-15)
        # Site-to-distal-palm and site-to-sole offsets, without moving either END site.
        self.assertEqual((HAND_SITE_OFFSET, FOOT_SITE_OFFSET, SHOE_FRICTION), (.006, .011, 1.8))
        self.assertAlmostEqual(-.04 - HAND_SITE_OFFSET, -.03 - .016)
        self.assertAlmostEqual(-.025 - FOOT_SITE_OFFSET, -.018 - .018)

    def test_default_box_frames_and_depth_height_independence(self):
        region = replace(self.region, half_size=(.08, .035, .055))
        g = canonical_geometry(region)
        self.assertEqual((g.shape, g.size), ("box", region.half_size))
        np.testing.assert_allclose(g.hand_frame.position, [-.33, -.061, .72], atol=1e-15)
        np.testing.assert_allclose(g.foot_surface_frame.position, [-.33, -.02, .775], atol=1e-15)
        np.testing.assert_allclose(g.foot_frame.position, [-.33, -.02, .786], atol=1e-15)
        np.testing.assert_array_equal(g.foot_frame.rotation, np.eye(3))
        changed = canonical_geometry(replace(region, radius=.3, half_size=(.1, .06, .08)))
        np.testing.assert_allclose(np.subtract(changed.hand_frame.position, g.hand_frame.position),
                                   [0, -.025, 0], atol=1e-15)
        np.testing.assert_allclose(np.subtract(changed.foot_frame.position, g.foot_frame.position),
                                   [0, 0, .025], atol=1e-15)

    def test_normals_project_up_and_all_frames_are_right_handed(self):
        normals = ((.2, -1, .3), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1),
                   (1e-14, 0, 1), (1e-8, -1e-8, 1))
        for normal in normals:
            for size in (None, (.08, .04, .06)):
                region = replace(self.region, normal=normal, half_size=size)
                g = canonical_geometry(region)
                with self.subTest(normal=normal, size=size):
                    n = np.array(region.normal)
                    body = np.array(g.body_frame.rotation)
                    up = body[:, 2]
                    np.testing.assert_allclose(body[:, 1], -n, atol=1e-14)
                    reference = np.array([0, 0, 1] if math.hypot(n[0], n[1]) >= 1e-12 else [0, 1, 0])
                    expected_up = np.cross(n, np.cross(reference, n))
                    expected_up /= np.linalg.norm(expected_up)
                    np.testing.assert_allclose(up, expected_up, atol=1e-14)
                    np.testing.assert_allclose(g.hand_frame.normal, n, atol=1e-14)
                    depth = region.radius if size is None else size[1]
                    np.testing.assert_allclose(np.subtract(g.hand_frame.position, region.position),
                                               (depth + .006) * n, atol=1e-14)
                    foot_normal = .5 * n + math.sqrt(3) / 2 * up if size is None else up
                    surface_offset = region.radius * foot_normal if size is None else size[2] * up
                    np.testing.assert_allclose(np.subtract(g.foot_surface_frame.position, region.position),
                                               surface_offset, atol=1e-14)
                    np.testing.assert_allclose(np.subtract(g.foot_frame.position, g.foot_surface_frame.position),
                                               .011 * foot_normal, atol=1e-14)
                    for frame in (g.body_frame, g.hand_frame, g.foot_surface_frame, g.foot_frame):
                        self.assertIsInstance(frame.position, tuple)
                        self.assertTrue(all(isinstance(row, tuple) for row in frame.rotation))
                        r = np.array(frame.rotation)
                        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
                        self.assertAlmostEqual(np.linalg.det(r), 1.0, places=14)
                        w, x, y, z = frame.quaternion
                        self.assertAlmostEqual(math.hypot(w, x, y, z), 1.0, places=14)
                        reconstructed = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
                        np.testing.assert_allclose(reconstructed, r, atol=1e-14)

    def test_quaternion_half_turns(self):
        for diagonal in ((1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            rotation = tuple(tuple(diagonal[i] if i == j else 0 for j in range(3)) for i in range(3))
            q = Frame((0, 0, 0), rotation).quaternion
            self.assertEqual(q[0], 0)
            self.assertEqual(q[1 + diagonal.index(1)], 1)


class ContactBuilderTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_synthetic_scene()
        self.profile = ClimberProfile(name="base")

    def test_physical_default_and_explicit_debug_policy(self):
        physical = ET.fromstring(build_mjcf(self.scene, self.profile))
        explicit = ET.fromstring(build_mjcf(self.scene, self.profile, ContactMode.PHYSICAL))
        self.assertEqual(ET.tostring(physical), ET.tostring(explicit))
        debug = ET.fromstring(build_mjcf(self.scene, self.profile, ContactMode.IDEALIZED_DEBUG))
        for root, mode, equality_count, exclude_count in ((physical, "0", 16, 15), (debug, "1", 30, 65)):
            with self.subTest(mode=mode):
                self.assertEqual(root.find("./custom/numeric[@name='contact_mode']").get("data"), mode)
                self.assertEqual(len(root.findall("./equality/connect")), equality_count)
                excludes = root.findall("./contact/exclude")
                self.assertEqual(len(excludes), exclude_count)
                environment = [e for e in excludes if e.get("body2").startswith(("contact_", "wall_"))]
                self.assertEqual(len(environment), 0 if mode == "0" else 50)
                self.assertEqual(len(root.findall("./contact/pair")), 20)
        physical_self = {(e.get("body1"), e.get("body2")) for e in physical.findall("./contact/exclude")}
        self.assertEqual(physical_self, {
            ("pelvis", "abdomen"), ("abdomen", "chest"), ("chest", "head"),
            ("pelvis", "left_thigh"), ("pelvis", "right_thigh"),
            ("chest", "left_upper_arm"), ("chest", "right_upper_arm"),
            ("left_thigh", "left_shin"), ("right_thigh", "right_shin"),
            ("left_shin", "left_foot"), ("right_shin", "right_foot"),
            ("left_upper_arm", "left_forearm"), ("right_upper_arm", "right_forearm"),
            ("left_forearm", "left_hand"), ("right_forearm", "right_hand"),
        })
        self.assertEqual(ET.tostring(physical.find(".//body[@name='climber_root']")),
                         ET.tostring(debug.find(".//body[@name='climber_root']")))
        self.assertEqual(ET.tostring(physical.find("./actuator")), ET.tostring(debug.find("./actuator")))

    def test_all_hand_equalities_use_end_sites_and_strict_physical_parameters(self):
        for mode in ContactMode:
            root = ET.fromstring(build_mjcf(self.scene, self.profile, mode))
            for region in self.scene.contact_regions:
                if Affordance.GRASP not in region.affordances:
                    continue
                for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
                    name = get_grasp_equality_name(limb, region.id)
                    eq = root.find(f"./equality/connect[@name='{name}']")
                    self.assertEqual(eq.get("site1"), END_EFFECTOR_SITES[limb])
                    self.assertEqual(eq.get("site2"), f"site_{region.id}")
                    self.assertEqual(eq.get("active"), "false")
                    self.assertEqual(tuple(map(float, eq.get("solref").split())),
                                     (.004 if mode == ContactMode.PHYSICAL else .02, 1))
                    self.assertEqual(tuple(map(float, eq.get("solimp").split())), (.99, .99, .001))
            if mode == ContactMode.PHYSICAL:
                self.assertFalse(any("foot" in eq.get("name") for eq in root.findall("./equality/connect")))

    def test_pairs_cover_both_shoes_all_holds_walls_floor_and_minimum_friction(self):
        regions = tuple(replace(r, friction=2.4) if r.id == "TOP" else r for r in self.scene.contact_regions)
        scene = replace(self.scene, contact_regions=regions, walls=(replace(self.scene.walls[0], friction=.25),))
        root = ET.fromstring(build_mjcf(scene, self.profile))
        expected = {f"geom_{r.id}": min(1.8, r.friction) for r in regions if r.source_type == SourceType.HOLD}
        expected.update({"wall_main_geom": .25, "floor": 1.0})
        pairs = root.findall("./contact/pair")
        self.assertEqual({(p.get("geom1"), p.get("geom2")) for p in pairs},
                         {(shoe, surface) for shoe in ("left_foot_geom", "right_foot_geom") for surface in expected})
        for pair in pairs:
            self.assertEqual(pair.get("condim"), "3")
            mu = expected[pair.get("geom2")]
            self.assertEqual(tuple(map(float, pair.get("friction").split())), (mu, mu, 0, 0, 0))
            self.assertEqual(tuple(map(float, pair.get("solref").split())), (.01, 1))

    def test_builder_rejects_wall_box_and_unsupported_box_domain(self):
        for half_size in ((.05, .05, .05), (1e-7, .05, .05), (.05, 101, .05)):
            region_id = "WALL_PATCH" if half_size == (.05, .05, .05) else "H1"
            region = replace(self.scene.region(region_id), half_size=half_size)
            scene = replace(self.scene, contact_regions=tuple(region if r.id == region_id else r
                                                             for r in self.scene.contact_regions))
            for mode in ContactMode:
                with self.subTest(region=region_id, size=half_size, mode=mode), self.assertRaises(ValueError):
                    build_mjcf(scene, self.profile, mode)
        for mode in (None, "physical", "idealized_debug", 0):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "ContactMode"):
                build_mjcf(self.scene, self.profile, mode)


@unittest.skipUnless(mujoco_available(), "MuJoCo is not available")
class CompiledContactGeometryTests(unittest.TestCase):
    def setUp(self):
        import mujoco

        self.mujoco = mujoco
        self.scene = make_synthetic_scene()
        self.profile = ClimberProfile(name="base")

    def test_compiled_sites_and_geoms_equal_canonical_frames_in_both_modes(self):
        normals = ((0, -1, 0), (.2, -1, .3), (0, 0, 1), (0, 0, -1))
        for normal in normals:
            for half_size in (None, (.08, .045, .06)):
                regions = tuple(replace(r, normal=normal, half_size=half_size,
                                        position=(r.position[0] + .01, r.position[1] - .025, r.position[2] + .015))
                                if r.source_type == SourceType.HOLD else r for r in self.scene.contact_regions)
                scene = replace(self.scene, contact_regions=regions)
                for mode in ContactMode:
                    with self.subTest(normal=normal, size=half_size, mode=mode):
                        model = self.mujoco.MjModel.from_xml_string(build_mjcf(scene, self.profile, mode))
                        data = self.mujoco.MjData(model)
                        self.mujoco.mj_forward(model, data)
                        for region in regions:
                            if region.source_type != SourceType.HOLD:
                                continue
                            g = canonical_geometry(region)
                            body = data.body(f"contact_{region.id}")
                            np.testing.assert_allclose(body.xpos, g.body_frame.position, atol=1e-14)
                            np.testing.assert_allclose(body.xmat.reshape(3, 3), g.body_frame.rotation, atol=1e-14)
                            geom = model.geom(f"geom_{region.id}")
                            expected_type = self.mujoco.mjtGeom.mjGEOM_SPHERE if half_size is None else self.mujoco.mjtGeom.mjGEOM_BOX
                            self.assertEqual(int(geom.type[0]), expected_type)
                            np.testing.assert_array_equal(geom.size[:len(g.size)], g.size)
                            np.testing.assert_allclose(data.geom(geom.name).xmat.reshape(3, 3), g.body_frame.rotation, atol=1e-14)
                            for name, frame in ((f"site_{region.id}", g.hand_frame),
                                                (f"site_step_{region.id}", g.foot_frame)):
                                if name.startswith("site_step_") and Affordance.STEP not in region.affordances:
                                    continue
                                site = data.site(name)
                                np.testing.assert_allclose(site.xpos, frame.position, atol=1e-14)
                                np.testing.assert_allclose(site.xmat.reshape(3, 3), frame.rotation, atol=1e-14)

    def test_compiled_mode_equalities_pairs_and_stage1_character(self):
        for mode in ContactMode:
            model = self.mujoco.MjModel.from_xml_string(build_mjcf(self.scene, self.profile, mode))
            self.assertEqual(model.numeric("contact_mode").data.tolist(), [0 if mode == ContactMode.PHYSICAL else 1])
            self.assertEqual((model.nq, model.nv, model.nu, model.njnt), (32, 31, 25, 26))
            self.assertEqual((model.neq, model.npair, model.nexclude),
                             (16, 20, 15) if mode == ContactMode.PHYSICAL else (30, 20, 65))
            self.assertFalse(np.any(model.eq_active0))
            self.assertAlmostEqual(model.opt.timestep, .002)
            self.assertEqual(model.opt.integrator, self.mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
            for eq in range(model.neq):
                self.assertEqual(model.eq_type[eq], self.mujoco.mjtEq.mjEQ_CONNECT)
                self.assertEqual(model.eq_objtype[eq], self.mujoco.mjtObj.mjOBJ_SITE)
                self.assertIn(model.site(int(model.eq_obj1id[eq])).name, END_EFFECTOR_SITES.values())
                self.assertTrue(model.site(int(model.eq_obj2id[eq])).name.startswith("site_"))
            for region in self.scene.contact_regions:
                if Affordance.GRASP not in region.affordances:
                    continue
                for limb in (Limb.LEFT_HAND, Limb.RIGHT_HAND):
                    name = get_grasp_equality_name(limb, region.id)
                    eq = self.mujoco.mj_name2id(model, self.mujoco.mjtObj.mjOBJ_EQUALITY, name)
                    self.assertGreaterEqual(eq, 0)
                    self.assertEqual(model.eq_obj1id[eq], model.site(END_EFFECTOR_SITES[limb]).id)
                    self.assertEqual(model.eq_obj2id[eq], model.site(f"site_{region.id}").id)
                    np.testing.assert_array_equal(model.eq_solref[eq],
                                                  [.004 if mode == ContactMode.PHYSICAL else .02, 1])
                    np.testing.assert_array_equal(model.eq_solimp[eq, :3], [.99, .99, .001])
            np.testing.assert_array_equal(model.pair_dim, [3] * 20)
            np.testing.assert_array_equal(model.pair_solref, [[.01, 1]] * 20)
            for pair in range(model.npair):
                geoms = {model.geom(int(model.pair_geom1[pair])).name, model.geom(int(model.pair_geom2[pair])).name}
                surface = (geoms - {"left_foot_geom", "right_foot_geom"}).pop()
                friction = 1.0 if surface == "floor" else model.geom(surface).friction[0]
                mu = min(SHOE_FRICTION, friction)
                np.testing.assert_array_equal(model.pair_friction[pair], [mu, mu, 0, 0, 0])
            for side in ("left", "right"):
                np.testing.assert_array_equal(model.site(f"{side}_hand_site").pos, [0, 0, -.04])
                np.testing.assert_array_equal(model.site(f"{side}_foot_site").pos, [0, .09, -.025])
                np.testing.assert_array_equal(model.geom(f"{side}_foot_geom").size, [.040, .095, .018])
                np.testing.assert_array_equal(model.geom(f"{side}_foot_geom").pos, [0, .035, -.018])
                np.testing.assert_array_equal(model.geom(f"{side}_foot_geom").friction, [1.8, .05, .005])

    def test_retarget_errors_are_measured_against_canonical_points_across_variations(self):
        variations = ({}, {"radius": .07}, {"normal": (.12, -1, .1)},
                      {"half_size": (.075, .060, .068)},
                      {"half_size": (.075, .060, .068), "normal": (.12, -1, .1)})
        for updates in variations:
            regions = tuple(replace(r, **updates) if r.source_type == SourceType.HOLD else r
                            for r in self.scene.contact_regions)
            scene = replace(self.scene, contact_regions=regions)
            if len(updates) == 2:
                scene = replace(scene, contact_regions=tuple(replace(r, position=(r.position[0], r.position[1] - .012,
                                                                                  r.position[2] + .006))
                                                            if r.source_type == SourceType.HOLD else r for r in regions))
            for mode in ContactMode:
                with self.subTest(updates=updates, mode=mode):
                    model = self.mujoco.MjModel.from_xml_string(build_mjcf(scene, self.profile, mode))
                    result = solve_retargeted_stance(model, scene, self.profile)
                    self.assertTrue(result.converged)
                    data = self.mujoco.MjData(model)
                    data.qpos[:] = result.qpos
                    self.mujoco.mj_forward(model, data)
                    self.assertEqual(set(result.errors), set(Limb))
                    for limb, region_id in scene.start_configuration.items():
                        g = canonical_geometry(scene.region(region_id))
                        frame = g.hand_frame if limb.is_hand else g.foot_frame
                        actual = float(np.linalg.norm(data.site(END_EFFECTOR_SITES[limb]).xpos - frame.position))
                        self.assertAlmostEqual(result.errors[limb], actual, places=14)
                        self.assertLess(actual, .002, f"{limb.value}: {actual}")

    def test_custom_targets_and_free_limb_semantics_remain_unchanged(self):
        model = self.mujoco.MjModel.from_xml_string(build_mjcf(self.scene, self.profile))
        anchor = canonical_geometry(self.scene.region("H3")).hand_frame.position
        custom = (anchor[0] + .012, anchor[1] - .003, anchor[2] + .004)
        spec = StanceSpecification(free_limbs=(Limb.LEFT_HAND, Limb.RIGHT_HAND),
                                   custom_limb_targets={Limb.LEFT_HAND: custom})
        result = solve_retargeted_stance(model, self.scene, self.profile, spec=spec)
        self.assertTrue(result.converged)
        data = self.mujoco.MjData(model)
        data.qpos[:] = result.qpos
        self.mujoco.mj_forward(model, data)
        self.assertNotIn(Limb.RIGHT_HAND, result.errors)
        actual = np.linalg.norm(data.site("left_hand_site").xpos - custom)
        self.assertAlmostEqual(result.errors[Limb.LEFT_HAND], actual, places=14)
        self.assertLess(actual, .002)


if __name__ == "__main__":
    unittest.main()
