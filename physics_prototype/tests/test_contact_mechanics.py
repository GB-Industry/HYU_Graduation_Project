import json
import math
from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from boulder_v1.contact_benchmarks import (
    FOOT_CASES, GRAVITY, MASS, SHOE_CENTER, SHOE_SIZE, _feet, _fresh, _hand_reactions, _native_contacts, _scene,
    make_foot_fixture, make_hand_fixture, make_mixed_fixture, run_capture_gates, run_foot_case,
    run_hand_case, run_mixed_case, shoe_measurement, verify_fixed_allocation,
)
from boulder_v1.mjcf_builder import build_mjcf
from boulder_v1.schema import ClimberProfile, Limb


class FreeShoeMechanicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = [run_foot_case(case, dt) for dt in (.002, .001) for case in FOOT_CASES]

    def test_native_support_friction_incline_and_separation(self):
        for result in self.results:
            with self.subTest(case=result["case"], dt=result["dt_s"]):
                self.assertTrue(result["contact_acceptance"], result["checks"])
                json.dumps(result, allow_nan=False)
        support = [r for r in self.results if r["case"].startswith("support_")]
        for result in support:
            self.assertAlmostEqual(result["tail_normal_mean_N"], MASS * GRAVITY, delta=.02 * MASS * GRAVITY)

    def test_shoe_geometry_mass_and_no_pins_match_production(self):
        fixture = make_foot_fixture(FOOT_CASES[0])
        model = fixture.model
        production = mujoco.MjModel.from_xml_string(build_mjcf(_scene(), ClimberProfile("shape_audit")))
        np.testing.assert_array_equal(model.geom("left_foot_geom").size, SHOE_SIZE)
        np.testing.assert_array_equal(model.geom("left_foot_geom").pos, SHOE_CENTER)
        for field in ("size", "pos", "friction", "type"):
            np.testing.assert_array_equal(getattr(model.geom("left_foot_geom"), field),
                                          getattr(production.geom("left_foot_geom"), field))
        self.assertEqual((model.nq, model.nv, model.neq, model.nu), (7, 6, 0, 0))
        self.assertAlmostEqual(model.body("left_foot").mass[0], MASS)
        self.assertFalse(np.any(fixture.data.qfrc_applied))
        self.assertFalse(np.any(fixture.data.xfrc_applied))

    def test_fresh_measurement_has_no_stale_support_or_grip_evaluation(self):
        fixture = make_foot_fixture(FOOT_CASES[0])
        for _ in range(250):
            mujoco.mj_step(fixture.model, fixture.data)
        self.assertTrue(shoe_measurement(fixture)["supporting"])
        fixture.data.qpos[2] += .5
        qpos, qvel, time = fixture.data.qpos.copy(), fixture.data.qvel.copy(), fixture.data.time
        with patch("boulder_v1.contact.GripController.evaluate", side_effect=AssertionError("feet are not grips")):
            states = _feet(fixture.model, fixture.data, fixture.scene)
        state = states[Limb.LEFT_FOOT]
        self.assertFalse(state.contacting)
        self.assertFalse(state.supporting)
        self.assertEqual(state.normal_force, 0.)
        self.assertEqual(state.contacts, ())
        np.testing.assert_array_equal(fixture.data.qpos, qpos)
        np.testing.assert_array_equal(fixture.data.qvel, qvel)
        self.assertEqual(fixture.data.time, time)


class HandMechanicsTests(unittest.TestCase):
    def test_physical_capture_gates_and_no_force_bypass(self):
        for dt in (.002, .001):
            result = run_capture_gates(dt)
            self.assertTrue(result["contact_acceptance"], result)
            self.assertTrue(any(c["gap_m"] == .001 and c["acquired"] for c in result["cases"]))
            self.assertTrue(any(c["speed_m_s"] == .05001 and not c["acquired"] for c in result["cases"]))
            for case in result["cases"]:
                self.assertTrue(math.isfinite(case["pre_capture"]["kinetic_energy_J"]))
                for event in case["capture_events"]:
                    self.assertAlmostEqual(event["kinetic_energy_J"], case["pre_capture"]["kinetic_energy_J"], places=12)

    def test_bound_capacity_and_identical_independent_foot_mechanics(self):
        weak, strong = make_hand_fixture(capacity=70.), make_hand_fixture(capacity=150.)
        for field in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_gravcomp",
                      "geom_size", "geom_pos", "geom_type", "geom_friction", "pair_friction", "eq_solref", "eq_solimp"):
            np.testing.assert_array_equal(getattr(weak.model, field), getattr(strong.model, field))
        for dt in (.002, .001):
            results = [run_hand_case(dt, capacity) for capacity in (70., 150.)]
            for result in results:
                self.assertTrue(result["contact_acceptance"], result["checks"])
                self.assertLessEqual(result["max_applied_load_N"], result["capacity_N"] + 1e-9)
                self.assertTrue(math.isfinite(result["max_kinetic_energy_J"]))
                self.assertEqual(result["model_proof"]["foot_equalities"], 0)
                self.assertTrue(result["capture_events"])
                self.assertTrue(all(math.isfinite(e["peak_window_reaction_N"]) for e in result["capture_events"]))
                self.assertLessEqual(result["max_held_penetration_m"], .001)
                json.dumps(result, allow_nan=False)
            self.assertGreater(results[0]["weak_rejected_load_N"], 70.)
            self.assertEqual(results[0]["fixed_field_max_applied_load_N"], 0.)
            self.assertGreater(results[1]["fixed_field_max_applied_load_N"], 99.)
            self.assertAlmostEqual(results[0]["shoe_normal_mean_N"], results[1]["shoe_normal_mean_N"], delta=1e-7)
            # A demand injected before 0.4 s must be guarded during settling too.
            settling = make_hand_fixture(dt, capacity=70., gap=0., speed=0.)
            manager, model, data = settling.manager, settling.model, settling.data
            self.assertTrue(manager.attach(Limb.LEFT_HAND, settling.scene.region("left_hand")))
            manager.synchronize_from_live()
            for step in range(round(.04 / dt)):
                if step >= round(.01 / dt):
                    data.xfrc_applied[model.body("left_hand").id, :3] = [0., -100., 0.]
                manager.evaluate_and_update()
                mujoco.mj_step(model, data)
                self.assertLessEqual(np.linalg.norm(_hand_reactions(model, data)[0]), 70. + 1e-9)
                manager.evaluate_and_update(applied_data=data)
            self.assertTrue(manager.releases)
            self.assertLess(manager.releases[0]["time_s"], .4)
            self.assertGreater(manager.releases[0]["required_load_N"], 70.)
            # An unattached palm can press a surface without acquiring a grip.
            detached = make_hand_fixture(dt, capacity=70., gap=-.0005, speed=0.)
            detached.manager.synchronize_from_live()
            detached.data.xfrc_applied[detached.model.body("left_hand").id, :3] = [0., 100., 0.]
            for _ in range(round(.06 / dt)):
                detached.manager.evaluate_and_update()
                mujoco.mj_step(detached.model, detached.data)
                detached.manager.evaluate_and_update(applied_data=detached.data)
            contacts, _ = _native_contacts(detached.model, _fresh(detached.model, detached.data), set())
            self.assertTrue(any("left_hand_geom" in c["geoms"] and np.linalg.norm(c["native_wrench"][:3]) > 5.
                                for c in contacts))
            self.assertEqual(detached.manager.get_grasp_load(Limb.LEFT_HAND), 0.)
            self.assertFalse(detached.manager.contact_snapshot().hands[Limb.LEFT_HAND].loaded)


class MixedContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = [run_mixed_case(dt) for dt in (.002, .001)]

    def test_fixed_certificate_recomputed_without_optimization_or_preload(self):
        for dt in (.002, .001):
            fixture = make_mixed_fixture(dt)
            certificate = verify_fixed_allocation(fixture)
            self.assertTrue(certificate["valid"], certificate["checks"])
            self.assertLess(certificate["max_balance_residual"], 1e-9)
            self.assertEqual(len(certificate["corner_forces"]), 8)
            self.assertEqual(len(certificate["tau_ff_Nm"]), 25)
            np.testing.assert_array_equal(fixture.data.qpos, fixture.reference)
        for result in self.results:
            self.assertEqual(result["initial"]["feet"]["left"]["Fn_N"], 0.)
            self.assertEqual(result["initial"]["feet"]["right"]["Fn_N"], 0.)
            self.assertGreater(result["initial"]["root_linear_qacc_norm"], .1)

    def test_production_human_invariants_and_only_hand_equalities(self):
        fixture = make_mixed_fixture()
        model = fixture.model
        original = mujoco.MjModel.from_xml_string(build_mjcf(_scene(), fixture.profile))
        self.assertEqual((model.nq, model.nv, model.nu, model.neq, model.nexclude), (32, 31, 25, 4, 15))
        self.assertAlmostEqual(model.body_subtreemass[model.body("climber_root").id], 78.3)
        for field in ("jnt_type", "jnt_range", "jnt_axis", "jnt_pos", "dof_damping", "dof_armature",
                      "actuator_gear", "actuator_ctrlrange", "actuator_trnid", "qpos0"):
            np.testing.assert_array_equal(getattr(original, field), getattr(model, field))
        for index in range(original.nbody):
            name = original.body(index).name
            if name == "world":
                continue
            target = model.body(name).id
            self.assertEqual(original.body(int(original.body_parentid[index])).name,
                             model.body(int(model.body_parentid[target])).name)
            for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos", "body_quat", "body_gravcomp"):
                np.testing.assert_array_equal(getattr(original, field)[index], getattr(model, field)[target])
        for index in range(original.ngeom):
            target = model.geom(original.geom(index).name).id
            for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "geom_contype", "geom_conaffinity"):
                np.testing.assert_array_equal(getattr(original, field)[index], getattr(model, field)[target])
        for index in range(model.neq):
            self.assertIn("hand", model.equality(index).name)
        fresh = _fresh(model, fixture.data)
        self.assertFalse(np.any(fresh.qfrc_applied))
        self.assertFalse(np.any(fresh.xfrc_applied))
        # Deliberately invalid full-depth ledge through the shin, not a root warp.
        regions = tuple(replace(r, position=(r.position[0], -.405, .355), half_size=(.06, .095, .03))
                        if r.id == "left_foot" else r for r in fixture.scene.contact_regions)
        bad_scene = replace(fixture.scene, contact_regions=regions)
        bad_model = mujoco.MjModel.from_xml_string(build_mjcf(bad_scene, fixture.profile))
        bad_data = mujoco.MjData(bad_model)
        bad_data.qpos[:] = fixture.reference
        bad_data.eq_active[:] = fixture.data.eq_active
        seed = bad_data.qpos.copy()
        mujoco.mj_forward(bad_model, bad_data)
        _, unexpected = _native_contacts(bad_model, bad_data, {
            frozenset((f"{side}_{part}_geom", f"geom_{side}_{part}"))
            for side in ("left", "right") for part in ("foot", "hand")})
        self.assertTrue(any("left_shin_collider" in c["geoms"] for c in unexpected), unexpected)
        negative_control = {"contact_acceptance": not unexpected, "unexpected_contacts": unexpected}
        self.assertFalse(negative_control["contact_acceptance"])
        np.testing.assert_array_equal(bad_data.qpos, seed)
        self.assertFalse(np.any(bad_data.qfrc_applied))
        self.assertFalse(np.any(bad_data.xfrc_applied))

    def test_six_seconds_contact_acceptance_not_controller_certification(self):
        for result in self.results:
            with self.subTest(dt=result["dt_s"]):
                self.assertTrue(result["contact_acceptance"], result["checks"])
                self.assertEqual(result["controller_convergence"], all(result["controller_checks"].values()))
                self.assertEqual(result["controller_checks"]["max_hinge_lt_0p2"], result["joint_max_rad_s"] < .2)
                self.assertEqual(result["controller_status"],
                                 "VALIDATED" if result["controller_convergence"] else "NOT VALIDATED")
                self.assertAlmostEqual(result["duration_s"], 6., places=9)
                self.assertEqual(result["scored_duration_s"], 5.)
                self.assertEqual(result["time_jump_count"], 0)
                self.assertFalse(result["transitions"])
                self.assertLess(result["max_torque_utilization"], 1.)
                for foot in result["feet"].values():
                    self.assertEqual(foot["support_fraction"], 1.)
                    self.assertGreater(foot["Fn_min_N"], 5.)
                    self.assertLessEqual(foot["slip_max_m_s"], .01)
                for applied in result["applied"]:
                    self.assertAlmostEqual(applied["interval_end_s"] - applied["interval_start_s"], result["dt_s"], places=12)
                    self.assertEqual(applied["force_state_time_s"], applied["interval_start_s"])
                json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
