from collections.abc import Mapping
import copy
from dataclasses import asdict, fields, is_dataclass, replace
from fractions import Fraction
import json
import math
import sys
import unittest

from boulder_v1.scene_factory import make_synthetic_scene
from boulder_v1.schema import Affordance, BoulderScene, ClimberProfile, Limb, SourceType, WallSurface


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.scene = make_synthetic_scene()
        self.wall = self.scene.walls[0]
        self.region = self.scene.contact_regions[0]

    def test_profile_rejects_nonpositive_morphology(self):
        with self.assertRaises(ValueError):
            ClimberProfile(name="bad", upper_arm_length=0.0)

    def test_wall_rejects_nonpositive_friction(self):
        with self.assertRaises(ValueError):
            WallSurface(id="bad", center=(0, 0, 0), size=(1, 1, 1), friction=0)

    def test_synthetic_scene_references_known_contacts(self):
        known = {r.id for r in self.scene.contact_regions}
        self.assertTrue(set(self.scene.start_configuration.values()).issubset(known))
        self.assertTrue(set(self.scene.goal_regions).issubset(known))

    def test_all_profile_scalars_require_finite_positive_real_numbers(self):
        invalid = (0, -1, True, False, float("nan"), float("inf"), -float("inf"),
                   "1", None, 1j, 10 ** 400)
        for field in fields(ClimberProfile):
            if field.name == "name":
                continue
            for value in invalid:
                with self.subTest(field=field.name, value=value):
                    with self.assertRaisesRegex(ValueError, field.name):
                        ClimberProfile(name="bad", **{field.name: value})

    def test_profile_has_no_anthropometric_or_backend_bounds(self):
        for field in fields(ClimberProfile):
            if field.name == "name":
                continue
            for value in (1e-300, 1e100):
                with self.subTest(field=field.name, value=value):
                    profile = ClimberProfile(name="generic", **{field.name: value})
                    self.assertEqual(getattr(profile, field.name), value)
        self.assertEqual(ClimberProfile(name="mobile", rom_scale=1.3).rom_scale, 1.3)

    def test_real_inputs_and_reserved_power_serialize_as_finite_numbers(self):
        profile = ClimberProfile(name="real", mass_scale=Fraction(1, 2), power_scale=2)
        payload = profile.to_dict()
        self.assertEqual(payload["mass_scale"], 0.5)
        self.assertEqual(payload["power_scale"], 2.0)
        self.assertTrue(all(math.isfinite(value) and value > 0
                            for name, value in payload.items() if name != "name"))
        json.dumps(payload, allow_nan=False)

    def test_identifiers_and_coordinate_system_require_nonempty_strings(self):
        for value in ("", " \t\n", None, 7):
            for kind in ("profile", "wall", "region", "coordinates"):
                with self.subTest(kind=kind, value=value):
                    with self.assertRaises(ValueError):
                        if kind == "profile":
                            ClimberProfile(name=value)
                        elif kind == "wall":
                            replace(self.wall, id=value)
                        elif kind == "region":
                            replace(self.region, id=value)
                        else:
                            replace(self.scene, coordinate_system=value)

    def test_all_vectors_require_three_finite_real_components(self):
        invalid = (None, "123", {"x": 1, "y": 2, "z": 3}, {1, 2, 3},
                   (), (1, 2), (1, 2, 3, 4))
        for owner, names in ((self.wall, ("center", "size", "normal")),
                             (self.region, ("position", "normal", "half_size"))):
            for name in names:
                for value in invalid:
                    if name == "half_size" and value is None:
                        continue
                    with self.subTest(owner=type(owner).__name__, field=name, value=value):
                        with self.assertRaises(ValueError):
                            replace(owner, **{name: value})
                for component in (True, False, "1", None, 1j, float("nan"),
                                  float("inf"), -float("inf"), 10 ** 400):
                    for index in range(3):
                        value = [1, 2, 3]
                        value[index] = component
                        with self.subTest(owner=type(owner).__name__, field=name,
                                          index=index, component=component):
                            with self.assertRaises(ValueError):
                                replace(owner, **{name: value})

    def test_wall_size_components_must_be_positive(self):
        for index in range(3):
            for value in (0, -1):
                size = [1, 2, 3]
                size[index] = value
                with self.subTest(index=index, value=value):
                    with self.assertRaisesRegex(ValueError, "wall size"):
                        replace(self.wall, size=size)
        self.assertEqual(replace(self.wall, size=[1e-300, 2, 3]).size, (1e-300, 2, 3))

    def test_optional_box_half_size_is_positive_defensive_and_serializable(self):
        self.assertIsNone(self.region.half_size)
        self.assertIsNone(self.region.to_dict()["half_size"])
        values = [.06, .05, .03]
        box = replace(self.region, half_size=values)
        values.clear()
        self.assertEqual(box.half_size, (.06, .05, .03))
        self.assertEqual(box.to_dict()["half_size"], (.06, .05, .03))
        self.assertEqual(json.loads(json.dumps(box.to_dict(), allow_nan=False))["half_size"], [.06, .05, .03])
        for index in range(3):
            for value in (0., -1.):
                size = [1., 2., 3.]
                size[index] = value
                with self.subTest(index=index, value=value), self.assertRaisesRegex(ValueError, "half_size"):
                    replace(self.region, half_size=size)

    def test_wall_contact_and_scene_scalars_require_finite_positive_reals(self):
        for owner, names in ((self.wall, ("friction",)),
                             (self.region, ("friction", "radius", "grip_quality")),
                             (self.scene, ("scale",))):
            for name in names:
                for value in (0, -1, True, False, "1", None, 1j, float("nan"),
                              float("inf"), -float("inf"), 10 ** 400):
                    with self.subTest(owner=type(owner).__name__, field=name, value=value):
                        with self.assertRaisesRegex(ValueError, name):
                            replace(owner, **{name: value})
                self.assertEqual(getattr(replace(owner, **{name: 1e-300}), name), 1e-300)

    def test_normals_are_normalized_without_overflow_or_underflow(self):
        normals = ((3, 4, 0), (1e308, 1e308, 0), (1e-300, -2e-300, 0),
                   (5e-324, 5e-324, 5e-324))
        for owner in (self.wall, self.region):
            for normal in normals:
                with self.subTest(owner=type(owner).__name__, normal=normal):
                    actual = replace(owner, normal=list(normal)).normal
                    self.assertIsInstance(actual, tuple)
                    self.assertAlmostEqual(math.hypot(*actual), 1.0)
                    scale = max(abs(value) for value in normal)
                    expected_norm = math.hypot(*(value / scale for value in normal))
                    for value, expected in zip(actual, normal):
                        self.assertAlmostEqual(value, (expected / scale) / expected_norm)
        self.assertEqual(replace(self.wall, normal=[0, -2, 0]).normal, (0, -1, 0))

    def test_normals_reject_zero_and_overflowed_norms(self):
        for owner in (self.wall, self.region):
            for normal in ((0, 0, 0), (sys.float_info.max, sys.float_info.max, 0)):
                with self.subTest(owner=type(owner).__name__, normal=normal):
                    with self.assertRaisesRegex(ValueError, "normal.*finite nonzero norm"):
                        replace(owner, normal=normal)

    def test_source_and_affordance_values_must_be_enums(self):
        for source in ("HOLD", "unknown", None, 1):
            with self.subTest(source=source):
                with self.assertRaisesRegex(ValueError, "SourceType"):
                    replace(self.region, source_type=source)
        for affordances in (None, "GRASP", (), {"GRASP"}, [Affordance.GRASP, "STEP"],
                            [1], [[]], {Affordance.GRASP: True}):
            with self.subTest(affordances=affordances):
                with self.assertRaises(ValueError):
                    replace(self.region, affordances=affordances)

    def test_generic_source_descriptors_and_backend_inputs_remain_valid(self):
        for source in SourceType:
            region = replace(self.region, source_type=source)
            self.assertEqual(region.to_dict()["source_type"], source.value)
        wall = replace(self.wall, normal=(1, 0, 0))
        scene = replace(self.scene, scale=2, walls=[wall])
        self.assertEqual(scene.scale, 2)
        self.assertEqual(scene.walls[0].normal, (1, 0, 0))
        patch = replace(self.region, source_type=SourceType.WALL,
                        affordances={Affordance.PRESS, Affordance.SMEAR})
        self.assertEqual(patch.affordances, frozenset({Affordance.PRESS, Affordance.SMEAR}))

    def test_vector_and_affordance_inputs_are_defensive_snapshots(self):
        center, size, normal = [0, 1, 2], [1, 2, 3], [0, -2, 0]
        wall = replace(self.wall, center=center, size=size, normal=normal)
        position = [3, 2, 1]
        affordances = {Affordance.GRASP, Affordance.STEP}
        region = replace(self.region, position=position, normal=normal, affordances=affordances)
        for values in (center, size, normal, position):
            values.clear()
        affordances.clear()
        self.assertEqual(wall.center, (0, 1, 2))
        self.assertEqual(wall.size, (1, 2, 3))
        self.assertEqual(wall.normal, (0, -1, 0))
        self.assertEqual(region.position, (3, 2, 1))
        self.assertEqual(region.normal, (0, -1, 0))
        self.assertEqual(region.affordances, frozenset({Affordance.GRASP, Affordance.STEP}))

    def test_scene_collection_and_mapping_inputs_are_defensive_snapshots(self):
        walls = list(self.scene.walls)
        regions = list(self.scene.contact_regions)
        goals = list(self.scene.goal_regions)
        starts = dict(self.scene.start_configuration)
        metadata = dict(self.scene.metadata)
        scene = replace(self.scene, walls=walls, contact_regions=regions,
                        goal_regions=goals, start_configuration=starts, metadata=metadata)
        for values in (walls, regions, goals, starts, metadata):
            values.clear()
        self.assertEqual(scene, self.scene)
        self.assertIsInstance(scene.walls, tuple)
        self.assertIsInstance(scene.contact_regions, tuple)
        self.assertIsInstance(scene.goal_regions, tuple)
        self.assertIsInstance(scene.start_configuration, Mapping)
        self.assertIsInstance(scene.metadata, Mapping)

    def test_set_collection_inputs_are_normalized(self):
        scene = BoulderScene("world", 1, {self.wall}, {self.region},
                             goal_regions={self.region.id})
        self.assertEqual(scene.walls, (self.wall,))
        self.assertEqual(scene.contact_regions, (self.region,))
        self.assertEqual(scene.goal_regions, (self.region.id,))

    def test_scene_collection_elements_are_typed(self):
        cases = (("walls", [self.region]), ("contact_regions", [self.wall]),
                 ("walls", None), ("contact_regions", "H1"),
                 ("goal_regions", "TOP"), ("goal_regions", [None]),
                 ("goal_regions", [1]), ("goal_regions", [""]))
        for name, value in cases:
            with self.subTest(field=name, value=value):
                with self.assertRaises(ValueError):
                    replace(self.scene, **{name: value})

    def test_scene_mappings_require_typed_immutable_contents(self):
        for value in (None, [], [(Limb.LEFT_HAND, "H3")], "H3"):
            with self.subTest(starts=value):
                with self.assertRaisesRegex(ValueError, "start_configuration.*mapping"):
                    replace(self.scene, start_configuration=value)
        for key in ("LEFT_HAND", 1, None):
            with self.subTest(limb=key):
                with self.assertRaisesRegex(ValueError, "Limb"):
                    replace(self.scene, start_configuration={key: "H3"})
        for value in (None, [], 1, "", " "):
            with self.subTest(region_id=value):
                with self.assertRaisesRegex(ValueError, "region id"):
                    replace(self.scene, start_configuration={Limb.LEFT_HAND: value})
        for value in (None, [], [("name", "scene")], "scene"):
            with self.subTest(metadata=value):
                with self.assertRaisesRegex(ValueError, "metadata.*mapping"):
                    replace(self.scene, metadata=value)
        for metadata in ({1: "scene"}, {"name": 1}, {"name": []}, {"name": {}},
                         {"name": None}):
            with self.subTest(metadata=metadata):
                with self.assertRaisesRegex(ValueError, "metadata.*strings"):
                    replace(self.scene, metadata=metadata)
        self.assertEqual(replace(self.scene, metadata={"note": ""}).metadata, {"note": ""})

    def test_scene_rejects_duplicate_ids_and_unknown_references(self):
        cases = (("walls", (*self.scene.walls, self.wall)),
                 ("contact_regions", (*self.scene.contact_regions, self.region)),
                 ("start_configuration", {Limb.LEFT_HAND: "missing"}),
                 ("goal_regions", ["missing"]))
        for name, value in cases:
            with self.subTest(field=name):
                with self.assertRaises(ValueError):
                    replace(self.scene, **{name: value})
        self.assertEqual(self.scene.region("H1"), self.region)
        with self.assertRaises(KeyError):
            self.scene.region("missing")

    def test_scene_mappings_are_immutable_and_mapping_compatible(self):
        self.assertEqual(self.scene.start_configuration, dict(self.scene.start_configuration))
        self.assertEqual(dict(self.scene.start_configuration), self.scene.start_configuration)
        self.assertEqual(list(self.scene.metadata), ["name", "purpose"])
        self.assertEqual(self.scene.metadata.get("missing", "fallback"), "fallback")
        with self.assertRaises(KeyError):
            self.scene.metadata["missing"]
        for mapping in (self.scene.start_configuration, self.scene.metadata):
            with self.subTest(mapping=mapping):
                with self.assertRaises(TypeError):
                    mapping["new"] = "value"
                with self.assertRaises(TypeError):
                    del mapping[next(iter(mapping))]
                with self.assertRaises(TypeError):
                    mapping._items = ()
                with self.assertRaises(TypeError):
                    del mapping._items
                with self.assertRaises(TypeError):
                    mapping.__init__({})
                for name in ("clear", "update", "pop", "popitem", "setdefault"):
                    self.assertFalse(hasattr(mapping, name))

    def test_deepcopy_and_asdict_preserve_scene_and_immutable_maps(self):
        copied = copy.deepcopy(self.scene)
        self.assertIsNot(copied, self.scene)
        self.assertEqual(copied, self.scene)
        for mapping in (copied.start_configuration, copied.metadata):
            self.assertIs(copy.copy(mapping), mapping)
            self.assertIs(copy.deepcopy(mapping), mapping)
            self.assertFalse(is_dataclass(mapping))
            with self.assertRaises(TypeError):
                mapping["new"] = "value"
        converted = asdict(self.scene)
        self.assertIsInstance(converted, dict)
        self.assertEqual(converted["start_configuration"], self.scene.start_configuration)
        self.assertEqual(converted["metadata"], self.scene.metadata)

    def test_replace_revalidates_and_preserves_immutable_scene_maps(self):
        unchanged = replace(self.scene)
        self.assertIsNot(unchanged, self.scene)
        self.assertEqual(unchanged, self.scene)
        starts = {Limb.LEFT_HAND: "H3"}
        metadata = {"purpose": "replacement"}
        changed = replace(self.scene, start_configuration=starts, metadata=metadata)
        starts.clear()
        metadata.clear()
        self.assertEqual(changed.start_configuration, {Limb.LEFT_HAND: "H3"})
        self.assertEqual(changed.metadata, {"purpose": "replacement"})
        self.assertEqual(len(self.scene.start_configuration), 4)
        with self.assertRaises(TypeError):
            changed.metadata["purpose"] = "mutated"
        with self.assertRaises(ValueError):
            replace(changed, scale=float("nan"))

    def test_empty_scene_defaults_remain_valid_and_immutable(self):
        scene = BoulderScene("world", 1, [], [])
        self.assertEqual(scene.start_configuration, {})
        self.assertEqual(scene.metadata, {})
        self.assertEqual(scene.goal_regions, ())
        with self.assertRaises(TypeError):
            scene.metadata["new"] = "value"

    def test_to_dict_preserves_shape_finiteness_and_detached_containers(self):
        wall = replace(self.wall, id="wall", center=[0, 0, 0], size=[1, 2, 3], friction=0.9)
        region = replace(self.region, id="hold", position=[1, 0, 2], friction=0.8,
                         radius=1e-5, grip_quality=1.1,
                         affordances={Affordance.STEP, Affordance.GRASP})
        scene = BoulderScene("world", 1, [wall], [region], {Limb.LEFT_HAND: "hold"},
                             ["hold"], {"note": ""})
        expected = {
            "coordinate_system": "world",
            "scale": 1.0,
            "walls": [{"id": "wall", "center": (0.0, 0.0, 0.0), "size": (1.0, 2.0, 3.0),
                       "normal": (0.0, -1.0, 0.0), "friction": 0.9}],
            "contact_regions": [{"id": "hold", "source_type": "HOLD", "position": (1.0, 0.0, 2.0),
                                 "normal": (0.0, -1.0, 0.0), "friction": 0.8,
                                 "affordances": ["GRASP", "STEP"], "radius": 1e-5,
                                 "grip_quality": 1.1, "half_size": None}],
            "start_configuration": {"LEFT_HAND": "hold"},
            "goal_regions": ["hold"],
            "metadata": {"note": ""},
        }
        payload = scene.to_dict()
        self.assertEqual(payload, expected)
        self.assertEqual(json.loads(json.dumps(payload, allow_nan=False))["contact_regions"][0]["radius"], 1e-5)
        payload["walls"][0]["center"] = (9, 9, 9)
        payload["contact_regions"][0]["affordances"].clear()
        payload["start_configuration"].clear()
        payload["goal_regions"].clear()
        payload["metadata"]["note"] = "changed"
        self.assertEqual(scene.to_dict(), expected)


if __name__ == "__main__":
    unittest.main()
