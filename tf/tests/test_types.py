from copy import copy, deepcopy
from unittest import TestCase

from tf import types
from tf.types import Unknown


class TypesTest(TestCase):
    def test_unknown_repr(self):
        # This makes it very easy to copy/paste from output to tests
        self.assertEqual(repr(Unknown), "Unknown")

    def test_unknown_copy_behavior(self):
        # Unknown is a singleton
        self.assertIs(Unknown, Unknown)
        self.assertEqual(Unknown, Unknown)
        self.assertEqual(id(Unknown), id(Unknown))
        self.assertIs(copy(Unknown), Unknown)
        self.assertIs(deepcopy(Unknown), Unknown)

    def test_integer_encoding(self):
        integer = types.Number()

        self.assertEqual(integer.encode(123), 123)
        self.assertEqual(integer.decode(123), 123)

    def test_string_encoding(self):
        string = types.String()

        self.assertEqual(string.encode("hello"), "hello")
        self.assertEqual(string.decode("hello"), "hello")

    def test_bool_encoding(self):
        boolean = types.Bool()

        self.assertEqual(boolean.encode(True), True)
        self.assertEqual(boolean.decode(True), True)

    def test_json_encoding(self):
        json = types.NormalizedJson()

        self.assertEqual(json.encode({"key": "value"}), '{"key": "value"}')
        self.assertEqual(json.decode('{"key":"value"}'), {"key": "value"})

    def test_json_encoding_normalization(self):
        json = types.NormalizedJson()

        self.assertEqual(
            json.encode({"key1": "value1", "key2": "value2"}),
            json.encode({"key2": "value2", "key1": "value1"}),
        )

    def test_json_decode_error(self):
        json = types.NormalizedJson()
        with self.assertRaises(ValueError) as e:
            json.decode("not valid json")

        self.assertEqual(str(e.exception), "Error parsing JSON: Expecting value: line 1 column 1 (char 0)")

    def test_list_encode(self):
        list_type = types.List(types.Number())

        self.assertEqual(list_type.encode(None), None)
        self.assertEqual(list_type.decode(None), None)

        self.assertEqual(list_type.encode(Unknown), Unknown)
        self.assertEqual(list_type.decode(Unknown), Unknown)

        self.assertEqual(list_type.encode([1, 2, 3]), [1, 2, 3])
        self.assertEqual(list_type.decode([1, 2, 3]), [1, 2, 3])

    def test_list_type_encoding(self):
        table = (
            (types.List(types.Number()), "list of int", b'["list","number"]'),
            (types.List(types.String()), "list of string", b'["list","string"]'),
            (types.List(types.Bool()), "list of bool", b'["list","bool"]'),
            (types.List(types.Set(types.Bool())), "list of set of bool", b'["list",["set","bool"]]'),
        )

        for set_type, test_name, expected in table:
            with self.subTest(test_name):
                self.assertEqual(set_type.tf_type(), expected)

    def test_list_equality(self):
        list_type = types.List(types.Number())

        self.assertTrue(list_type.semantically_equal([1, 2, 3], [1, 2, 3]))
        self.assertFalse(list_type.semantically_equal([1, 2, 3], [1, 2, 4]))
        self.assertFalse(list_type.semantically_equal([1, 2, 3], [1, 2]))

    def test_set_type_encoding(self):
        table = (
            (types.Set(types.Number()), "set of int", b'["set","number"]'),
            (types.Set(types.String()), "set of string", b'["set","string"]'),
            (types.Set(types.Bool()), "set of bool", b'["set","bool"]'),
            (types.Set(types.Set(types.Bool())), "set of set of bool", b'["set",["set","bool"]]'),
        )

        for set_type, test_name, expected in table:
            with self.subTest(test_name):
                self.assertEqual(set_type.tf_type(), expected)

    def test_set_semantic_equality(self):
        set_type = types.Set(types.Number())

        self.assertTrue(set_type.semantically_equal([1, 2, 3], [1, 2, 3]))
        self.assertFalse(set_type.semantically_equal([1, 2, 3], [1, 2, 4]))
        self.assertFalse(set_type.semantically_equal([1, 2, 3], [1, 2]))
        self.assertTrue(set_type.semantically_equal([1, 2, 3], [3, 2, 1]))
        self.assertTrue(set_type.semantically_equal(set(), set()))
        self.assertTrue(set_type.semantically_equal(None, None))
        self.assertTrue(set_type.semantically_equal(Unknown, Unknown))

    def test_set_semantic_equality_with_dicts(self):
        # Test with dict elements to ensure string conversion works
        set_type = types.Set(types.NormalizedJson())

        # Same dicts in same order
        self.assertTrue(set_type.semantically_equal([{"a": 1}, {"b": 2}], [{"a": 1}, {"b": 2}]))
        # Same dicts in different order
        self.assertTrue(set_type.semantically_equal([{"a": 1}, {"b": 2}], [{"b": 2}, {"a": 1}]))
        # Different dicts
        self.assertFalse(set_type.semantically_equal([{"a": 1}, {"b": 2}], [{"a": 1}, {"c": 3}]))
        # Different lengths
        self.assertFalse(set_type.semantically_equal([{"a": 1}], [{"a": 1}, {"b": 2}]))
        # Empty lists
        self.assertTrue(set_type.semantically_equal([], []))

    def test_map_encode(self):
        map_type = types.Map(types.String())

        self.assertIsNone(map_type.encode(None))
        self.assertIsNone(map_type.decode(None))
        self.assertIs(map_type.encode(Unknown), Unknown)
        self.assertIs(map_type.decode(Unknown), Unknown)

        self.assertEqual(map_type.encode({"a": "x", "b": "y"}), {"a": "x", "b": "y"})
        self.assertEqual(map_type.decode({"a": "x", "b": "y"}), {"a": "x", "b": "y"})

    def test_map_encode_delegates_to_value_type(self):
        """Map.encode/decode applies the value type to each value."""
        from tf.types import NormalizedJson

        map_type = types.Map(NormalizedJson())
        self.assertEqual(map_type.encode({"k": {"a": 1}}), {"k": '{"a": 1}'})
        self.assertEqual(map_type.decode({"k": '{"a": 1}'}), {"k": {"a": 1}})

    def test_map_type_encoding(self):
        table = (
            (types.Map(types.String()), "map of string", b'["map","string"]'),
            (types.Map(types.Number()), "map of number", b'["map","number"]'),
            (types.Map(types.Bool()), "map of bool", b'["map","bool"]'),
            (types.Map(types.List(types.String())), "map of list of string", b'["map",["list","string"]]'),
        )
        for map_type, test_name, expected in table:
            with self.subTest(test_name):
                self.assertEqual(map_type.tf_type(), expected)

    def test_map_semantic_equality(self):
        map_type = types.Map(types.String())

        self.assertTrue(map_type.semantically_equal({"a": "x"}, {"a": "x"}))
        self.assertFalse(map_type.semantically_equal({"a": "x"}, {"a": "y"}))
        self.assertFalse(map_type.semantically_equal({"a": "x"}, {"b": "x"}))
        self.assertTrue(map_type.semantically_equal(None, None))
        self.assertTrue(map_type.semantically_equal(Unknown, Unknown))
        self.assertTrue(map_type.semantically_equal({}, {}))
