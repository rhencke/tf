from unittest import TestCase

from tf.gen import tfplugin_pb2 as pb
from tf.schema import Attribute, Schema, TextFormat
from tf.types import String


class SchemaTest(TestCase):
    def test_encode_empty(self):
        schema = Schema()
        self.assertEqual(
            schema.to_pb(),
            pb.Schema(block=pb.Schema.Block(attributes=[])),
        )

    def test_version_encode(self):
        schema = Schema(version=9)
        self.assertEqual(
            schema.to_pb(),
            pb.Schema(
                version=9,
                block=pb.Schema.Block(attributes=[]),
            ),
        )

    def test_encode_description(self):
        schema = Schema(
            description="This is a test schema",
        )
        self.assertEqual(
            schema.to_pb(),
            pb.Schema(
                block=pb.Schema.Block(
                    description="This is a test schema",
                    description_kind="MARKDOWN",
                ),
            ),
        )

    def test_encode_description_plain(self):
        schema = Schema(
            description="This is a test schema",
            description_kind=TextFormat.Plain,
        )
        self.assertEqual(
            schema.to_pb(),
            pb.Schema(
                block=pb.Schema.Block(
                    description="This is a test schema",
                    description_kind="PLAIN",
                ),
            ),
        )

    def test_encode_deprecated(self):
        schema = Schema(
            deprecated=True,
        )
        self.assertEqual(
            schema.to_pb(),
            pb.Schema(
                block=pb.Schema.Block(
                    deprecated=True,
                ),
            ),
        )


class WriteOnlyAttributeTest(TestCase):
    def test_write_only_false_by_default(self):
        attr = Attribute("name", String(), required=True)
        self.assertFalse(attr.to_pb().write_only)

    def test_write_only_serialized_in_schema(self):
        attr = Attribute("secret", String(), required=True, write_only=True)
        self.assertTrue(attr.to_pb().write_only)

    def test_write_only_false_serializes_as_false(self):
        # proto3 non-optional scalar bool has no field presence — we can only assert
        # the value is False, not that the field was "omitted"
        attr = Attribute("name", String(), optional=True, write_only=False)
        pb_attr = attr.to_pb()
        self.assertFalse(pb_attr.write_only)

    def test_write_only_combined_with_required(self):
        # write_only is documented as only valid with required or optional
        attr = Attribute("token", String(), required=True, write_only=True)
        pb_attr = attr.to_pb()
        self.assertTrue(pb_attr.required)
        self.assertTrue(pb_attr.write_only)

    def test_write_only_combined_with_optional(self):
        attr = Attribute("token", String(), optional=True, write_only=True)
        pb_attr = attr.to_pb()
        self.assertTrue(pb_attr.optional)
        self.assertTrue(pb_attr.write_only)

    def test_write_only_with_computed_raises(self):
        with self.assertRaises(ValueError):
            Attribute("token", String(), computed=True, write_only=True)

    def test_write_only_without_required_or_optional_raises(self):
        with self.assertRaises(ValueError):
            Attribute("token", String(), write_only=True)
