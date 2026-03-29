"""Tests for proto 6.9 resource identity stubs.

Resource identity is a versioned, stable identifier for a managed resource
separate from its full Terraform state.  Providers opt in by implementing
ResourceWithIdentity (exposes get_identity_schema()) and calling
ctx.set_identity() from their CRUD methods.

NOTE: As of OpenTofu 1.11, GetResourceIdentitySchemas and UpgradeResourceIdentity
satisfy the gRPC interface but panic("unimplemented") — the client never calls
them.  Full CRUD wiring (threading identity through Read/Plan/Apply/Import) is
deferred until OpenTofu ships support.  Terraform (HashiCorp) introduced full
support in v1.12.

Tests here cover:

  1. Schema layer — IdentityAttribute and IdentitySchema serialize correctly.
  2. GetResourceIdentitySchemas — includes schemas for ResourceWithIdentity
     resources and omits schemas for plain resources.
  3. UpgradeResourceIdentity — passes through identity unchanged when no
     ResourceWithUpgradeIdentity; calls upgrade_identity() when implemented;
     returns an error for unknown resource types.
  4. ctx.set_identity / ctx.current_identity API surface on context objects.
  5. Proto shape smoke tests.
"""

import json
from typing import Optional, Type
from unittest import TestCase, mock
from unittest.mock import MagicMock

from tf import schema, types
from tf.gen import tfplugin_pb2 as pb
from tf.iface import (
    Config,
    CreateContext,
    ReadContext,
    ResourceWithIdentity,
    ResourceWithUpgradeIdentity,
    State,
    UpdateContext,
    UpgradeContext,
)
from tf.provider import Provider, ProviderServicer
from tf.schema import IdentityAttribute, IdentitySchema
from tf.tests.test_capabilities import _SimpleResource
from tf.utils import Diagnostics, read_dynamic_value

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _PlainResource(_SimpleResource):
    """Resource with no identity support; reuses _SimpleResource CRUD stubs."""

    @classmethod
    def get_name(cls) -> str:
        return "plain"

    @classmethod
    def get_schema(cls) -> schema.Schema:
        return schema.Schema(attributes=[schema.Attribute("val", types.String(), required=True)])

    def plan(self, ctx, current: Optional[State], planned: State) -> Optional[State]:
        return planned


class _IdentityResource(_PlainResource, ResourceWithIdentity):
    """Resource that exposes a stable identity (key_id)."""

    @classmethod
    def get_name(cls) -> str:
        return "identified"

    @classmethod
    def get_schema(cls) -> schema.Schema:
        return schema.Schema(
            attributes=[
                schema.Attribute("val", types.String(), required=True),
                schema.Attribute("key_id", types.String(), computed=True),
            ]
        )

    @classmethod
    def get_identity_schema(cls) -> IdentitySchema:
        return IdentitySchema(
            attributes=[
                IdentityAttribute("key_id", types.String(), required_for_import=True),
            ]
        )


class _UpgradingResource(_IdentityResource, ResourceWithUpgradeIdentity):
    """Resource that can upgrade old identity data."""

    @classmethod
    def get_name(cls) -> str:
        return "upgrading"

    @classmethod
    def get_identity_schema(cls) -> IdentitySchema:
        return IdentitySchema(
            attributes=[
                IdentityAttribute("key_id", types.String(), required_for_import=True),
            ],
            version=2,
        )

    def upgrade_identity(self, ctx: UpgradeContext, version: int, old_identity: dict) -> Optional[dict]:
        # v0/v1 had "id" field; v2 renamed it to "key_id"
        if version < 2:
            return {"key_id": old_identity.get("id", "")}
        return old_identity


class _TestProvider(Provider):
    def full_name(self) -> str:
        return "tf.example.com/example/example"

    def get_model_prefix(self) -> str:
        return "test_"

    def get_provider_schema(self, diags: Diagnostics) -> schema.Schema:
        return schema.Schema()

    def validate_config(self, diags: Diagnostics, config: Config):
        pass

    def configure_provider(self, diags: Diagnostics, config: Config):
        pass

    def get_data_sources(self) -> list[Type]:
        return []

    def get_resources(self) -> list[Type]:
        return [_PlainResource, _IdentityResource, _UpgradingResource]


def _make_servicer():
    return ProviderServicer(_TestProvider())


# ---------------------------------------------------------------------------
# Schema layer
# ---------------------------------------------------------------------------


class TestIdentityAttributeToPb(TestCase):
    def test_required_for_import(self):
        attr = IdentityAttribute("key_id", types.String(), required_for_import=True)
        pb_attr = attr.to_pb()
        self.assertEqual(pb_attr.name, "key_id")
        self.assertTrue(pb_attr.required_for_import)
        self.assertFalse(pb_attr.optional_for_import)

    def test_optional_for_import(self):
        attr = IdentityAttribute("region", types.String(), optional_for_import=True)
        pb_attr = attr.to_pb()
        self.assertTrue(pb_attr.optional_for_import)
        self.assertFalse(pb_attr.required_for_import)

    def test_description_included_when_set(self):
        attr = IdentityAttribute("key_id", types.String(), required_for_import=True, description="The key")
        pb_attr = attr.to_pb()
        self.assertEqual(pb_attr.description, "The key")

    def test_description_absent_when_none(self):
        attr = IdentityAttribute("key_id", types.String(), required_for_import=True)
        pb_attr = attr.to_pb()
        self.assertEqual(pb_attr.description, "")

    def test_both_import_flags_raises(self):
        with self.assertRaises(ValueError):
            IdentityAttribute("id", types.String(), required_for_import=True, optional_for_import=True)


class TestIdentitySchemaVersion(TestCase):
    def test_default_version_is_zero(self):
        s = IdentitySchema()
        self.assertEqual(s.version, 0)

    def test_custom_version(self):
        s = IdentitySchema(version=3)
        pb_s = s.to_pb()
        self.assertEqual(pb_s.version, 3)

    def test_to_pb_includes_attributes(self):
        s = IdentitySchema(
            attributes=[
                IdentityAttribute("key_id", types.String(), required_for_import=True),
            ]
        )
        pb_s = s.to_pb()
        self.assertEqual(len(pb_s.identity_attributes), 1)
        self.assertEqual(pb_s.identity_attributes[0].name, "key_id")


# ---------------------------------------------------------------------------
# GetResourceIdentitySchemas
# ---------------------------------------------------------------------------


class TestGetResourceIdentitySchemas(TestCase):
    def test_identity_resources_included(self):
        svc = _make_servicer()
        resp = svc.GetResourceIdentitySchemas(pb.GetResourceIdentitySchemas.Request(), MagicMock())
        self.assertIn("test_identified", resp.identity_schemas)
        self.assertIn("test_upgrading", resp.identity_schemas)

    def test_plain_resource_not_included(self):
        svc = _make_servicer()
        resp = svc.GetResourceIdentitySchemas(pb.GetResourceIdentitySchemas.Request(), MagicMock())
        self.assertNotIn("test_plain", resp.identity_schemas)

    def test_schema_version_correct(self):
        svc = _make_servicer()
        resp = svc.GetResourceIdentitySchemas(pb.GetResourceIdentitySchemas.Request(), MagicMock())
        self.assertEqual(resp.identity_schemas["test_upgrading"].version, 2)

    def test_no_diagnostics(self):
        svc = _make_servicer()
        resp = svc.GetResourceIdentitySchemas(pb.GetResourceIdentitySchemas.Request(), MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)


# ---------------------------------------------------------------------------
# UpgradeResourceIdentity
# ---------------------------------------------------------------------------


class TestUpgradeResourceIdentity(TestCase):
    def test_pass_through_when_no_upgrade_mixin(self):
        svc = _make_servicer()
        old = {"key_id": "kid-old"}
        req = pb.UpgradeResourceIdentity.Request(
            type_name="test_identified",
            version=0,
            raw_identity=pb.RawState(json=json.dumps(old).encode()),
        )
        resp = svc.UpgradeResourceIdentity(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)
        identity = read_dynamic_value(resp.upgraded_identity.identity_data)
        self.assertEqual(identity["key_id"], "kid-old")

    def test_upgrade_mixin_called(self):
        svc = _make_servicer()
        old = {"id": "old-id"}  # v1 format; upgrade_identity renames to "key_id"
        req = pb.UpgradeResourceIdentity.Request(
            type_name="test_upgrading",
            version=1,
            raw_identity=pb.RawState(json=json.dumps(old).encode()),
        )
        resp = svc.UpgradeResourceIdentity(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)
        identity = read_dynamic_value(resp.upgraded_identity.identity_data)
        self.assertEqual(identity["key_id"], "old-id")

    def test_unknown_resource_type_returns_error(self):
        svc = _make_servicer()
        req = pb.UpgradeResourceIdentity.Request(
            type_name="test_nonexistent",
            version=0,
            raw_identity=pb.RawState(json=b"{}"),
        )
        resp = svc.UpgradeResourceIdentity(req, MagicMock())
        self.assertGreater(len(resp.diagnostics), 0)

    def test_empty_raw_identity_returns_empty_dict(self):
        svc = _make_servicer()
        req = pb.UpgradeResourceIdentity.Request(
            type_name="test_identified",
            version=0,
            raw_identity=pb.RawState(),
        )
        resp = svc.UpgradeResourceIdentity(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)

    def test_upgrade_identity_returns_none_yields_empty_response(self):
        """When upgrade_identity() returns None the response carries no identity data."""
        svc = _make_servicer()
        with mock.patch.object(_UpgradingResource, "upgrade_identity", return_value=None):
            req = pb.UpgradeResourceIdentity.Request(
                type_name="test_upgrading",
                version=1,
                raw_identity=pb.RawState(json=json.dumps({"id": "x"}).encode()),
            )
            resp = svc.UpgradeResourceIdentity(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)
        self.assertFalse(resp.HasField("upgraded_identity"))


# ---------------------------------------------------------------------------
# ctx.set_identity / ctx.current_identity API surface
# ---------------------------------------------------------------------------


class TestContextIdentity(TestCase):
    def test_set_identity_stores_on_context(self):
        ctx = CreateContext(Diagnostics(), "test_thing")
        self.assertIsNone(ctx._identity_out)
        ctx.set_identity({"key_id": "x"})
        self.assertEqual(ctx._identity_out, {"key_id": "x"})

    def test_current_identity_default_is_none(self):
        ctx = ReadContext(Diagnostics(), "test_thing")
        self.assertIsNone(ctx.current_identity)

    def test_current_identity_passable_as_keyword(self):
        ctx = UpdateContext(Diagnostics(), "test_thing", current_identity={"key_id": "y"})
        self.assertEqual(ctx.current_identity, {"key_id": "y"})


# ---------------------------------------------------------------------------
# Proto shape smoke tests
# ---------------------------------------------------------------------------


class TestResourceIdentityProtoShape(TestCase):
    def test_identity_schema_has_version_and_attributes(self):
        attr = pb.ResourceIdentitySchema.IdentityAttribute(
            name="id",
            description="The resource ID",
            required_for_import=True,
        )
        identity_schema = pb.ResourceIdentitySchema(version=1, identity_attributes=[attr])
        self.assertEqual(identity_schema.version, 1)
        self.assertEqual(len(identity_schema.identity_attributes), 1)
        self.assertEqual(identity_schema.identity_attributes[0].name, "id")
        self.assertTrue(identity_schema.identity_attributes[0].required_for_import)

    def test_identity_attribute_optional_for_import(self):
        attr = pb.ResourceIdentitySchema.IdentityAttribute(
            name="region",
            optional_for_import=True,
        )
        self.assertTrue(attr.optional_for_import)
        self.assertFalse(attr.required_for_import)

    def test_resource_identity_data_wraps_dynamic_value(self):
        import msgpack

        payload = msgpack.packb({"id": "abc-123"})
        data = pb.ResourceIdentityData(
            identity_data=pb.DynamicValue(msgpack=payload),
        )
        self.assertTrue(data.HasField("identity_data"))

    def test_get_resource_identity_schemas_response_accepts_schemas(self):
        attr = pb.ResourceIdentitySchema.IdentityAttribute(name="id", required_for_import=True)
        resp = pb.GetResourceIdentitySchemas.Response(
            identity_schemas={"test_thing": pb.ResourceIdentitySchema(version=0, identity_attributes=[attr])}
        )
        self.assertIn("test_thing", resp.identity_schemas)

    def test_invalid_json_returns_diagnostic_error(self):
        svc = _make_servicer()
        req = pb.UpgradeResourceIdentity.Request(
            type_name="test_identified",
            version=0,
            raw_identity=pb.RawState(json=b"{not valid json}"),
        )
        resp = svc.UpgradeResourceIdentity(req, MagicMock())
        self.assertGreater(len(resp.diagnostics), 0)
        self.assertIn("Invalid identity JSON", resp.diagnostics[0].summary)
