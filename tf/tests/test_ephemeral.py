"""Tests for ephemeral resource support (proto 6.9)."""

from typing import Optional
from unittest import TestCase
from unittest.mock import MagicMock, patch

import grpc

from tf import schema, types
from tf.gen import tfplugin_pb2 as pb
from tf.iface import EphemeralResource
from tf.provider import Provider, ProviderServicer
from tf.schema import Schema
from tf.utils import Diagnostics, read_dynamic_value, to_dynamic_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _PingEphemeral(EphemeralResource):
    _schema = Schema(attributes=[schema.Attribute("host_id", types.String(), required=True)])

    def __init__(self, provider):
        self._provider = provider

    @classmethod
    def get_name(cls) -> str:
        return "ping"

    @classmethod
    def get_schema(cls) -> Optional[Schema]:
        return cls._schema

    def validate(self, diags: Diagnostics, config: dict):
        if not config.get("host_id"):
            diags.add_error("host_id required", "")

    def open(self, diags: Diagnostics, config: dict) -> dict:
        return {"result": "pong"}

    def close(self, diags: Diagnostics, private: bytes):
        pass


class _EphemeralProvider(Provider):
    def full_name(self) -> str:
        return "tf.example.com/example/example"

    def get_model_prefix(self) -> str:
        return "test_"

    def get_provider_schema(self, diags: Diagnostics) -> schema.Schema:
        return schema.Schema()

    def validate_config(self, diags: Diagnostics, config: dict):
        pass

    def configure_provider(self, diags: Diagnostics, config: dict):
        pass

    def get_data_sources(self):
        return []

    def get_resources(self):
        return []

    def get_ephemeral_resources(self):
        return [_PingEphemeral]


def _make_servicer(provider=None):
    return ProviderServicer(provider or _EphemeralProvider())


# ---------------------------------------------------------------------------
# iface.EphemeralResource
# ---------------------------------------------------------------------------


class TestEphemeralResourceInterface(TestCase):
    def test_get_name_implemented(self):
        self.assertEqual(_PingEphemeral.get_name(), "ping")

    def test_get_schema_returns_schema(self):
        schema = _PingEphemeral.get_schema()
        self.assertIsNotNone(schema)

    def test_validate_adds_error_for_missing_host_id(self):
        diags = Diagnostics()
        inst = _PingEphemeral(MagicMock())
        inst.validate(diags, {})
        self.assertTrue(diags.has_errors())

    def test_open_returns_dict(self):
        diags = Diagnostics()
        inst = _PingEphemeral(MagicMock())
        result = inst.open(diags, {"host_id": "h1"})
        self.assertEqual(result, {"result": "pong"})

    def test_close_is_noop(self):
        diags = Diagnostics()
        inst = _PingEphemeral(MagicMock())
        inst.close(diags, b"")  # should not raise


# ---------------------------------------------------------------------------
# Provider.get_ephemeral_resources() registered and mapped with prefix
# ---------------------------------------------------------------------------


class TestEphemeralClsMap(TestCase):
    def test_ephemeral_cls_map_includes_prefix(self):
        svc = _make_servicer()
        cls_map = svc._load_ephemeral_cls_map()
        self.assertIn("test_ping", cls_map)
        self.assertIs(cls_map["test_ping"], _PingEphemeral)

    def test_ephemeral_cls_map_empty_when_not_provided(self):
        class _NoEphemeral(Provider):
            def full_name(self): return "x"
            def get_model_prefix(self): return "test_"
            def get_provider_schema(self, d): return Schema()
            def validate_config(self, d, c): pass
            def configure_provider(self, d, c): pass
            def get_data_sources(self): return []
            def get_resources(self): return []

        svc = _make_servicer(_NoEphemeral())
        self.assertEqual(svc._load_ephemeral_cls_map(), {})

    def test_ephemeral_cls_map_cached(self):
        svc = _make_servicer()
        map1 = svc._load_ephemeral_cls_map()
        map2 = svc._load_ephemeral_cls_map()
        self.assertIs(map1, map2)


# ---------------------------------------------------------------------------
# GetProviderSchema includes ephemeral_resource_schemas
# ---------------------------------------------------------------------------


class TestGetProviderSchemaEphemeral(TestCase):
    def test_ephemeral_schema_in_get_provider_schema(self):
        svc = _make_servicer()
        resp = svc.GetProviderSchema(pb.GetProviderSchema.Request(), MagicMock())
        self.assertIn("test_ping", resp.ephemeral_resource_schemas)

    def test_no_ephemeral_resources_omits_field(self):
        class _NoEphemeral(Provider):
            def full_name(self): return "x"
            def get_model_prefix(self): return "test_"
            def get_provider_schema(self, d): return Schema()
            def validate_config(self, d, c): pass
            def configure_provider(self, d, c): pass
            def get_data_sources(self): return []
            def get_resources(self): return []

        svc = _make_servicer(_NoEphemeral())
        resp = svc.GetProviderSchema(pb.GetProviderSchema.Request(), MagicMock())
        self.assertEqual(len(resp.ephemeral_resource_schemas), 0)


# ---------------------------------------------------------------------------
# ValidateEphemeralResourceConfig
# ---------------------------------------------------------------------------


class TestValidateEphemeralResourceConfig(TestCase):
    def test_valid_config_no_errors(self):
        svc = _make_servicer()
        config_dv = to_dynamic_value({"host_id": "h1"})
        req = pb.ValidateEphemeralResourceConfig.Request(
            type_name="test_ping",
            config=pb.DynamicValue(msgpack=config_dv.msgpack),
        )
        resp = svc.ValidateEphemeralResourceConfig(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)

    def test_invalid_config_returns_error(self):
        svc = _make_servicer()
        config_dv = to_dynamic_value({})
        req = pb.ValidateEphemeralResourceConfig.Request(
            type_name="test_ping",
            config=pb.DynamicValue(msgpack=config_dv.msgpack),
        )
        resp = svc.ValidateEphemeralResourceConfig(req, MagicMock())
        self.assertGreater(len(resp.diagnostics), 0)

    def test_unknown_type_sets_grpc_not_found(self):
        svc = _make_servicer()
        context = MagicMock()
        req = pb.ValidateEphemeralResourceConfig.Request(type_name="test_missing")
        svc.ValidateEphemeralResourceConfig(req, context)
        context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)


# ---------------------------------------------------------------------------
# OpenEphemeralResource
# ---------------------------------------------------------------------------


class TestOpenEphemeralResource(TestCase):
    def test_open_returns_result(self):
        svc = _make_servicer()
        config_dv = to_dynamic_value({"host_id": "h1"})
        req = pb.OpenEphemeralResource.Request(
            type_name="test_ping",
            config=pb.DynamicValue(msgpack=config_dv.msgpack),
        )
        resp = svc.OpenEphemeralResource(req, MagicMock())
        result = read_dynamic_value(resp.result)
        self.assertEqual(result, {"result": "pong"})

    def test_open_unknown_type_sets_grpc_not_found(self):
        svc = _make_servicer()
        context = MagicMock()
        req = pb.OpenEphemeralResource.Request(type_name="test_missing")
        svc.OpenEphemeralResource(req, context)
        context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)


# ---------------------------------------------------------------------------
# RenewEphemeralResource / CloseEphemeralResource
# ---------------------------------------------------------------------------


class TestRenewCloseEphemeral(TestCase):
    def test_renew_is_noop(self):
        svc = _make_servicer()
        req = pb.RenewEphemeralResource.Request(type_name="test_ping")
        resp = svc.RenewEphemeralResource(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)

    def test_close_calls_close(self):
        svc = _make_servicer()
        req = pb.CloseEphemeralResource.Request(type_name="test_ping", private=b"")
        resp = svc.CloseEphemeralResource(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)


# ---------------------------------------------------------------------------
# write_only Attribute
# ---------------------------------------------------------------------------


class TestWriteOnlyAttribute(TestCase):
    def test_write_only_serialized_in_schema(self):
        attr = schema.Attribute("secret", types.String(), required=True, write_only=True)
        pb_attr = attr.to_pb()
        self.assertTrue(pb_attr.write_only)

    def test_write_only_false_by_default(self):
        attr = schema.Attribute("name", types.String(), required=True)
        pb_attr = attr.to_pb()
        self.assertFalse(pb_attr.write_only)
