"""Tests for proto 6.9 ClientCapabilities support.

ClientCapabilities is advertised by Terraform on each request to signal which
optional protocol extensions it supports:

    deferral_allowed              — client can handle deferred responses
    write_only_attributes_allowed — client understands write_only attributes

The framework extracts these into ctx.client_capabilities so providers can
inspect them without parsing proto objects directly.  Basic providers can
ignore capabilities entirely — the defaults (both False) are always safe.

Tests here cover:

    1. ClientCapabilities dataclass defaults and settability.
    2. _extract_capabilities() correctly maps proto → ClientCapabilities.
    3. All RPC handlers accept requests carrying ClientCapabilities without error.
"""

from typing import Optional, Type
from unittest import TestCase
from unittest.mock import MagicMock

from tf import schema, types
from tf.gen import tfplugin_pb2 as pb
from tf.iface import (
    ClientCapabilities,
    Config,
    CreateContext,
    DeleteContext,
    ReadContext,
    ReadDataContext,
    State,
    UpdateContext,
)
from tf.provider import Provider, ProviderServicer, _extract_capabilities
from tf.utils import Diagnostics, to_dynamic_value

# ---------------------------------------------------------------------------
# Minimal provider fixtures
# ---------------------------------------------------------------------------


class _SimpleResource:
    @classmethod
    def get_name(cls) -> str:
        return "thing"

    @classmethod
    def get_schema(cls) -> schema.Schema:
        return schema.Schema(
            attributes=[
                schema.Attribute("value", types.String(), required=True),
            ]
        )

    def __init__(self, provider):
        pass

    def validate(self, diags: Diagnostics, type_name: str, config: Config):
        pass

    def create(self, ctx: CreateContext, planned: State) -> Optional[State]:
        return planned

    def read(self, ctx: ReadContext, current: State) -> Optional[State]:
        return current

    def update(self, ctx: UpdateContext, current: State, planned: State) -> Optional[State]:
        return planned

    def delete(self, ctx: DeleteContext, current: State):
        pass


class _SimpleDataSource:
    @classmethod
    def get_name(cls) -> str:
        return "info"

    @classmethod
    def get_schema(cls) -> schema.Schema:
        return schema.Schema(
            attributes=[
                schema.Attribute("value", types.String(), computed=True),
            ]
        )

    def __init__(self, provider):
        pass

    def validate(self, diags: Diagnostics, type_name: str, config: Config):
        pass

    def read(self, ctx: ReadDataContext, config: Config) -> Optional[State]:
        return {"value": "hello"}


class _CapabilitiesProvider(Provider):
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
        return [_SimpleDataSource]

    def get_resources(self) -> list[Type]:
        return [_SimpleResource]


def _make_servicer():
    return ProviderServicer(_CapabilitiesProvider())


class _ServicerTest(TestCase):
    def setUp(self):
        super().setUp()
        self.svc = _make_servicer()


def _capabilities(**kwargs) -> pb.ClientCapabilities:
    return pb.ClientCapabilities(**kwargs)


def _config_dv(value: str) -> pb.DynamicValue:
    return to_dynamic_value({"value": value})


_STATE = _config_dv("x")


def _req(cls, **kwargs):
    kwargs.setdefault("type_name", "test_thing")
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# ClientCapabilities dataclass
# ---------------------------------------------------------------------------


class TestClientCapabilitiesDataclass(TestCase):
    def test_defaults_are_false(self):
        caps = ClientCapabilities()
        self.assertFalse(caps.deferral_allowed)
        self.assertFalse(caps.write_only_attributes_allowed)

    def test_fields_settable(self):
        caps = ClientCapabilities(deferral_allowed=True, write_only_attributes_allowed=True)
        self.assertTrue(caps.deferral_allowed)
        self.assertTrue(caps.write_only_attributes_allowed)


# ---------------------------------------------------------------------------
# _extract_capabilities helper
# ---------------------------------------------------------------------------


class TestExtractCapabilities(TestCase):
    def test_missing_field_returns_defaults(self):
        # Objects without client_capabilities use defaults
        class _NoCapField:
            pass

        caps = _extract_capabilities(_NoCapField())
        self.assertFalse(caps.deferral_allowed)
        self.assertFalse(caps.write_only_attributes_allowed)

    def test_all_false_returns_defaults(self):
        req = _req(
            pb.ValidateResourceConfig.Request, config=_STATE, client_capabilities=_capabilities(deferral_allowed=False)
        )
        caps = _extract_capabilities(req)
        self.assertFalse(caps.deferral_allowed)

    def test_deferral_allowed_extracted(self):
        req = _req(
            pb.ReadResource.Request, current_state=_STATE, client_capabilities=_capabilities(deferral_allowed=True)
        )
        caps = _extract_capabilities(req)
        self.assertTrue(caps.deferral_allowed)

    def test_write_only_extracted(self):
        req = _req(
            pb.ReadResource.Request,
            current_state=_STATE,
            client_capabilities=_capabilities(write_only_attributes_allowed=True),
        )
        caps = _extract_capabilities(req)
        self.assertTrue(caps.write_only_attributes_allowed)


# ---------------------------------------------------------------------------
# RPC handlers accept ClientCapabilities without error (smoke tests)
# ---------------------------------------------------------------------------


class TestClientCapabilitiesValidateResource(_ServicerTest):
    def test_request_without_capabilities_accepted(self):
        req = _req(pb.ValidateResourceConfig.Request, config=_STATE)
        resp = self.svc.ValidateResourceConfig(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)

    def test_request_with_deferral_allowed_capability_accepted(self):
        req = _req(
            pb.ValidateResourceConfig.Request, config=_STATE, client_capabilities=_capabilities(deferral_allowed=True)
        )
        resp = self.svc.ValidateResourceConfig(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)

    def test_request_with_write_only_capability_accepted(self):
        req = _req(
            pb.ValidateResourceConfig.Request,
            config=_STATE,
            client_capabilities=_capabilities(write_only_attributes_allowed=True),
        )
        resp = self.svc.ValidateResourceConfig(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)

    def test_client_capabilities_field_accessible_on_request(self):
        caps = _capabilities(deferral_allowed=True, write_only_attributes_allowed=True)
        req = _req(pb.ValidateResourceConfig.Request, config=_STATE, client_capabilities=caps)
        self.assertTrue(req.client_capabilities.deferral_allowed)
        self.assertTrue(req.client_capabilities.write_only_attributes_allowed)

    def test_read_resource_with_capabilities_accepted(self):
        svc = _make_servicer()
        req = pb.ReadResource.Request(
            type_name="test_thing",
            current_state=_config_dv("x"),
            client_capabilities=_capabilities(deferral_allowed=True),
        )
        resp = svc.ReadResource(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)

    def test_ctx_client_capabilities_populated(self):
        """Verify ctx.client_capabilities is populated from the request."""
        received = []

        class _CapInspectingResource(_SimpleResource):
            def read(self, ctx: ReadContext, current: State) -> Optional[State]:
                received.append(ctx.client_capabilities)
                return current

        class _Provider(_CapabilitiesProvider):
            def get_resources(self):
                return [_CapInspectingResource]

        svc = ProviderServicer(_Provider())
        req = pb.ReadResource.Request(
            type_name="test_thing",
            current_state=_config_dv("x"),
            client_capabilities=_capabilities(deferral_allowed=True, write_only_attributes_allowed=True),
        )
        svc.ReadResource(req, MagicMock())
        self.assertEqual(len(received), 1)
        self.assertTrue(received[0].deferral_allowed)
        self.assertTrue(received[0].write_only_attributes_allowed)


class TestClientCapabilitiesConfigureProvider(_ServicerTest):
    def test_configure_with_capabilities_accepted(self):
        req = pb.ConfigureProvider.Request(
            terraform_version="1.9.0",
            config=to_dynamic_value({}),
            client_capabilities=_capabilities(deferral_allowed=True),
        )
        resp = self.svc.ConfigureProvider(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)
