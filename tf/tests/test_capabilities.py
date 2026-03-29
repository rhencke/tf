"""Tests for proto 6.6 ClientCapabilities and Deferred (ctx.defer()) support.

ClientCapabilities is advertised by Terraform on each request to signal which
optional protocol extensions it supports:

    deferral_allowed              — client can handle deferred responses
    write_only_attributes_allowed — client understands write_only attributes

Deferred lets a provider signal that it cannot complete an operation right now.
Providers call ``ctx.defer(reason)`` and the framework encodes the Deferred
message in the response automatically.

Tests here cover:

    1. _extract_capabilities() correctly maps proto → ClientCapabilities.
    2. ctx.defer() / DeferReason round-trip through the provider servicer.
    3. Responses that omit deferred continue to work (no regression).
    4. Existing client-capabilities-on-request smoke tests still pass.
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
    DeferReason,
    DeleteContext,
    ImportContext,
    OpenContext,
    PlanContext,
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

    def plan(self, ctx: PlanContext, current, planned):
        return planned

    def delete(self, ctx: DeleteContext, current: State):
        pass


class _PlanDeferringResource(_SimpleResource):
    """Resource whose plan() defers the create plan."""

    @classmethod
    def get_name(cls) -> str:
        return "plan_deferring"

    def plan(self, ctx: PlanContext, current, planned):
        ctx.defer(DeferReason.RESOURCE_CONFIG_UNKNOWN)
        return planned


class _DeferringResource(_SimpleResource):
    """Resource that always defers on read."""

    @classmethod
    def get_name(cls) -> str:
        return "deferring"

    def create(self, ctx: CreateContext, planned: State) -> Optional[State]:
        ctx.defer(DeferReason.RESOURCE_CONFIG_UNKNOWN)
        return planned

    def read(self, ctx: ReadContext, current: State) -> Optional[State]:
        ctx.defer(DeferReason.ABSENT_PREREQ)
        return current

    def import_(self, ctx: ImportContext, id: str) -> Optional[State]:
        ctx.defer(DeferReason.PROVIDER_CONFIG_UNKNOWN)
        return {"value": id}


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
        ctx.defer(DeferReason.PROVIDER_CONFIG_UNKNOWN)
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
        return [_SimpleResource, _DeferringResource, _PlanDeferringResource]


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
# DeferReason enum
# ---------------------------------------------------------------------------


class TestDeferReasonEnum(TestCase):
    def test_values_match_proto(self):
        cases = [
            (DeferReason.UNKNOWN, 0),
            (DeferReason.RESOURCE_CONFIG_UNKNOWN, 1),
            (DeferReason.PROVIDER_CONFIG_UNKNOWN, 2),
            (DeferReason.ABSENT_PREREQ, 3),
        ]
        for reason, expected in cases:
            with self.subTest(reason=reason):
                self.assertEqual(reason, expected)


# ---------------------------------------------------------------------------
# ctx.defer() / _Context
# ---------------------------------------------------------------------------


class TestContextDefer(TestCase):
    def test_defer_sets_reason(self):
        ctx = ReadContext(Diagnostics(), "test_thing")
        self.assertIsNone(ctx._deferred)
        ctx.defer(DeferReason.ABSENT_PREREQ)
        self.assertEqual(ctx._deferred, DeferReason.ABSENT_PREREQ)

    def test_defer_default_reason_is_resource_config_unknown(self):
        ctx = CreateContext(Diagnostics(), "test_thing")
        ctx.defer()
        self.assertEqual(ctx._deferred, DeferReason.RESOURCE_CONFIG_UNKNOWN)

    def test_open_context_defer(self):
        ctx = OpenContext(Diagnostics(), "test_thing")
        ctx.defer(DeferReason.PROVIDER_CONFIG_UNKNOWN)
        self.assertEqual(ctx._deferred, DeferReason.PROVIDER_CONFIG_UNKNOWN)

    def test_plan_context_defer(self):
        ctx = PlanContext(Diagnostics(), "test_thing")
        ctx.defer(DeferReason.ABSENT_PREREQ)
        self.assertEqual(ctx._deferred, DeferReason.ABSENT_PREREQ)


# ---------------------------------------------------------------------------
# Deferred encoded in responses
# ---------------------------------------------------------------------------


class TestDeferredInReadResource(_ServicerTest):
    def test_deferred_absent_when_not_set(self):
        req = pb.ReadResource.Request(
            type_name="test_thing",
            current_state=_config_dv("x"),
        )
        resp = self.svc.ReadResource(req, MagicMock())
        self.assertFalse(resp.HasField("deferred"))

    def test_deferred_present_when_ctx_defer_called(self):
        req = pb.ReadResource.Request(
            type_name="test_deferring",
            current_state=_config_dv("x"),
        )
        resp = self.svc.ReadResource(req, MagicMock())
        self.assertTrue(resp.HasField("deferred"))
        self.assertEqual(resp.deferred.reason, pb.Deferred.ABSENT_PREREQ)


class TestDeferredInReadDataSource(_ServicerTest):
    def test_deferred_present_from_data_source(self):
        req = pb.ReadDataSource.Request(
            type_name="test_info",
            config=to_dynamic_value({}),
        )
        resp = self.svc.ReadDataSource(req, MagicMock())
        self.assertTrue(resp.HasField("deferred"))
        self.assertEqual(resp.deferred.reason, pb.Deferred.PROVIDER_CONFIG_UNKNOWN)

    def test_deferred_field_absent_from_response_by_default(self):
        # A data source that does NOT defer should have no deferred field.
        class _NonDeferringDS(_SimpleDataSource):
            def read(self, ctx, config):
                return {"value": "ok"}

        class _Provider(_CapabilitiesProvider):
            def get_data_sources(self):
                return [_NonDeferringDS]

        svc = ProviderServicer(_Provider())
        req = pb.ReadDataSource.Request(
            type_name="test_info",
            config=to_dynamic_value({}),
        )
        resp = svc.ReadDataSource(req, MagicMock())
        self.assertFalse(resp.HasField("deferred"))


class TestDeferredInApplyResourceChange(_ServicerTest):
    def test_deferred_in_create(self):
        state = to_dynamic_value({"value": "x"})
        req = pb.ApplyResourceChange.Request(
            type_name="test_deferring",
            prior_state=to_dynamic_value(None),
            planned_state=state,
            config=state,
        )
        resp = self.svc.ApplyResourceChange(req, MagicMock())
        # The deferring resource calls ctx.defer() in create; framework should
        # encode it in the response — but ApplyResourceChange.Response does not
        # carry a deferred field in the proto; assert the response has no error.
        self.assertEqual(len(resp.diagnostics), 0)

    def test_deferred_in_import(self):
        req = pb.ImportResourceState.Request(
            type_name="test_deferring",
            id="abc",
            client_capabilities=_capabilities(deferral_allowed=True),
        )
        resp = self.svc.ImportResourceState(req, MagicMock())
        self.assertTrue(resp.HasField("deferred"))
        self.assertEqual(resp.deferred.reason, pb.Deferred.PROVIDER_CONFIG_UNKNOWN)


class TestDeferredInPlanResourceChange(_ServicerTest):
    def test_deferred_in_create_when_plan_defers(self):
        """plan() is called for CREATE; ctx.defer() propagates into the response."""
        state = to_dynamic_value({"value": "x"})
        req = pb.PlanResourceChange.Request(
            type_name="test_plan_deferring",
            prior_state=to_dynamic_value(None),
            proposed_new_state=state,
            config=state,
            client_capabilities=_capabilities(deferral_allowed=True),
        )
        resp = self.svc.PlanResourceChange(req, MagicMock())
        self.assertTrue(resp.HasField("deferred"))
        self.assertEqual(resp.deferred.reason, pb.Deferred.RESOURCE_CONFIG_UNKNOWN)

    def test_deferred_absent_in_create_when_plan_does_not_defer(self):
        """CREATE plan response has no deferred field when plan() does not defer."""
        state = to_dynamic_value({"value": "x"})
        req = pb.PlanResourceChange.Request(
            type_name="test_thing",
            prior_state=to_dynamic_value(None),
            proposed_new_state=state,
            config=state,
        )
        resp = self.svc.PlanResourceChange(req, MagicMock())
        self.assertFalse(resp.HasField("deferred"))


# ---------------------------------------------------------------------------
# ClientCapabilities on ValidateResourceConfig (smoke tests — unchanged)
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


# ---------------------------------------------------------------------------
# ClientCapabilities on ConfigureProvider
# ---------------------------------------------------------------------------


class TestClientCapabilitiesConfigureProvider(_ServicerTest):
    def test_configure_with_capabilities_accepted(self):
        req = pb.ConfigureProvider.Request(
            terraform_version="1.9.0",
            config=to_dynamic_value({}),
            client_capabilities=_capabilities(deferral_allowed=True),
        )
        resp = self.svc.ConfigureProvider(req, MagicMock())
        self.assertEqual(len(resp.diagnostics), 0)


# ---------------------------------------------------------------------------
# Deferred proto message shape (proto binding smoke tests)
# ---------------------------------------------------------------------------


class TestDeferredMessage(TestCase):
    def test_deferred_reason_enum_values_present(self):
        self.assertEqual(pb.Deferred.Reason.UNKNOWN, 0)
        self.assertEqual(pb.Deferred.Reason.RESOURCE_CONFIG_UNKNOWN, 1)
        self.assertEqual(pb.Deferred.Reason.PROVIDER_CONFIG_UNKNOWN, 2)
        self.assertEqual(pb.Deferred.Reason.ABSENT_PREREQ, 3)

    def test_deferred_message_constructible(self):
        d = pb.Deferred(reason=pb.Deferred.Reason.RESOURCE_CONFIG_UNKNOWN)
        self.assertEqual(d.reason, pb.Deferred.Reason.RESOURCE_CONFIG_UNKNOWN)

    def test_deferred_can_be_set_on_read_resource_response(self):
        d = pb.Deferred(reason=pb.Deferred.Reason.PROVIDER_CONFIG_UNKNOWN)
        resp = pb.ReadResource.Response(deferred=d)
        self.assertEqual(resp.deferred.reason, pb.Deferred.Reason.PROVIDER_CONFIG_UNKNOWN)


# ---------------------------------------------------------------------------
# Resource identity stub RPCs (proto 6.9 — minimal stubs at this commit)
# ---------------------------------------------------------------------------


class TestResourceIdentityStubs(_ServicerTest):
    def test_get_resource_identity_schemas_returns_response(self):
        resp = self.svc.GetResourceIdentitySchemas(pb.GetResourceIdentitySchemas.Request(), MagicMock())
        self.assertIsInstance(resp, pb.GetResourceIdentitySchemas.Response)

    def test_upgrade_resource_identity_returns_response(self):
        resp = self.svc.UpgradeResourceIdentity(pb.UpgradeResourceIdentity.Request(), MagicMock())
        self.assertIsInstance(resp, pb.UpgradeResourceIdentity.Response)
