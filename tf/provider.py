import functools
import json
import traceback
from copy import deepcopy
from typing import Any, Optional, Tuple, Type, cast

import grpc

from tf.function import CallContext, Function
from tf.gen import tfplugin_pb2 as pb
from tf.gen import tfplugin_pb2_grpc as rpc
from tf.iface import (
    ClientCapabilities,
    CreateContext,
    DataSource,
    DeferReason,
    DeleteContext,
    EphemeralResource,
    ImportContext,
    OpenContext,
    PlanContext,
    Provider,
    ReadContext,
    ReadDataContext,
    Resource,
    ResourceWithUpgradeIdentity,
    UpdateContext,
    UpgradeContext,
    has_identity,
    is_importable,
)

# TODO(identity): Wire identity data through CRUD handlers (Read, Plan, Apply, Import)
# once OpenTofu ships support for the resource identity RPCs.  As of OpenTofu 1.11,
# GetResourceIdentitySchemas and UpgradeResourceIdentity satisfy the gRPC interface
# but panic("unimplemented") — meaning the client never calls them and identity data
# flowing through CRUD would be untestable.  The RPC stubs below are kept so the
# proto surface is complete; the per-operation wiring is deferred until a real
# integration test can validate the round-trip.  Terraform (HashiCorp) introduced
# full support in v1.12.
from tf.schema import Attribute, NestedBlock
from tf.types import Unknown
from tf.utils import (
    Diagnostic,
    Diagnostics,
    _to_attribute_path,
    read_dynamic_value,
    to_dynamic_value,
)

_DEFER_REASON_TO_PB = {
    DeferReason.UNKNOWN: pb.Deferred.UNKNOWN,
    DeferReason.RESOURCE_CONFIG_UNKNOWN: pb.Deferred.RESOURCE_CONFIG_UNKNOWN,
    DeferReason.PROVIDER_CONFIG_UNKNOWN: pb.Deferred.PROVIDER_CONFIG_UNKNOWN,
    DeferReason.ABSENT_PREREQ: pb.Deferred.ABSENT_PREREQ,
}


def _check_no_write_only(
    attributes: list[Attribute], block_types: list[NestedBlock], context: str, diags: "Diagnostics"
) -> None:
    """Emit an error diagnostic for every write_only attribute found in *attributes* or nested blocks.

    write_only is only valid for managed resource schemas; provider, data-source, and ephemeral
    resource schemas must not use it.
    """
    for attr in attributes:
        if attr.write_only:
            diags.add_error(
                f"Invalid write_only attribute in {context} schema",
                f"Attribute '{attr.name}' has write_only=True, but write_only is only valid for managed resource schemas.",
            )
    for nb in block_types:
        _check_no_write_only(nb.block.attributes, nb.block.block_types, f"{context}/{nb.type_name}", diags)


def _null_write_only_attrs(attrs: dict[str, Attribute], blocks: dict[str, NestedBlock], state: Optional[dict]) -> None:
    """Null out write_only attribute values in place, recursing into nested blocks.

    Called after plan and read operations when the client supports write_only
    (``caps.write_only_attributes_allowed`` is True) and after apply to strip
    write_only values from returned state.  A None *state* is a no-op.
    """
    if state is None:
        return
    for k, attr in attrs.items():
        if attr.write_only and k in state:
            state[k] = None
    for k, block in blocks.items():
        block_val = state.get(k)
        if not isinstance(block_val, list):
            continue
        for item in block_val:
            if item is not None:
                _null_write_only_attrs(block._amap(), block._bmap(), item)


def _find_matching_config_item(attrs: dict[str, Attribute], planned_item: dict, config_items: list) -> Optional[dict]:
    """Find the config item whose non-write_only fields all match planned_item.

    Set-mode nested blocks have no guaranteed ordering — zip would pair the wrong
    items when Terraform reorders them between config and plan.  We locate the
    config entry by comparing the stable (non-secret) fields instead.

    When all attributes are write_only there are no stable fields to match on.
    A vacuous all() would match every config item against the first planned item,
    potentially cross-injecting secrets.  In that case we fall back to a
    positional, consume-once strategy: we return the first non-None config item
    and mark it None so it cannot be matched again.
    """
    stable_keys = [k for k, a in attrs.items() if not a.write_only]
    if not stable_keys:
        for idx, config_item in enumerate(config_items):
            if config_item is None:
                continue
            config_items[idx] = None
            return config_item
        return None
    for config_item in config_items:
        if config_item is None:
            continue
        if all(planned_item.get(k) == config_item.get(k) for k in stable_keys):
            return config_item
    return None


def _reinject_write_only_attrs(
    attrs: dict[str, Attribute],
    blocks: dict[str, NestedBlock],
    planned_state: dict,
    config_state: dict,
) -> bool:
    """Copy write_only values from config_state into planned_state, recursing into nested blocks.

    ApplyResourceChange carries no client_capabilities field.  The client signals
    write_only support by nulling write_only values in the planned state while keeping
    them non-null in config.  This function restores those values so the resource
    implementation receives them during apply.

    Returns True if any reinjection occurred (used to detect write_only support).
    """
    reinjected = False
    for k, attr in attrs.items():
        if attr.write_only and k in config_state and planned_state.get(k) is None and config_state.get(k) is not None:
            planned_state[k] = config_state[k]
            reinjected = True
    for k, block in blocks.items():
        planned_items = planned_state.get(k)
        config_items = config_state.get(k)
        if not isinstance(planned_items, list) or not isinstance(config_items, list):
            continue
        for planned_item in planned_items:
            if planned_item is None:
                continue
            config_item = _find_matching_config_item(block._amap(), planned_item, config_items)
            if config_item is not None:
                if _reinject_write_only_attrs(block._amap(), block._bmap(), planned_item, config_item):
                    reinjected = True
    return reinjected


def _deferred_pb(ctx) -> Optional[pb.Deferred]:
    """Return a pb.Deferred if the context signalled deferral, else None."""
    if ctx._deferred is None:
        return None
    return pb.Deferred(reason=_DEFER_REASON_TO_PB[ctx._deferred])


def _extract_capabilities(request) -> ClientCapabilities:
    """Extract ClientCapabilities from any proto request that carries the field."""
    caps = getattr(request, "client_capabilities", None)
    if caps is None:
        return ClientCapabilities()
    return ClientCapabilities(
        deferral_allowed=caps.deferral_allowed,
        write_only_attributes_allowed=caps.write_only_attributes_allowed,
    )


def _decode_state(
    diags: Diagnostics,
    attrs: dict[str, Attribute],
    blocks: dict[str, NestedBlock],
    state: pb.DynamicValue | dict[str, Any],
) -> Tuple[Optional[dict], Optional[dict]]:
    st = read_dynamic_value(state) if isinstance(state, pb.DynamicValue) else state

    if st is None:
        return None, None

    def try_decode(field_name: str, decode_f, v: Any) -> Any:
        try:
            return decode_f(v)
        except Exception as exc:
            diags.add_error(
                f"Failed to decode field '{field_name}'",
                detail=f"Error decoding field '{field_name}': {exc}",
                path=[field_name],
            )

            return Unknown

    attr_state = {
        k: try_decode(k, attrs[k].type.decode, v) if v is not Unknown else v for k, v in st.items() if k in attrs
    }
    block_state = {
        k: try_decode(k, blocks[k].decode, v) if v is not Unknown else v for k, v in st.items() if k in blocks
    }

    return st, {**attr_state, **block_state}


def _encode_state_d(
    attrs: dict[str, Attribute],
    blocks: dict[str, NestedBlock],
    state: Optional[dict],
    old: Optional[dict],
) -> dict[str, Any] | None:
    """If any encoded values of state matches the old state, we will use the old state's encoded value"""
    # This preserves byte-for-byte equality for JSON

    if state is None:
        return None

    encoded = {}

    for k, v in state.items():
        try:
            if v is Unknown:
                encoded[k] = Unknown
            elif k in attrs:
                # Check if we can reuse the old encoded value
                if old and k in old:
                    # For simple types, compare encoded values directly
                    if attrs[k].type.__class__.__name__ in ("Number", "String", "Bool"):
                        new_encoded = attrs[k].type.encode(v)
                        if old[k] == new_encoded:
                            encoded[k] = old[k]
                        else:
                            encoded[k] = new_encoded

                    # If the previous value was Unknown and the new one is not, we just accept the new one
                    elif old[k] is Unknown:
                        encoded[k] = attrs[k].type.encode(v)

                    # For complex types, use semantic equality
                    elif attrs[k].type.semantically_equal(attrs[k].type.decode(old[k]), v):
                        encoded[k] = old[k]
                    else:
                        encoded[k] = attrs[k].type.encode(v)
                else:
                    encoded[k] = attrs[k].type.encode(v)
            else:
                # block
                if old and k in old and blocks[k].semantically_equal(blocks[k].decode(old[k]), v):
                    encoded[k] = old[k]
                else:
                    encoded[k] = blocks[k].encode(v)
        except Exception as exc:
            raise EncodeError(f"Failed to encode field '{k}': {type(exc).__name__}: {exc}") from exc

    return encoded


def _encode_state(
    attrs: dict[str, Attribute],
    blocks: dict[str, NestedBlock],
    state: Optional[dict],
    old: Optional[dict],
) -> pb.DynamicValue:
    """If any encoded values of state matches the old state, we will use the old state's encoded value"""
    # This preserves byte-for-byte equality for JSON
    return to_dynamic_value(_encode_state_d(attrs, blocks, state, old))


def _log_errors(f):
    """Decorator because there is no global try/catch mechanism in grpc??"""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            raise

    return wrapper


class ProviderServicer(rpc.ProviderServicer):
    def __init__(self, app: Provider):
        self.app = app
        self._ds_cls_map: Optional[dict[str, Type[DataSource]]] = None
        self._res_cls_map: Optional[dict[str, Type[Resource]]] = None
        self._func_cls_map: Optional[dict[str, Type[Function]]] = None
        self._eph_cls_map: Optional[dict[str, Type[EphemeralResource]]] = None

        # Getting a resource's attributes in k-v form is very common, want to cache
        self._res_attr_map: dict[str, dict[str, Attribute]] = {}
        # Sam for res blocks
        self._res_block_map: dict[str, dict[str, NestedBlock]] = {}

        # Cache for schemas to avoid repeated computation
        self._ds_schema_cache: dict[str, Any] = {}
        self._res_schema_cache: dict[str, Any] = {}
        self._eph_schema_cache: dict[str, Any] = {}

    def _load_cls_map(self, cache_attr: str, getter, use_prefix: bool = True) -> dict:
        """Generic lazy-initialised name→class map for any provider collection."""
        cache = getattr(self, cache_attr)
        if cache is None:
            prefix = self.app.get_model_prefix() if use_prefix else ""
            cache = {prefix + cls.get_name(): cls for cls in getter()}
            setattr(self, cache_attr, cache)
        return cache

    def _load_ds_cls_map(self) -> dict[str, Type[DataSource]]:
        return self._load_cls_map("_ds_cls_map", self.app.get_data_sources)

    def _get_ds_cls(self, type_name: str) -> Type[DataSource]:
        return self._load_ds_cls_map()[type_name]

    def _load_res_cls_map(self) -> dict[str, Type[Resource]]:
        return self._load_cls_map("_res_cls_map", self.app.get_resources)

    def _get_res_cls(self, type_name: str) -> Type[Resource]:
        return self._load_res_cls_map()[type_name]

    def _get_res_attrs(self, type_name: str) -> dict[str, Attribute]:
        if type_name not in self._res_attr_map:
            klass = self._get_res_cls(type_name)
            self._res_attr_map[type_name] = {a.name: a for a in klass.get_schema().attributes}

        return self._res_attr_map[type_name]

    def _get_res_blocks(self, type_name: str) -> dict[str, NestedBlock]:
        if type_name not in self._res_block_map:
            klass = self._get_res_cls(type_name)
            self._res_block_map[type_name] = {b.type_name: b for b in klass.get_schema().block_types}

        return self._res_block_map[type_name]

    def _load_func_cls_map(self) -> dict[str, Type[Function]]:
        return self._load_cls_map("_func_cls_map", self.app.get_functions, use_prefix=False)

    def _get_func_cls(self, name: str) -> Type[Function]:
        return self._load_func_cls_map()[name]

    def _load_ephemeral_cls_map(self) -> dict[str, Type[EphemeralResource]]:
        getter = getattr(self.app, "get_ephemeral_resources", lambda: [])
        return self._load_cls_map("_eph_cls_map", getter)

    def _get_ephemeral_cls(self, type_name: str, context: grpc.ServicerContext) -> Optional[Type[EphemeralResource]]:
        klass = self._load_ephemeral_cls_map().get(type_name)
        if klass is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Unknown ephemeral resource type: {type_name}")
        return klass

    @_log_errors
    def GetMetadata(self, request: pb.GetMetadata.Request, context: grpc.ServicerContext):
        # Return empty metadata - this is called by Terraform to check capabilities
        return pb.GetMetadata.Response(
            server_capabilities=pb.ServerCapabilities(
                # We support plan_destroy for proper cleanup
                plan_destroy=True,
                # GetProviderSchemaOptional indicates we can handle GetProviderSchema being called
                # conditionally based on whether Terraform has a cached schema
                get_provider_schema_optional=True,
            )
        )

    # ----------------- Provider ----------------- #
    @_log_errors
    def GetProviderSchema(self, request: pb.GetProviderSchema.Request, context: grpc.ServicerContext):
        diags = Diagnostics()
        provider_schema_obj = self.app.get_provider_schema(diags)
        _check_no_write_only(provider_schema_obj.attributes, provider_schema_obj.block_types, "provider", diags)
        schema = provider_schema_obj.to_pb()
        self._load_ds_cls_map()
        self._load_res_cls_map()
        self._load_func_cls_map()

        # Use cached schemas
        ds_schemas = {}
        for type_name, klass in self._load_ds_cls_map().items():
            if type_name not in self._ds_schema_cache:
                ds_schema_obj = klass.get_schema()
                _check_no_write_only(
                    ds_schema_obj.attributes, ds_schema_obj.block_types, f"data source '{type_name}'", diags
                )
                self._ds_schema_cache[type_name] = ds_schema_obj.to_pb()
            ds_schemas[type_name] = self._ds_schema_cache[type_name]

        res_schema = {}
        for type_name, klass in self._load_res_cls_map().items():
            if type_name not in self._res_schema_cache:
                self._res_schema_cache[type_name] = klass.get_schema().to_pb()
            res_schema[type_name] = self._res_schema_cache[type_name]

        func_schemas = {name: klass.get_signature().to_pb() for name, klass in self._load_func_cls_map().items()}

        eph_schemas = {}
        for type_name, klass in self._load_ephemeral_cls_map().items():
            s = klass.get_schema()
            if s is not None:
                if type_name not in self._eph_schema_cache:
                    _check_no_write_only(s.attributes, s.block_types, f"ephemeral resource '{type_name}'", diags)
                    self._eph_schema_cache[type_name] = s.to_pb()
                eph_schemas[type_name] = self._eph_schema_cache[type_name]

        # Create a proper provider_meta schema
        # This is an empty schema with an empty block - valid but with no attributes
        provider_meta = pb.Schema(
            version=0,
            block=pb.Schema.Block(
                version=0,
                attributes=[],
                block_types=[],
                description="",
                description_kind=pb.PLAIN,
                deprecated=False,
            ),
        )

        resp = pb.GetProviderSchema.Response(
            provider=schema,
            provider_meta=provider_meta,
            diagnostics=diags.to_pb(),
            data_source_schemas=ds_schemas,
            resource_schemas=res_schema,
            functions=func_schemas,
            ephemeral_resource_schemas=eph_schemas,
        )
        return resp

    @_log_errors
    def ValidateProviderConfig(self, request: pb.ValidateProviderConfig.Request, context: grpc.ServicerContext):
        R = pb.ValidateProviderConfig.Response
        diags = Diagnostics()
        config = read_dynamic_value(request.config)
        self.app.validate_config(diags, config)
        return R(diagnostics=diags.to_pb())

    @_log_errors
    def ValidateResourceConfig(self, request: pb.ValidateResourceConfig.Request, context: grpc.ServicerContext):
        # request.client_capabilities carries flags that Terraform sets to signal which
        # proto 6.8 features it supports (deferral_allowed since 6.6; write_only_attributes_allowed
        # since 6.8).  Basic providers can ignore these; advanced providers may inspect them to opt
        # in to write_only attribute enforcement or deferred planning.
        conf = read_dynamic_value(request.config)
        type_name = request.type_name
        klass = self._get_res_cls(type_name)
        inst = self.app.new_resource(klass)
        diags = Diagnostics()

        inst.validate(diags, type_name, conf)
        return pb.ValidateResourceConfig.Response(diagnostics=diags.to_pb())

    @_log_errors
    def ValidateDataResourceConfig(
        self,
        request: pb.ValidateDataResourceConfig.Request,
        context: grpc.ServicerContext,
    ):
        conf = read_dynamic_value(request.config)
        klass = self._get_ds_cls(request.type_name)
        inst = self.app.new_data_source(klass)
        diags = Diagnostics()

        inst.validate(diags, request.type_name, conf)
        return pb.ValidateDataResourceConfig.Response(diagnostics=diags.to_pb())

    @_log_errors
    def UpgradeResourceState(self, request: pb.UpgradeResourceState.Request, context: grpc.ServicerContext):
        diags = Diagnostics()

        if len(request.raw_state.flatmap) != 0:
            # Not sure what this field is for. I need an example to implement this
            diags.add_error(
                "UpgradeResourceState is not supported",
                detail="UpgradeResourceState using flatmap is not supported. This is a bug in the Plugin SDK.",
            )
            return pb.UpgradeResourceState.Response(diagnostics=diags.to_pb())

        state = json.loads(request.raw_state.json)
        klass = self._get_res_cls(request.type_name)
        inst = self.app.new_resource(klass)
        schema = klass.get_schema()

        old_version = request.version
        new_version = schema.version

        if new_version is None or old_version is None or old_version < new_version:
            state = inst.upgrade(UpgradeContext(diags, request.type_name), old_version, deepcopy(state))

        return pb.UpgradeResourceState.Response(
            upgraded_state=to_dynamic_value(state),
            diagnostics=diags.to_pb(),
        )

    # ----------------- One-time init ----------------- #
    @_log_errors
    def ConfigureProvider(self, request: pb.ConfigureProvider.Request, context: grpc.ServicerContext):
        # request.client_capabilities is available here as well (see ValidateResourceConfig).
        conf = read_dynamic_value(request.config)
        diags = Diagnostics()
        self.app.configure_provider(diags, conf)
        return pb.ConfigureProvider.Response(diagnostics=diags.to_pb())

    # ----------------- Resource Lifecycle ----------------- #
    @_log_errors
    def ReadResource(self, request: pb.ReadResource.Request, context: grpc.ServicerContext):
        diags = Diagnostics()

        type_name = request.type_name
        attrs = self._get_res_attrs(type_name)
        blocks = self._get_res_blocks(type_name)
        current_enc, current_state = _decode_state(diags, attrs, blocks, request.current_state)

        if diags.has_errors():
            return pb.ReadResource.Response(diagnostics=diags.to_pb())

        if current_state is None:
            # I think this is not possible
            diags.add_error(
                f"ReadResource {type_name} called with no state",
                detail="This is a bug in the Plugin SDK",
            )
            return pb.ReadResource.Response(diagnostics=diags.to_pb())

        klass = self._get_res_cls(type_name)
        inst = self.app.new_resource(klass)
        caps = _extract_capabilities(request)
        ctx = ReadContext(diags, type_name, caps)
        new_state = inst.read(ctx, current_state)

        if caps.write_only_attributes_allowed:
            _null_write_only_attrs(attrs, blocks, new_state)

        return pb.ReadResource.Response(
            new_state=_encode_state(attrs, blocks, new_state, current_enc),
            diagnostics=diags.to_pb(),
            deferred=_deferred_pb(ctx),
        )

    @_log_errors
    def PlanResourceChange(self, request: pb.PlanResourceChange.Request, context: grpc.ServicerContext):
        type_name = request.type_name
        diags = Diagnostics()

        attrs = self._get_res_attrs(type_name)
        blocks = self._get_res_blocks(type_name)
        _, prior_state = _decode_state(diags, attrs, blocks, request.prior_state)
        if diags.has_errors():
            return pb.PlanResourceChange.Response(diagnostics=diags.to_pb())

        proposed_enc, proposed_new_state = _decode_state(diags, attrs, blocks, request.proposed_new_state)
        if diags.has_errors():
            return pb.PlanResourceChange.Response(diagnostics=diags.to_pb())

        # config = read_dynamic_value(request.config)
        # prior_private = request.prior_private

        klass = self._get_res_cls(type_name)
        inst = self.app.new_resource(klass)

        caps = _extract_capabilities(request)

        # We simplify the logic here. Instead of requiring each implementing resource to implement
        # plan_resource_change and apply_resource_change, we can figure
        # our what is changing and then map that to resource.(create|update|delete) which is easier to implement.
        # prior_state = None => CREATE
        # proposed_new_state = None => DELETE
        # Otherwise UPDATE

        if prior_state is None and proposed_new_state is not None:
            # We don't need to visit the resource to make a create plan
            # delete nulls = UNKNOWN
            new_state = {}

            for k, v in proposed_new_state.items():
                # IDK I gotta figure out what to do with this
                if k in blocks:
                    new_state[k] = v
                else:
                    # Attribute
                    if v is not None:
                        new_state[k] = v
                    elif not attrs[k].computed:
                        # TF requires non-computed unspecified fields to be set to None as their planned value
                        new_state[k] = None
                    else:
                        new_state[k] = attrs[k].default

            plan_ctx = PlanContext(diags, type_name, caps)
            planned_state = inst.plan(plan_ctx, None, deepcopy(new_state))
            if planned_state is not None:
                new_state = planned_state

            # write_only attributes must be null in planned state — but only when the client
            # signals write_only support.  Older clients that don't set
            # write_only_attributes_allowed treat write_only like a regular attribute and
            # will reject a null planned value that doesn't match the config.
            if caps.write_only_attributes_allowed:
                _null_write_only_attrs(attrs, blocks, new_state)

            new_state_encoded = _encode_state(attrs, blocks, new_state, proposed_enc)
            return pb.PlanResourceChange.Response(
                planned_state=new_state_encoded,
                diagnostics=diags.to_pb(),
                deferred=_deferred_pb(plan_ctx),
            )

        # Kind of interesting, TF does not send us DELETE (old_state = SOME and new_state = None)
        if proposed_new_state is None and prior_state is not None:
            diags.add_error(
                "DELETE should never be sent to PlanResourceChange",
                "This is a bug in the Plugin SDK",
            )
            return pb.PlanResourceChange.Response(diagnostics=diags.to_pb())

        if proposed_new_state is None and prior_state is None:
            diags.add_error(
                "Both prior_state and proposed_new_state are None",
                "We should never be send this by TF.",
            )
            return pb.PlanResourceChange.Response(diagnostics=diags.to_pb())

        # Help the type checker out that this is not None
        proposed_new_state = cast(dict, proposed_new_state)
        prior_state = cast(dict, prior_state)

        # Otherwise we have an update
        # We are just going to naively assume that we are going to reach the desired state for planning
        # purposes but check each attribute to see if it requires a replace
        # TODO(Hunter): Wrap this into some common code so we can supply the computed different fields to UpdateContext
        requires_replace = []
        changed_keys = {
            k
            for k in set(proposed_new_state.keys()) & set(prior_state.keys()) & set(attrs.keys())
            if not attrs[k].type.semantically_equal(prior_state[k], proposed_new_state[k])
        }

        for k in changed_keys:
            if attrs[k].requires_replace:
                requires_replace.append(_to_attribute_path([k]))

        # Only deepcopy if states are not None to avoid unnecessary copies
        prior_copy = deepcopy(prior_state) if prior_state is not None else None
        proposed_copy = deepcopy(proposed_new_state) if proposed_new_state is not None else None

        plan_ctx = PlanContext(diags, type_name, caps, changed_fields=changed_keys)
        planned = inst.plan(
            plan_ctx,
            prior_copy,
            proposed_copy or {},
        )
        if planned is not None:
            proposed_new_state = planned

        # write_only attributes must be null in planned state — but only when the client supports it.
        if caps.write_only_attributes_allowed:
            _null_write_only_attrs(attrs, blocks, proposed_new_state)
        return pb.PlanResourceChange.Response(
            planned_state=_encode_state(attrs, blocks, proposed_new_state, proposed_enc),
            requires_replace=requires_replace,
            diagnostics=diags.to_pb(),
            deferred=_deferred_pb(plan_ctx),
        )

    @_log_errors
    def ApplyResourceChange(self, request: pb.ApplyResourceChange.Request, context: grpc.ServicerContext):
        diags = Diagnostics()

        type_name = request.type_name
        attrs = self._get_res_attrs(type_name)
        blocks = self._get_res_blocks(type_name)

        _, prior_state = _decode_state(diags, attrs, blocks, request.prior_state)
        if diags.has_errors():
            return pb.ApplyResourceChange.Response(diagnostics=diags.to_pb())

        planned_enc, planned_state = _decode_state(diags, attrs, blocks, request.planned_state)
        if diags.has_errors():
            return pb.ApplyResourceChange.Response(diagnostics=diags.to_pb())

        # ApplyResourceChange.Request carries no client_capabilities field; detect write_only
        # support from the planned_state instead.  When the client understands write_only it
        # nulls write_only values in the plan, so if a write_only attr is null in planned_state
        # but non-null in config, the client has already performed write_only processing and we
        # need to re-inject the real value so the resource implementation can use it.
        _, config_state = _decode_state(diags, attrs, blocks, request.config)
        if diags.has_errors():
            return pb.ApplyResourceChange.Response(diagnostics=diags.to_pb())
        write_only_supported = False
        if config_state is not None and planned_state is not None:
            write_only_supported = _reinject_write_only_attrs(attrs, blocks, planned_state, config_state)
        klass = self._get_res_cls(type_name)
        inst = self.app.new_resource(klass)

        # ApplyResourceChange.Request has no client_capabilities field; derive write_only support
        # from the detection above.  deferral_allowed is always False — apply cannot be deferred.
        caps = ClientCapabilities(write_only_attributes_allowed=write_only_supported)
        if prior_state is None and planned_state is not None:
            # Create
            apply_ctx = CreateContext(diags, type_name, caps)
            new_state = inst.create(apply_ctx, planned_state)
        elif prior_state is not None and planned_state is None:
            # Delete
            apply_ctx = DeleteContext(diags, type_name, caps)
            new_state = inst.delete(apply_ctx, prior_state)
        else:
            prior_state = cast(dict, prior_state)
            planned_state = cast(dict, planned_state)
            apply_ctx = UpdateContext(diags, type_name, caps)
            new_state = inst.update(apply_ctx, prior_state, planned_state)

        # The SDK enforces write_only semantics when the client supports it: attributes declared
        # write_only must never be stored in state.  Terraform rejects non-null write_only values,
        # so we strip here at the framework layer.  Only strip when write_only_supported — older
        # clients that don't understand write_only pass the real value through the plan unchanged,
        # and those values must be preserved in state so they don't cause perpetual plan diffs.
        if write_only_supported:
            _null_write_only_attrs(attrs, blocks, new_state)
        # We use the planned field values if they are semantically equivalent to the new state.
        # For most fields on update and create, the TF client will have already done the hard work
        # of encoding the field values to provide the planned state.
        # We can skip re-encoding them if they semantically match what we got back from the resource.
        encoded_state = _encode_state(attrs, blocks, new_state, old=planned_enc)

        return pb.ApplyResourceChange.Response(
            new_state=encoded_state,
            diagnostics=diags.to_pb(),
        )

    @_log_errors
    def ImportResourceState(self, request: pb.ImportResourceState.Request, context: grpc.ServicerContext):
        type_name = request.type_name
        klass = self._get_res_cls(type_name)

        if not is_importable(klass):
            diags = Diagnostics().add_error(
                f"{type_name} does not support resource import",
                f"This provider has not implemented import_ for {type_name}",
            )
            return pb.ImportResourceState.Response(diagnostics=diags.to_pb())

        ctx = ImportContext(Diagnostics(), type_name, _extract_capabilities(request))
        inst = self.app.new_resource(klass)
        state = inst.import_(ctx, request.id)
        attrs = self._get_res_attrs(type_name)
        blocks = self._get_res_blocks(type_name)

        return pb.ImportResourceState.Response(
            imported_resources=(
                [
                    pb.ImportResourceState.ImportedResource(
                        type_name=type_name,
                        state=_encode_state(attrs, blocks, state, old=None),
                    ),
                ]
                if state is not None
                else []
            ),
            diagnostics=ctx.diagnostics.to_pb(),
            deferred=_deferred_pb(ctx),
        )

    @_log_errors
    def MoveResourceState(self, request, context: grpc.ServicerContext):
        diags = Diagnostics().add_error(
            "MoveResourceState is not implemented",
            "MoveResourceState is not implemented",
        )
        return pb.MoveResourceState.Response(diagnostics=diags.to_pb())

    @_log_errors
    def ReadDataSource(self, request: pb.ReadDataSource.Request, context: grpc.ServicerContext):
        config = read_dynamic_value(request.config)

        klass = self._get_ds_cls(request.type_name)
        inst = self.app.new_data_source(klass)
        diags = Diagnostics()

        ctx = ReadDataContext(diags, request.type_name, _extract_capabilities(request))
        state = inst.read(ctx, config)

        return pb.ReadDataSource.Response(
            diagnostics=diags.to_pb(),
            state=to_dynamic_value(state),
            deferred=_deferred_pb(ctx),
        )

    # ----------------- Functions ----------------- #
    @_log_errors
    def GetFunctions(self, request: pb.GetFunctions.Request, context: grpc.ServicerContext):
        diags = Diagnostics()
        self._load_func_cls_map()
        func_schemas = {name: klass.get_signature().to_pb() for name, klass in self._load_func_cls_map().items()}
        return pb.GetFunctions.Response(functions=func_schemas, diagnostics=diags.to_pb())

    @_log_errors
    def CallFunction(self, request: pb.CallFunction.Request, context: grpc.ServicerContext):
        diags = Diagnostics()

        try:
            func_cls = self._get_func_cls(request.name)
        except KeyError:
            return pb.CallFunction.Response(error=pb.FunctionError(text=f"Function '{request.name}' not found"))

        func_inst = self.app.new_function(func_cls)
        signature = func_cls.get_signature()

        # Decode arguments
        decoded_args = []
        for i, arg in enumerate(request.arguments):
            arg_value = read_dynamic_value(arg)
            if i < len(signature.parameters):
                param = signature.parameters[i]
                decoded_args.append(param.type.decode(arg_value))
            elif signature.variadic_parameter:
                decoded_args.append(signature.variadic_parameter.type.decode(arg_value))
            else:
                return pb.CallFunction.Response(
                    error=pb.FunctionError(
                        text=f"Too many arguments for function '{request.name}'",
                        function_argument=i,
                    )
                )

        # Check for missing required arguments
        if len(decoded_args) < len(signature.parameters):
            return pb.CallFunction.Response(
                error=pb.FunctionError(text=f"Missing required arguments for function '{request.name}'")
            )

        # Call the function
        ctx = CallContext(diags, request.name)
        try:
            result = func_inst.call(ctx, decoded_args)

            # Check for diagnostics that would be errors
            if diags.has_errors():
                errors = [d for d in diags.diagnostics if d.severity == Diagnostic.ERROR]
                return pb.CallFunction.Response(
                    error=pb.FunctionError(text=errors[0].summary if errors else "Function call failed")
                )

            # Encode the result
            encoded_result = signature.return_type.type.encode(result)
            return pb.CallFunction.Response(result=to_dynamic_value(encoded_result))
        except Exception as e:
            return pb.CallFunction.Response(error=pb.FunctionError(text=f"Function execution error: {str(e)}"))

    # ----------------- Ephemeral resource lifecycle (proto 6.7) ----------------- #

    @_log_errors
    def ValidateEphemeralResourceConfig(
        self,
        request: pb.ValidateEphemeralResourceConfig.Request,
        context: grpc.ServicerContext,
    ):
        klass = self._get_ephemeral_cls(request.type_name, context)
        if klass is None:
            return pb.ValidateEphemeralResourceConfig.Response()
        config = read_dynamic_value(request.config)
        diags = Diagnostics()
        inst = self.app.new_ephemeral_resource(klass)
        inst.validate(diags, config or {})
        return pb.ValidateEphemeralResourceConfig.Response(diagnostics=diags.to_pb())

    @_log_errors
    def OpenEphemeralResource(self, request: pb.OpenEphemeralResource.Request, context: grpc.ServicerContext):
        klass = self._get_ephemeral_cls(request.type_name, context)
        if klass is None:
            return pb.OpenEphemeralResource.Response()
        config = read_dynamic_value(request.config)
        diags = Diagnostics()
        ctx = OpenContext(diags, request.type_name, _extract_capabilities(request))
        inst = self.app.new_ephemeral_resource(klass)
        result = inst.open(ctx, config or {})
        dv = to_dynamic_value(result)
        # TODO: extend EphemeralResource.open() to support returning renew_at and
        # private bytes so providers can implement renewal and stateful cleanup.
        return pb.OpenEphemeralResource.Response(
            diagnostics=diags.to_pb(),
            result=pb.DynamicValue(msgpack=dv.msgpack),
            deferred=_deferred_pb(ctx),
        )

    @_log_errors
    def RenewEphemeralResource(self, request: pb.RenewEphemeralResource.Request, context: grpc.ServicerContext):
        # Renew is optional; providers that don't implement a renewal window return an empty response.
        # TODO: extend EphemeralResource to support an optional renew() callback that can
        # update renew_at and private, enabling providers to implement lease renewal.
        return pb.RenewEphemeralResource.Response()

    @_log_errors
    def CloseEphemeralResource(self, request: pb.CloseEphemeralResource.Request, context: grpc.ServicerContext):
        klass = self._get_ephemeral_cls(request.type_name, context)
        if klass is None:
            return pb.CloseEphemeralResource.Response()
        diags = Diagnostics()
        inst = self.app.new_ephemeral_resource(klass)
        inst.close(diags, request.private)
        return pb.CloseEphemeralResource.Response(diagnostics=diags.to_pb())

    # ----------------- Resource identity (proto 6.9) ----------------- #

    @_log_errors
    def GetResourceIdentitySchemas(
        self,
        request: pb.GetResourceIdentitySchemas.Request,
        context: grpc.ServicerContext,
    ):
        """Return identity schemas for resource types that implement ResourceWithIdentity.

        Resource types that do not implement the mixin are silently omitted —
        Terraform treats their absence as "no identity support" for that type.
        """
        schemas = {}
        for type_name, klass in self._load_res_cls_map().items():
            if has_identity(klass):
                schemas[type_name] = klass.get_identity_schema().to_pb()
        diags = Diagnostics()
        return pb.GetResourceIdentitySchemas.Response(
            identity_schemas=schemas,
            diagnostics=diags.to_pb(),
        )

    @_log_errors
    def UpgradeResourceIdentity(self, request: pb.UpgradeResourceIdentity.Request, context: grpc.ServicerContext):
        """Upgrade identity data from an older schema version to the current one.

        Called when Terraform has identity data encoded at an older
        ``IdentitySchema.version`` than the provider currently advertises.
        Resource types that implement :class:`ResourceWithUpgradeIdentity` handle
        the migration; others return the raw identity unchanged (safe when the
        shape has not changed, only the version number).
        """
        diags = Diagnostics()
        try:
            klass = self._get_res_cls(request.type_name)
        except KeyError:
            diags.add_error(
                f"Unknown resource type: {request.type_name}",
                f"UpgradeResourceIdentity called for unknown type '{request.type_name}'",
            )
            return pb.UpgradeResourceIdentity.Response(diagnostics=diags.to_pb())

        raw = request.raw_identity
        if raw.json:
            import json as _json

            try:
                old_identity = _json.loads(raw.json)
            except (_json.JSONDecodeError, UnicodeDecodeError) as exc:
                diags.add_error(
                    "Invalid identity JSON",
                    f"Failed to decode raw identity for '{request.type_name}' " f"(version {request.version}): {exc}",
                )
                return pb.UpgradeResourceIdentity.Response(diagnostics=diags.to_pb())
        else:
            old_identity = {}

        inst = self.app.new_resource(klass)
        if isinstance(inst, ResourceWithUpgradeIdentity):
            ctx = UpgradeContext(diags, request.type_name)
            upgraded = inst.upgrade_identity(ctx, request.version, old_identity)
        else:
            upgraded = old_identity

        if upgraded is not None:
            return pb.UpgradeResourceIdentity.Response(
                upgraded_identity=pb.ResourceIdentityData(identity_data=to_dynamic_value(upgraded)),
                diagnostics=diags.to_pb(),
            )
        return pb.UpgradeResourceIdentity.Response(diagnostics=diags.to_pb())

    # ----------------- Graceful shutdown ----------------- #
    @_log_errors
    def StopProvider(self, request: pb.StopProvider.Request, context: grpc.ServicerContext):
        # Return empty response to acknowledge shutdown request
        # The actual shutdown is handled by the interceptor and server loop
        return pb.StopProvider.Response()


class EncodeError(Exception):
    pass
