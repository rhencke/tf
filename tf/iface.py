from abc import abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Protocol, Sequence, Type, TypeAlias, TypeGuard, runtime_checkable

from tf.schema import Attribute, IdentitySchema, NestedBlock, Schema
from tf.utils import Diagnostics

if TYPE_CHECKING:  # pragma: no cover
    from tf.function import Function


@dataclass
class ClientCapabilities:
    """Capabilities advertised by Terraform on each request (proto 6.6).

    Providers that do not need to inspect these can ignore them — the defaults
    (both False) are safe for all existing behaviour.

    .. seealso::
        Write-only arguments: https://developer.hashicorp.com/terraform/plugin/framework/resources/write-only-arguments
        Deferred actions: https://developer.hashicorp.com/terraform/plugin/framework/actions
    """

    deferral_allowed: bool = False
    """Terraform will retry a deferred plan in a subsequent planning cycle."""

    write_only_attributes_allowed: bool = False
    """Terraform supports the write_only attribute protocol extension."""


class DeferReason(IntEnum):
    """Reason a provider is deferring a resource operation (proto 6.6).

    Pass one of these to ``ctx.defer()`` to signal that the current operation
    cannot be completed yet.  Terraform will retry the plan in a subsequent
    planning cycle when ``client_capabilities.deferral_allowed`` is True.

    .. seealso:: https://developer.hashicorp.com/terraform/plugin/framework/actions/implementation
    """

    UNKNOWN = 0
    RESOURCE_CONFIG_UNKNOWN = 1
    """One or more resource config values are not yet known."""
    PROVIDER_CONFIG_UNKNOWN = 2
    """The provider configuration is not yet fully known."""
    ABSENT_PREREQ = 3
    """A prerequisite resource does not yet exist."""


State: TypeAlias = dict
"""
State is the current state of a resource.
It is a dictionary where field names are mapped to Python values (or None, or Unknown).
Resource operations are mostly just pushing around, mutating, and returning State.
"""


Config: TypeAlias = dict
"""
Config is like State, except its used in configuration validation and the values are null when
they are not bound to a value.
This is because the configuration is not yet bound to a resource.
This is merely for validating that set of input parameters or values are correct.
"""


class AbstractResource(Protocol):
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Get the type name for this resource type"""

    @classmethod
    @abstractmethod
    def get_schema(cls) -> Schema:
        """Get the schema for this resource"""


@dataclass
class _Context:
    diagnostics: Diagnostics
    type_name: str
    client_capabilities: ClientCapabilities = field(default_factory=ClientCapabilities)
    current_identity: Optional[dict] = field(default=None)
    _deferred: Optional[DeferReason] = field(default=None, init=False, repr=False)
    _identity_out: Optional[dict] = field(default=None, init=False, repr=False)

    def set_identity(self, identity: dict) -> None:
        """Set the resource identity to return to Terraform.

        Call this from ``create``, ``update``, or ``read`` to provide the
        stable identity of the resource.

        .. note::
            The API surface is defined but CRUD wiring is not yet active.
            ``set_identity()`` stores the value on the context but the framework
            does not yet attach it to RPC responses.  Full wiring is deferred
            until OpenTofu ships support (panics on GetResourceIdentitySchemas
            as of v1.11).  Terraform (HashiCorp) supports this from v1.12.
        """
        self._identity_out = identity

    def defer(self, reason: DeferReason = DeferReason.RESOURCE_CONFIG_UNKNOWN) -> None:
        """Signal that this operation cannot be completed yet.

        Terraform will retry the plan in a subsequent planning cycle.  Only
        meaningful when ``self.client_capabilities.deferral_allowed`` is True;
        calling ``defer()`` when deferral is not allowed will still set the
        flag and the framework will encode it in the response, but Terraform
        may treat it as an error.
        """
        self._deferred = reason


class ReadDataContext(_Context): ...


def _validate_config(
    diags: Diagnostics, type_name: str, config: Config, attributes: Sequence[Attribute], blocks: Sequence[NestedBlock]
):
    """Validate the configuration"""
    # Verify each supplied field is valid
    attr_names = {a.name for a in attributes}
    block_names = {b.type_name for b in blocks}
    known_keys = attr_names | block_names
    unknown_keys = set(config.keys()) - known_keys

    # Kind of lazy about this right now... might need schema version promotion logic later
    for k in unknown_keys:
        diags.add_error(
            f"Unknown field {type_name}.{k}",
            f"The field '{k}' was supplied, but is not a valid field for {type_name}."
            f" This is likely a bug in your state file."
            f" If you did not manually edit state, please report this to your provider.",
        )

    if len(unknown_keys) > 0:
        return

    amap = {a.name: a for a in attributes}

    for k in attr_names:
        if k not in config:
            # If the field is not set, we don't need to validate it -- makes unit tests easy
            continue

        v = config[k]
        a = amap[k]
        if v is not None and a.computed and not a.optional and not a.required:
            diags.add_error(
                f"Field {type_name}.{k} is read-only and should not be set",
                path=[k],
            )


class DataSource(AbstractResource, Protocol):
    def validate(self, diags: Diagnostics, type_name: str, config: Config):
        """Validate the data source configuration"""
        schema = self.get_schema()
        _validate_config(diags, type_name, config, schema.attributes, schema.block_types)

    @abstractmethod
    def read(self, ctx: ReadDataContext, config: Config) -> Optional[State]:
        """Read the data source"""


# Right now all the same but UpdateContext should have a planning/delta fields method
class CreateContext(_Context): ...


class ReadContext(_Context): ...


class UpdateContext(_Context): ...


class DeleteContext(_Context): ...


class UpgradeContext(_Context): ...


class ImportContext(_Context): ...


@dataclass
class PlanContext(_Context):
    changed_fields: set[str] = field(default_factory=set)


class Resource(AbstractResource, Protocol):
    def validate(self, diags: Diagnostics, type_name: str, config: Config):
        """
        Validate the resource configuration

        This is called before any other operation to validate the configuration of the resource.
        You should run parameter validation here.
        Generate errors and warnings through the `diags` object.

        :param diags: Diagnostics
        :param type_name: The type name of the resource
        :param config: The configuration to validate
        """
        schema = self.get_schema()
        _validate_config(diags, type_name, config, schema.attributes, schema.block_types)

    @abstractmethod
    def create(self, ctx: CreateContext, planned: State) -> Optional[State]:
        """
        Create the resource, returning the actual state after creation.

        This is called when a user runs `opentofu apply` and the resource needs to be initially created.

        :param ctx: CreateContext
        :param planned: The planned state of the resource
        """

    @abstractmethod
    def read(self, ctx: ReadContext, current: State) -> Optional[State]:
        """Read the current state of the resource"""

    @abstractmethod
    def update(self, ctx: UpdateContext, current: State, planned: State) -> Optional[State]:
        """Update the resource to the planned state, returning the actual state after the update"""

    @abstractmethod
    def delete(self, ctx: DeleteContext, current: State):
        """Delete the resource, returning None generally"""

    def plan(self, ctx: PlanContext, current: Optional[State], planned: State) -> Optional[State]:
        """Modify the resource change plan"""
        return planned

    def import_(self, ctx: ImportContext, id: str) -> Optional[State]:
        """
        Import a resource

        This is called when a user runs `opentofu import` and provides a resource ID to import.

        :param ctx: ImportContext
        :param id: The resource ID to import
        """

    # TODO(Hunter): move

    def upgrade(self, ctx: UpgradeContext, version: int, old: State) -> Optional[State]:
        """Upgrade an old resource state to the newest schema version"""
        ctx.diagnostics.add_warning(f"Using default upgrade for {ctx.type_name}.")
        # TODO(Hunter): This should probably iterate all attrs, find ones not in the old state, and set defaults
        return old


class OpenContext(_Context): ...


class EphemeralResource(Protocol):
    """An ephemeral resource exists only during plan/apply — never persisted to state.

    Right semantic for one-shot operations (commands, URI fetches, scripts) where
    drift detection is meaningless.
    """

    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """Short name (without provider prefix). The provider prefix is added by the servicer."""

    @classmethod
    @abstractmethod
    def get_schema(cls) -> Optional[Schema]:
        """Schema for this ephemeral resource, or None."""

    @abstractmethod
    def validate(self, diags: Diagnostics, config: Config):
        """Validate the resource configuration."""

    @abstractmethod
    def open(self, ctx: OpenContext, config: Config) -> State:
        """Execute and return results. Called on OpenEphemeralResource.

        :param ctx: Context carrying diagnostics, client capabilities, and defer support.
        :param config: The decoded configuration dict.
        """

    def close(self, diags: Diagnostics, private: bytes) -> None:
        """Finalize. Called on CloseEphemeralResource. Override to release resources."""


def is_importable(klass: Type[Resource]) -> bool:
    """Has the resource implemented the import_ method"""
    return hasattr(klass, "import_") and klass.import_ is not Resource.import_


@runtime_checkable
class ResourceWithIdentity(Protocol):
    """Mixin protocol for resources that expose a stable identity (proto 6.9).

    Implement this alongside :class:`Resource` to opt into Terraform's identity
    protocol.  The identity schema describes the minimal set of attributes that
    uniquely identify the resource instance — used for import and cross-provider
    move operations.

    .. seealso:: https://developer.hashicorp.com/terraform/plugin/framework/resources/identity

    In your ``create``, ``update``, and ``read`` implementations call
    ``ctx.set_identity({"attr": value, ...})`` to record the identity on the
    context.  As of now, this only stores the identity on the context; wiring
    it into CRUD responses and the Terraform protocol encoding is deferred and
    may be added in a future version.

    Example::

        class MyResource(Resource, ResourceWithIdentity):
            @classmethod
            def get_identity_schema(cls) -> IdentitySchema:
                return IdentitySchema(
                    attributes=[
                        IdentityAttribute("id", types.String(), required_for_import=True),
                    ]
                )

            def create(self, ctx: CreateContext, planned: State) -> Optional[State]:
                result = _create_thing(planned)
                ctx.set_identity({"id": result["id"]})
                return result
    """

    @classmethod
    @abstractmethod
    def get_identity_schema(cls) -> IdentitySchema:
        """Return the identity schema for this resource type."""


@runtime_checkable
class ResourceWithUpgradeIdentity(Protocol):
    """Mixin for resources that can migrate old identity data to a new schema.

    When :attr:`IdentitySchema.version` is incremented, Terraform may send
    identity data encoded with an older schema version.  Implement this mixin
    to convert old identity dicts to the current shape.

    .. seealso:: https://developer.hashicorp.com/terraform/plugin/framework/resources/identity-upgrade
    """

    @abstractmethod
    def upgrade_identity(self, ctx: UpgradeContext, version: int, old_identity: dict) -> Optional[dict]:
        """Upgrade ``old_identity`` (encoded at ``version``) to the current schema.

        :param ctx: Context carrying diagnostics.
        :param version: The schema version the identity data was encoded with.
        :param old_identity: The decoded identity dict from the older schema.
        :returns: The identity dict in the current schema shape.
        """


def has_identity(klass: Type[Resource]) -> TypeGuard[Type[ResourceWithIdentity]]:
    """Return True if *klass* implements :class:`ResourceWithIdentity`."""
    return issubclass(klass, ResourceWithIdentity)


class Provider(Protocol):
    @abstractmethod
    def get_model_prefix(self) -> str:
        """Get the model prefix for all loaded resources"""

    @abstractmethod
    def get_provider_schema(self, diags: Diagnostics) -> Schema:
        """Get the schema for the provider"""

    @abstractmethod
    def full_name(self) -> str:
        """Get the full provider name eg `terraform.example.com/ex/ex`"""

    @abstractmethod
    def validate_config(self, diags: Diagnostics, config: Config):
        """Validate the provider configuration"""

    @abstractmethod
    def configure_provider(self, diags: Diagnostics, config: Config):
        """Called when the provider is configured. This is a good place to set up any global state"""

    @abstractmethod
    def get_data_sources(self) -> list[Type[DataSource]]:
        """Get all the data source types that this provider supports"""

    @abstractmethod
    def get_resources(self) -> list[Type[Resource]]:
        """Get all the resource types that this provider supports"""

    def get_functions(self) -> list[Type["Function"]]:
        """Get all the function types that this provider supports"""
        return []

    def get_ephemeral_resources(self) -> list[Type[EphemeralResource]]:
        """Get all the ephemeral resource types that this provider supports"""
        return []

    def new_resource(self, klass: Type[Resource]) -> Resource:
        return klass(self)  # pyre-ignore[19]: noqa: Don't care about __init__

    def new_data_source(self, klass: Type[DataSource]) -> DataSource:
        return klass(self)  # pyre-ignore[19]: noqa: Don't care about __init__

    def new_function(self, klass: Type["Function"]) -> "Function":
        return klass(self)  # pyre-ignore[19]: noqa: Don't care about __init__

    def new_ephemeral_resource(self, klass: Type[EphemeralResource]) -> EphemeralResource:
        return klass(self)  # pyre-ignore[19]: noqa: Don't care about __init__
