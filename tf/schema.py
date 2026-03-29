from abc import abstractmethod
from enum import Enum
from typing import Any, Optional, cast

from tf.gen import tfplugin_pb2 as pb
from tf.types import TfType, Unknown


class TextFormat(Enum):
    Plain = "plain"
    Markdown = "markdown"


_desc_format_map = {
    TextFormat.Plain: pb.StringKind.PLAIN,
    TextFormat.Markdown: pb.StringKind.MARKDOWN,
}


class Attribute:
    """
    An attribute is a single field in a schema.

    :param name: Name of the attribute
    :param type: Type of the attribute
    :param description: Description of the attribute
    :param required: Required?
    :param optional: Optional?
    :param computed: Computed?
    :param sensitive: Sensitive?
    :param description_kind: Description kind (defaults to Markdown)
    :param deprecated: Deprecated?
    :param requires_replace: Should a change of this value require a replace of the resource?
    :param default: If this value is computed but not set, this will be the default value in the change plan
    """

    def __init__(
        self,
        name: str,
        type: TfType,
        description: Optional[str] = None,
        required: Optional[bool] = False,
        optional: Optional[bool] = False,
        computed: Optional[bool] = False,
        sensitive: Optional[bool] = False,
        description_kind: Optional[TextFormat] = None,
        deprecated: Optional[bool] = None,
        # -- Simplified provider logic --
        # Will changing this attribute require a replace of the resource?
        # Some fields are immutable and changing them requires a new resource to be created
        requires_replace: Optional[bool] = None,
        read_only: Optional[bool] = False,  # TODO(Hunter): Actually enforce this in CREATE/UPDATE
        # If computed and not set by the caller, what should the default value be?
        default: Any = Unknown,
        # write_only: value provided via config but omitted from state (secrets, keys)
        write_only: Optional[bool] = False,
    ):
        if write_only:
            if computed:
                raise ValueError("Attribute: write_only cannot be combined with computed")
            if not required and not optional:
                raise ValueError("Attribute: write_only requires required=True or optional=True")
        self.name = name
        self.type = type
        self.description = description
        self.required = required
        self.optional = optional
        self.computed = computed
        self.sensitive = sensitive
        self.description_kind = description_kind
        self.deprecated = deprecated
        self.requires_replace = requires_replace
        self.default = default
        self.write_only = write_only

    def to_pb(self) -> pb.Schema.Attribute:
        can_be_null = dict(
            description=self.description,
            required=self.required or None,
            optional=self.optional or None,
            computed=self.computed or None,
            sensitive=self.sensitive,
            deprecated=self.deprecated,
            write_only=self.write_only or None,
        )

        return pb.Schema.Attribute(
            name=self.name,
            type=self.type.tf_type(),
            description_kind=_desc_format_map[self.description_kind or TextFormat.Markdown],
            **can_be_null,  # pyre-ignore[6]: we can actually pass in Nones here for defaults
        )


class Schema:
    """
    A schema is a description of the data model for a resource/data source/provider.

    :param attributes: List of attributes
    :param version: Version of the schema
    :param block_types: List of nested block types

    Example::

        from tf.schema import Schema, Attribute
        from tf.types import Number

        schema = schema.Schema(
            version=2,
            attributes=[
                schema.Attribute("a", types.Number(), required=True),
                schema.Attribute("b", types.Number(), required=True, requires_replace=True),
                schema.Attribute("sum", types.Number(), computed=True),
            ],
        )
    """

    def __init__(
        self,
        attributes: Optional[list[Attribute]] = None,
        version: Optional[int] = None,
        block_types: Optional[list["NestedBlock"]] = None,
        description: Optional[str] = None,
        description_kind: Optional[TextFormat] = None,
        deprecated: Optional[bool] = None,
    ):
        self.attributes = attributes or []
        self.version: Optional[int] = version
        self.block_types = block_types or []
        self.description = description
        self.description_kind = description_kind
        self.deprecated = deprecated

    def to_pb(self) -> pb.Schema:
        more = {"version": self.version} if self.version is not None else {}

        return pb.Schema(
            block=Block(
                attributes=self.attributes,
                block_types=self.block_types,
                description=self.description,
                description_kind=self.description_kind,
                deprecated=self.deprecated,
            ).to_pb(),
            **more,
        )


class Block:
    def __init__(
        self,
        attributes: Optional[list[Attribute]] = None,
        block_types: Optional[list["NestedBlock"]] = None,
        description: Optional[str] = None,
        description_kind: Optional[TextFormat] = None,
        deprecated: Optional[bool] = None,
    ):
        self.attributes = attributes or []
        self.block_types = block_types or []
        self.description = description
        self.description_kind = description_kind
        self.deprecated = deprecated

    def to_pb(self) -> pb.Schema.Block:
        more = {
            "block_types": [nb.to_pb() for nb in self.block_types] or None,
            "description": self.description,
            "description_kind": _desc_format_map[self.description_kind]
            if self.description_kind
            else (_desc_format_map[TextFormat.Markdown] if self.description else None),
            "deprecated": self.deprecated,
        }

        not_none = {k: v for k, v in more.items() if v is not None}

        return pb.Schema.Block(
            attributes=[attr.to_pb() for attr in self.attributes],
            **cast(dict, not_none),
        )


class NestMode(Enum):
    Set = "set"
    Single = "single"


class NestedBlock:
    _mode_map = {
        NestMode.Set: pb.Schema.NestedBlock.NestingMode.SET,
        NestMode.Single: pb.Schema.NestedBlock.NestingMode.SINGLE,
    }

    def __init__(
        self,
        type_name: str,
        nesting_mode: NestMode,
        block: Block,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
    ):
        self.type_name = type_name
        self.block = block
        self.min_items = min_items
        self.max_items = max_items
        self.nesting_mode = nesting_mode

    def to_pb(self) -> pb.Schema.NestedBlock:
        return pb.Schema.NestedBlock(
            type_name=self.type_name,
            block=self.block.to_pb(),
            min_items=self.min_items,
            max_items=self.max_items,
            nesting=self._mode_map[self.nesting_mode],
        )

    @abstractmethod
    def encode(self, value: Any) -> Any:
        """Encode the python representation into the tf-serializable"""

    @abstractmethod
    def decode(self, value: Any) -> Any:
        """Decode the tf-serializable representation into the python representation"""

    @abstractmethod
    def semantically_equal(self, a_decoded, b_decoded) -> bool:
        """
        Check if two Python-types (represented by the implementing type) are semantically equal.
        For Integers, ints will be passed in, and so on.
        """

    def _amap(self) -> dict[str, Attribute]:
        return {a.name: a for a in self.block.attributes}

    def _bmap(self) -> dict[str, "NestedBlock"]:
        return {b.type_name: b for b in self.block.block_types}


class IdentityAttribute:
    """A single attribute in a resource identity schema.

    Identity attributes uniquely identify a resource instance for import and
    cross-provider move operations.  Only primitive types (bool, number, string)
    and lists of primitives are supported — no nested blocks, no maps.

    Typically, exactly one of ``required_for_import`` or ``optional_for_import``
    should be True.  ``required_for_import`` attributes must be supplied by the
    user when running ``terraform import``; ``optional_for_import`` attributes
    may be omitted and will be populated by the provider during import.
    Setting both to False means the attribute does not participate in import;
    setting both to True is not supported by Terraform and should be avoided.

    .. seealso:: https://developer.hashicorp.com/terraform/plugin/framework/resources/identity
    """

    def __init__(
        self,
        name: str,
        type: TfType,
        required_for_import: bool = False,
        optional_for_import: bool = False,
        description: Optional[str] = None,
    ):
        if required_for_import and optional_for_import:
            raise ValueError("IdentityAttribute: required_for_import and optional_for_import " "cannot both be True")
        self.name = name
        self.type = type
        self.required_for_import = required_for_import
        self.optional_for_import = optional_for_import
        self.description = description

    def to_pb(self) -> pb.ResourceIdentitySchema.IdentityAttribute:
        kwargs: dict = dict(
            name=self.name,
            type=self.type.tf_type(),
            required_for_import=self.required_for_import,
            optional_for_import=self.optional_for_import,
        )
        if self.description is not None:
            kwargs["description"] = self.description
        return pb.ResourceIdentitySchema.IdentityAttribute(**kwargs)


class IdentitySchema:
    """Schema for a resource's stable identity (proto 6.9).

    The identity schema describes the minimal set of attributes that uniquely
    identifies a resource instance.  It is separate from the resource's main
    schema and is used by Terraform for import and cross-provider move support.

    :param attributes: List of :class:`IdentityAttribute` descriptors.
    :param version: Schema version — increment when the identity shape changes
        and implement ``ResourceWithUpgradeIdentity`` to migrate old identity
        data to the new shape.

    .. seealso::
        Identity: https://developer.hashicorp.com/terraform/plugin/framework/resources/identity
        Upgrade: https://developer.hashicorp.com/terraform/plugin/framework/resources/identity-upgrade
    """

    def __init__(
        self,
        attributes: Optional[list[IdentityAttribute]] = None,
        version: int = 0,
    ):
        self.attributes = attributes or []
        self.version = version

    def to_pb(self) -> pb.ResourceIdentitySchema:
        return pb.ResourceIdentitySchema(
            version=self.version,
            identity_attributes=[a.to_pb() for a in self.attributes],
        )
