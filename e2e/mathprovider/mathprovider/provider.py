import sys
from typing import Optional, Type

from tf import runner
from tf import types as t
from tf.iface import (
    Config,
    CreateContext,
    DataSource,
    DeleteContext,
    EphemeralResource,
    OpenContext,
    ReadContext,
    ReadDataContext,
    Resource,
    State,
    UpdateContext,
)
from tf.provider import Diagnostics, Provider
from tf.schema import Attribute, Schema


class Divider(DataSource):
    @classmethod
    def get_name(cls) -> str:
        return "div"

    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            attributes=[
                Attribute("dividend", t.Number(), required=True),
                Attribute("divisor", t.Number(), required=True),
                Attribute("quotient", t.Number(), computed=True),
            ]
        )

    def validate(self, diags: Diagnostics, type_name: str, config: Config):
        super().validate(diags, type_name, config)
        if config["divisor"] == 0:
            diags.add_error(
                "Invalid divisor",
                "The 'divisor' attribute cannot be zero.",
            )

    def read(self, ctx: ReadDataContext, config: Config) -> Optional[State]:
        return {
            "dividend": config["dividend"],
            "divisor": config["divisor"],
            "quotient": config["dividend"] / config["divisor"],
        }

    def __init__(self, provider: "MathProvider"):
        pass


class Constant(Resource):
    @classmethod
    def get_name(cls) -> str:
        return "constant"

    @classmethod
    def get_schema(cls) -> Schema:
        return Schema(
            attributes=[
                Attribute("name", t.String(), required=True),
                Attribute("approx_value", t.Number(), required=True),
                Attribute("tags", t.Map(t.String()), required=True),
            ]
        )

    def create(self, ctx: CreateContext, planned: State) -> Optional[State]:
        return planned

    def read(self, ctx: ReadContext, current: State) -> Optional[State]:
        return current

    def update(self, ctx: UpdateContext, current: State, planned: State) -> Optional[State]:
        return planned

    def delete(self, ctx: DeleteContext, current: State):
        return None

    def __init__(self, provider: "MathProvider"):
        pass


class ScaledSecret(EphemeralResource):
    """Demonstrates the ephemeral resource lifecycle.

    Multiplies `seed` by `multiplier` and returns the result as `value`.
    The result exists only during plan/apply and is never written to state.
    """

    @classmethod
    def get_name(cls) -> str:
        return "scaled_secret"

    @classmethod
    def get_schema(cls) -> Optional[Schema]:
        return Schema(
            attributes=[
                Attribute("seed", t.Number(), required=True),
                Attribute("multiplier", t.Number(), optional=True),
                Attribute("value", t.Number(), computed=True),
            ]
        )

    def validate(self, diags: Diagnostics, config: Config):
        if config.get("seed") is None:
            diags.add_error("seed is required", "The 'seed' attribute must be set.")

    def open(self, ctx: OpenContext, config: Config) -> State:
        seed = config["seed"]
        multiplier = config.get("multiplier") if config.get("multiplier") is not None else 1
        return {
            "seed": seed,
            "multiplier": multiplier,
            "value": seed * multiplier,
        }

    def __init__(self, provider: "MathProvider"):
        pass


class MathProvider(Provider):
    def get_model_prefix(self) -> str:
        return "math_"

    def get_provider_schema(self, diags: Diagnostics) -> Schema:
        return Schema(attributes=[])

    def full_name(self) -> str:
        return "test.terraform.io/test/math"

    def validate_config(self, diags: Diagnostics, config: Config):
        pass

    def configure_provider(self, diags: Diagnostics, config: Config):
        pass

    def get_data_sources(self) -> list[Type[DataSource]]:
        return [Divider]

    def get_resources(self) -> list[Type[Resource]]:
        return [Constant]

    def get_ephemeral_resources(self) -> list[Type[EphemeralResource]]:
        return [ScaledSecret]


def main():
    provider = MathProvider()
    runner.run_provider(provider, sys.argv)
