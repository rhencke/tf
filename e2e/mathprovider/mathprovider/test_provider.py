from e2e_framework import ProviderTest

# Minimum OpenTofu version for proto 6.6–6.9 features (write_only, ephemeral, identity).
_PROTO_69_MIN = (1, 11, 0)  # (major, minor, maint)


class DataSourceTest(ProviderTest):
    def test_plan_happy(self):
        result = self.tf_plan(
            """\
            data "math_div" "test" {
                dividend = 10
                divisor = 2
            }

            output "result" {
                value = data.math_div.test.quotient
            }
            """,
            expect_error=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("data.math_div.test: Reading...", result.stdout)
        self.assertIn("result = 5", result.stdout)

    def test_plan_error(self):
        self.tf_plan(
            """\
            data "math_div" "test" {
                dividend = 10
                divisor = 0
            }
            """,
            expect_error=True,
            expect_in_output=[
                "Error: Invalid divisor",
                "The 'divisor' attribute cannot be zero.",
            ],
        )

    def test_apply_happy(self):
        result = self.tf_apply(
            """\
            data "math_div" "test" {
                dividend = 20
                divisor = 4
            }

            output "result" {
                value = data.math_div.test.quotient
            }
            """,
            expect_error=False,
        )
        self.assertIn("data.math_div.test: Reading...", result.stdout)
        self.assertIn("result = 5", result.stdout)

        self.assertEqual(
            {
                "outputs": {"result": {"sensitive": False, "value": 5, "type": "number"}},
                "root_module": {
                    "resources": [
                        {
                            "address": "data.math_div.test",
                            "mode": "data",
                            "type": "math_div",
                            "name": "test",
                            "provider_name": "test.terraform.io/test/math",
                            "schema_version": 0,
                            "values": {"dividend": 20, "divisor": 4, "quotient": 5},
                            "sensitive_values": {},
                        }
                    ]
                },
            },
            self.tf_state()["values"],
        )

    def test_apply_error(self):
        self.tf_apply(
            """\
            data "math_div" "test" {
                dividend = 20
                divisor = 0
            }
            """,
            expect_error=True,
            expect_in_output=[
                "Error: Invalid divisor",
                "The 'divisor' attribute cannot be zero.",
            ],
        )

    def test_apply_for_each(self):
        result = self.tf_apply(
            """\
            data "math_div" "test" {
                for_each = toset(["2", "5", "10"])
                
                dividend = 20
                divisor  = tonumber(each.value)
            }

            output "the_sum" {
                value = sum([for d in data.math_div.test : d.quotient])
            }
            """,
            expect_error=False,
        )
        self.assertIn("the_sum = 16", result.stdout)


class MapTest(ProviderTest):
    def test_map_happy(self):
        self.tf_apply(
            """\
            resource "math_constant" "pi" {
                name         = "pi"
                approx_value = 3.14159
                tags = {
                    category = "irrational"
                    source   = "demo"
                }
            }

            output "constant_name" {
                value = math_constant.pi.name
            }
            """,
            expect_error=False,
        )

        self.assertEqual(
            {
                "outputs": {
                    "constant_name": {
                        "sensitive": False,
                        "value": "pi",
                        "type": "string",
                    }
                },
                "root_module": {
                    "resources": [
                        {
                            "address": "math_constant.pi",
                            "mode": "managed",
                            "type": "math_constant",
                            "name": "pi",
                            "provider_name": "test.terraform.io/test/math",
                            "schema_version": 0,
                            "values": {
                                "name": "pi",
                                "approx_value": 3.14159,
                                "tags": {
                                    "category": "irrational",
                                    "source": "demo",
                                },
                            },
                            "sensitive_values": {"tags": {}},
                        }
                    ]
                },
            },
            self.tf_state()["values"],
        )

    def test_map_key_order_irrelevant(self):
        """Verify map key order does not affect plan diffs."""
        self.tf_apply(
            """\
            resource "math_constant" "e" {
                name         = "e"
                approx_value = 2.71828
                tags = {
                    source   = "demo"
                    category = "irrational"
                }
            }
            """,
            expect_error=False,
        )
        self.tf_plan(
            """\
            resource "math_constant" "e" {
                name         = "e"
                approx_value = 2.71828
                tags = {
                    category = "irrational"
                    source   = "demo"
                }
            }
            """,
            expect_error=False,
            expect_changes=False,
        )

    def test_real_map_change(self):
        self.tf_apply(
            """\
            resource "math_constant" "phi" {
                name         = "phi"
                approx_value = 1.61803
                tags = {
                    category = "irrational"
                    source   = "demo"
                }
            }
            """,
            expect_error=False,
        )
        self.tf_plan(
            """\
            resource "math_constant" "phi" {
                name         = "phi"
                approx_value = 1.61803
                tags = {
                    category = "irrational"
                    source   = "updated_demo"
                }
            }
            """,
            expect_error=False,
            expect_changes=True,
            expect_in_output=[
                '~ "source"',
                '"demo" -> "updated_demo"',
            ],
        )


class ClientCapabilitiesE2ETest(ProviderTest):
    """End-to-end tests for ClientCapabilities and ctx.defer() (proto 6.6).

    ClientCapabilities is sent by Terraform on request types such as
    ValidateResourceConfig, ConfigureProvider, ReadResource, PlanResourceChange,
    and ImportResourceState to advertise which optional protocol features it
    supports.  The SDK extracts them into the context object passed to each
    resource method (ctx.client_capabilities).

    Every plan/apply call implicitly exercises the capabilities code path.  We
    verify that the provider handles them correctly by confirming successful
    plan/apply operations — the provider would error if capabilities were
    mishandled.

    Deferred responses require the Terraform client to retry a deferred plan in
    a subsequent planning cycle.  OpenTofu's CLI test suite does not exercise
    this flow end-to-end, so deferred round-trip tests live exclusively in
    tf/tests/test_capabilities.py.
    """

    def test_provider_accepts_requests_with_capabilities(self):
        # ConfigureProvider, ValidateResourceConfig, and ReadDataSource all carry
        # client_capabilities — exercised by any plan that reads a data source.
        result = self.tf_plan(
            """\
            data "math_div" "test" {
                dividend = 6
                divisor  = 3
            }
            output "result" {
                value = data.math_div.test.quotient
            }
            """,
            expect_error=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_apply_with_capabilities_succeeds(self):
        # A full apply exercises the same capability-gated code paths as plan.
        # ApplyResourceChange itself does not carry client_capabilities in proto
        # 6.9; successful completion confirms the provider handled them without
        # error.
        result = self.tf_apply(
            """\
            resource "math_constant" "e" {
                name         = "e"
                approx_value = 2.71828
                tags         = {}
            }
            """,
            expect_error=False,
        )
        self.assertEqual(result.returncode, 0)
