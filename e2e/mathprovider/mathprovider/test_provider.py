from e2e_framework import ProviderTest


class DataSourceTest(ProviderTest):
    PROVIDER_NAME = "test.terraform.io/test/math"

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
    PROVIDER_NAME = "test.terraform.io/test/math"

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
