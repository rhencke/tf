"""Tests for e2e_framework helpers.

Covers requires_tf(major, minor, patch) — a decorator that skips a
ProviderTest subclass or individual test case when the active
Terraform/OpenTofu binary is older than the specified minimum version.

These tests use object.__new__ to bypass ProviderTest.setUp's file-system
side-effects (temp-dir creation) so no binary is needed to run them.
"""

from unittest import SkipTest, TestCase

from e2e_framework import ProviderTest, requires_tf


def _make_class(version: tuple[int, int, int], *decorator_args, setUp=None):
    """Return a requires_tf-decorated ProviderTest subclass with a fixed tf binary version."""
    v = ".".join(str(x) for x in version)

    class _T(ProviderTest):
        PROVIDER_NAME = "test.example.com/test/test"

        def _get_tf_command(self_inner):
            return f"tofu-v{v}"

    if setUp is not None:
        _T.setUp = setUp

    return requires_tf(*decorator_args)(_T)


def _make_class_with_setup(version: tuple[int, int, int], *decorator_args, side_effect: list):
    """Like _make_class but the class has its own setUp that appends to side_effect."""

    def setUp(self_inner):
        side_effect.append(True)

    return _make_class(version, *decorator_args, setUp=setUp)


# (label, active_version, min_version, should_skip)
_SKIP_CASES = [
    ("major too old", (0, 99, 99), (1, 0, 0), True),
    ("minor too old", (1, 10, 99), (1, 11, 0), True),
    ("patch too old", (1, 11, 0), (1, 11, 1), True),
    ("exact match", (1, 11, 0), (1, 11, 0), False),
    ("exceeds minimum", (1, 12, 0), (1, 11, 0), False),
]


class TestRequiresTf(TestCase):
    def test_skip_behaviour(self):
        for label, version, min_ver, should_skip in _SKIP_CASES:
            with self.subTest(label):
                klass = _make_class(version, *min_ver)
                if should_skip:
                    with self.assertRaises(SkipTest):
                        klass.setUp(object.__new__(klass))
                else:
                    try:
                        klass.setUp(object.__new__(klass))
                    except SkipTest:
                        self.fail(f"setUp raised SkipTest for '{label}'")
                    except Exception:
                        pass  # super().setUp() may fail on a bare object — expected

    def test_skip_message_includes_version(self):
        klass = _make_class((1, 10, 0), 1, 11, 0)
        with self.assertRaises(SkipTest) as cm:
            klass.setUp(object.__new__(klass))
        self.assertIn("1.11.0", str(cm.exception))

    def test_existing_setup_called_when_version_ok(self):
        called = []
        klass = _make_class_with_setup((1, 11, 0), 1, 11, 0, side_effect=called)
        try:
            klass.setUp(object.__new__(klass))
        except Exception:
            pass
        self.assertTrue(called, "Original setUp was not called")

    def test_existing_setup_not_called_when_skipped(self):
        called = []
        klass = _make_class_with_setup((1, 10, 0), 1, 11, 0, side_effect=called)
        with self.assertRaises(SkipTest):
            klass.setUp(object.__new__(klass))
        self.assertFalse(called, "Original setUp was called before skip")

    def test_unparseable_binary_name_raises(self):
        """When TF_BINARY_NAME contains no version string, a RuntimeError is raised."""

        @requires_tf(1, 0, 0)
        class _T(ProviderTest):
            PROVIDER_NAME = "test.example.com/test/test"

            def _get_tf_command(self_inner):
                return "tofu"  # no version number in the string

        with self.assertRaises(RuntimeError):
            _T.setUp(object.__new__(_T))


class TestRequiresTfMethodDecorator(TestCase):
    def _make_method(self, version_str: str, *decorator_args):
        """Return a requires_tf-decorated test method bound to a fake self with the given version."""

        class _FakeTest:
            def _get_tf_command(self):
                return version_str

            def skipTest(self, msg):
                raise SkipTest(msg)

        @requires_tf(*decorator_args)
        def test_method(self_inner):
            pass

        return test_method, _FakeTest()

    def test_skip_behaviour(self):
        for label, version, min_ver, should_skip in _SKIP_CASES:
            with self.subTest(label):
                v = ".".join(str(x) for x in version)
                method, fake = self._make_method(f"tofu-v{v}", *min_ver)
                if should_skip:
                    with self.assertRaises(SkipTest):
                        method(fake)
                else:
                    method(fake)  # must not raise

    def test_skip_message_includes_version(self):
        method, fake = self._make_method("tofu-v1.10.0", 1, 11, 0)
        with self.assertRaises(SkipTest) as cm:
            method(fake)
        self.assertIn("1.11.0", str(cm.exception))

    def test_method_preserves_name(self):
        @requires_tf(1, 11, 0)
        def test_my_feature(self_inner):
            pass

        self.assertEqual(test_my_feature.__name__, "test_my_feature")

    def test_unparseable_binary_name_raises(self):
        """When command string contains no version, a RuntimeError is raised."""
        method, fake = self._make_method("tofu", 1, 0, 0)
        with self.assertRaises(RuntimeError):
            method(fake)
