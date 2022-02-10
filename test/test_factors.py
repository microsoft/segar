__copyright__ = (
    "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
)
__license__ = "MIT"
import unittest

from segar.factors import (
    factors,
    number_factors,
    arrays,
    bools,
    noise,
    properties,
)
from segar.factors import Factor


_ALL_MODS = (factors, number_factors, arrays, bools, noise, properties)


class TestFactor(unittest.TestCase):
    def _test_creation_from_mod(self, mod):
        for name in mod.__all__:
            if name in ["DEFAULTS", "FACTORS", "Deterministic"]:
                continue
            cls = getattr(mod, name)
            if issubclass(cls, Factor):
                try:
                    print(f"Testing creations Factor {cls}.")
                    c = cls()
                except Exception:
                    raise AssertionError(
                        f"Creation failed on factor `" f"{name}`."
                    )

                if cls not in (Factor, number_factors.NumericFactor):
                    if c.value is not None:
                        assert isinstance(c.value, c.t), (cls, c.value, c.t)

    def _test_abstract_factor_creation(self, t):
        f = Factor[t]()
        self.assertTrue(f.t == t)

    def test_creations(self):
        for mod in _ALL_MODS:
            self._test_creation_from_mod(mod)

    def test_creation_abstracts(self):
        for t in (float, str, bool, int, type(None), list, dict, tuple):
            print(f"Testing abstract creations of {Factor[t]}.")
            self._test_abstract_factor_creation(t)

    def test_protected(self):
        c = number_factors.Charge(0.1)
        with self.assertRaises(ValueError):
            c += 1

        with self.assertRaises(ValueError):
            c -= 1

        with self.assertRaises(ValueError):
            c /= 2
            print(c)

        with self.assertRaises(ValueError):
            c *= 2

        with self.assertRaises(ValueError):
            c.set(1)

        with c.in_place():
            c += 1
            c -= 1
            c /= 1
            c *= 1
            c.set(1)

    def test_create_from_reference(self, val=0.1):
        c = number_factors.Charge(val)
        q = number_factors.Magnetism(c)
        self.assertEqual(c, q)
        with c.in_place():
            c += 1
        self.assertEqual(c, val + 1)
        self.assertEqual(q, val)
        with q.in_place():
            q -= 1
        self.assertEqual(q, val - 1)
        self.assertEqual(c, val + 1)

    def test_noise(self):
        for n in noise.__all__:
            if n in ("Noise", "Deterministic"):
                continue
            print(f"Testing noise {n}.")
            n_cls = getattr(noise, n)

            if hasattr(n_cls, "_test_init"):
                noise_dist = n_cls._test_init()
            else:
                noise_dist = n_cls()

            f = noise_dist.sample()
            if n != "Choice":
                self.assertEqual(f.t, type(f.value))
            else:
                self.assertIn(f, ("foo", "bar"))

        u = noise.GaussianNoise()
        c = number_factors.Charge(u.sample())
        v = c.value
        self.assertEqual(c.t, type(c.value))
        u.sample()
        self.assertEqual(c.value, v)


def test():
    unittest.main()


if __name__ == "__main__":
    test()
