__copyright__ = "Copyright (c) Microsoft Corporation and Mila - Quebec AI Institute"
__license__ = "MIT"
import itertools
import unittest

import numpy as np

from segar import get_sim
from segar.factors import Shape, Size, Circle, Order, Position
from segar.rules import IsOn
from segar.sim import Simulator
from segar.things import Object, Tile


sim = Simulator()


class TestThings(unittest.TestCase):
    def test_size_shape(self):
        os = [Object({Size: 0.4}), Object({Shape: Circle(0.4)}), Object()]

        for i, o in enumerate(os):
            self.assertIs(o[Size], o[Shape].size, msg=f"failed on {i}")
            self.assertIs(o[Size], o[Shape].value.size)
            with o[Size].in_place():
                o[Size].set(0.3)

            self.assertIs(o[Size], o[Shape].size)
            self.assertIs(o[Size], o[Shape].value.size)
            self.assertEqual(o[Shape].value.size, 0.3)

    def test_is_on(self):
        sim = get_sim()
        sim.reset()

        o = Object(sim=sim)
        t1 = Tile({Position: [5, 5]}, sim=sim)
        t2 = Tile(sim=sim)
        t3 = Tile(sim=sim)
        tiles = [t1, t2, t3]

        def set_orders(orders):
            o.set_factor(Order, orders[0], allow_in_place=True)
            t1.set_factor(Order, orders[1], allow_in_place=True)
            t2.set_factor(Order, orders[2], allow_in_place=True)
            t3.set_factor(Order, orders[3], allow_in_place=True)

        order = [0, 1, 2, 3]
        is_on = IsOn()

        all_orders = list(itertools.permutations(order))
        for orders in all_orders:
            set_orders(orders)
            sim.sort_things()
            #  Tile 1 is out of the way, so always False
            top_tile = tiles[np.argmin(orders[2:4]) + 1]
            for tile in tiles:
                o_is_on_tile = is_on(o, tile)
                if tile is top_tile:
                    self.assertTrue(
                        o_is_on_tile,
                        msg=f"Object {o} with order o[Order] was "
                        f"not on tile {top_tile} with order "
                        f"{top_tile[Order]}.",
                    )
                else:
                    self.assertFalse(
                        o_is_on_tile,
                        msg=f"Object {o} with order o[Order] was "
                        f"on tile {tile} with order "
                        f"{tile[Order]} instead of "
                        f"{top_tile} with order "
                        f"{top_tile[Order]}.",
                    )

    def test_factor_membership(self):
        o1 = Object({Position: [1.0, 1.0]})

        for factor_type, factor in o1.factors.items():
            self.assertIn(factor_type, o1)


def test():
    unittest.main()


if __name__ == "__main__":
    test()
