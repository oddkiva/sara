from unittest import TestCase

from do.sara import ramer_douglas_peucker


class TestRamerDouglasPeucker(TestCase):

    def test_square(self):
        square = [(0, 0), (0.25, 0), (0.5, 0), (0.75, 0), (1, 0), (1, 1),
                  (0, 1), (0, 0)]

        actual_polygon = ramer_douglas_peucker(square, 0.1)

        expected_polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]

        self.assertEqual(expected_polygon, actual_polygon)
