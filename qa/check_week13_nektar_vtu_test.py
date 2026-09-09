"""Self-contained fixture tests: python3 -m unittest discover -s qa -p '*nektar_vtu_test.py'."""
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

from check_week13_nektar_vtu import compare


def fixture(path, u="0 1 0", p="0 1 2", coords="0 0 0 0.5 5 0 1 0 0", encoding="ascii"):
    root = ET.Element("VTKFile")
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(grid, "Piece", NumberOfPoints="3")
    pointdata = ET.SubElement(piece, "PointData")
    for name, value in (("u", u), ("v", "0 0 0"), ("p", p)):
        ET.SubElement(pointdata, "DataArray", Name=name, format=encoding).text = value
    points = ET.SubElement(piece, "Points")
    ET.SubElement(points, "DataArray", format="ascii").text = coords
    cells = ET.SubElement(piece, "Cells")
    for name, value in (("connectivity", "0 1 2"), ("offsets", "3"), ("types", "5")):
        ET.SubElement(cells, "DataArray", Name=name, format="ascii").text = value
    ET.ElementTree(root).write(path)


class CheckVtu(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.a = Path(self.temp.name)/"a.vtu"
        self.b = Path(self.temp.name)/"b.vtu"
        fixture(self.a)

    def test_identical(self):
        fixture(self.b)
        result = compare(self.a, self.b, 1e-12, 1e-8)
        self.assertTrue(result["numeric_comparison_passed"])
        self.assertFalse(result["time_match_verified"])

    def test_pressure_gauge(self):
        fixture(self.b, p="10 11 12")
        self.assertTrue(compare(self.a, self.b, 1e-12, 1e-8)["numeric_comparison_passed"])

    def test_mismatch(self):
        fixture(self.b, u="0 1.1 0")
        self.assertFalse(compare(self.a, self.b, 1e-12, 1e-8)["numeric_comparison_passed"])

    def test_nonfinite(self):
        fixture(self.b, u="0 nan 0")
        with self.assertRaises(ValueError):
            compare(self.a, self.b, 1e-12, 1e-8)

    def test_grid_mismatch(self):
        fixture(self.b, coords="0 0 0 0.6 5 0 1 0 0")
        with self.assertRaises(ValueError):
            compare(self.a, self.b, 1e-12, 1e-8)

    def test_encoding_rejected(self):
        fixture(self.b, encoding="binary")
        with self.assertRaises(ValueError):
            compare(self.a, self.b, 1e-12, 1e-8)


if __name__ == "__main__":
    unittest.main()
