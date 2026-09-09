"""Structural tests; these do not certify a Nektar++ trajectory."""
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
from prepare_week13_nektar import make_case
from run_week13_nektar_production import validate_vtu, field_time


class TestPilotGates(unittest.TestCase):
    def test_generator_parameters(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            make_case(tmp, dt=.00025, steps=4000, purpose='test')
            params = [p.text for p in ET.parse(Path(tmp)/'cavity.xml').findall('.//PARAMETERS/P')]
            self.assertIn('NumSteps = 4000', params)
            self.assertIn('TimeStep = 0.00025', params)

    def test_finite_fields_and_time(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            path = Path(tmp)/'test.xml'
            def write_fields(value):
                path.write_text('<VTKFile><Piece NumberOfPoints="2"><PointData>'+''.join(
                    f'<DataArray Name="{v}" format="ascii">0 {value}</DataArray>'
                    for v in ('u','v','p'))+'</PointData><Points><DataArray format="ascii">'
                    '0 0 0 1 0 0</DataArray></Points></Piece></VTKFile>')
            write_fields('1')
            validate_vtu(path)
            write_fields('nan')
            with self.assertRaises(ValueError):
                validate_vtu(path)
            path.write_text('<NEKTAR><Metadata><Time>1.0</Time></Metadata></NEKTAR>')
            self.assertEqual(field_time(path), 1.0)


if __name__ == '__main__':
    unittest.main()
