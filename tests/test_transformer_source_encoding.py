"""Exercise the real notebook producer under a Windows legacy text fallback."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from qa import build_transformer_course as builder


class TransformerSourceEncodingTests(unittest.TestCase):
    def test_lecture_contract_preserves_python_exponents(self):
        self.assertEqual(builder.paragraph_text('**Squared error:** divide by `sum(truth**2)`.'),
                         'Squared error: divide by sum(truth**2).')
        self.assertEqual(builder.paragraph_text('floor**2/max(field**2,1e-20)>0.95'),
                         'floor**2/max(field**2,1e-20)&gt;0.95')
        self.assertEqual(builder.paragraph_text('**MSE:** `sum(error**2)/sum(truth**2)`'),
                         'MSE: sum(error**2)/sum(truth**2)')

    def test_both_editions_preserve_unicode_with_cp1252_default(self):
        repo = Path(__file__).resolve().parents[1]
        spec = json.loads((repo / 'course/lessons/week21.json').read_text(encoding='utf-8'))
        spec['reading_map'] += '\nUnicode contract: frames 160\u2013280; \u0394u and \u03c9.'
        spec['demos'][0]['code'] += '\n# Signed \u0394u; frames 160\u2013280.'
        temp_root = repo / 'tmp'
        temp_root.mkdir(exist_ok=True)
        original = Path.read_text

        def legacy_default(path, *args, **kwargs):
            # Python on some Windows hosts otherwise decodes UTF-8 as CP1252.
            if not args and kwargs.get('encoding') is None:
                kwargs['encoding'] = 'cp1252'
            return original(path, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            source = root / 'course/lessons/week21.json'
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(spec, ensure_ascii=False), encoding='utf-8', newline='\n')
            output = root / 'rendered'
            with patch.object(builder, 'ROOT', root), patch.object(Path, 'read_text', legacy_default):
                builder.main(['--output', str(output), '--notebooks-only', '--weeks', '21'])
            for solutions in (False, True):
                relative = ('instructor/week21/W21_Solutions.ipynb' if solutions else
                            'notebooks/week21/W21_CFD_Transformer.ipynb')
                actual = nbformat.read(output / relative, as_version=4)
                expected = builder.notebook(spec, solutions)
                self.assertEqual([c.source for c in actual.cells], [c.source for c in expected.cells])
                self.assertIn('frames 160\u2013280', actual.cells[1].source)
                self.assertNotIn('\u00e2\u20ac', '\n'.join(c.source for c in actual.cells))


if __name__ == '__main__':
    unittest.main()
