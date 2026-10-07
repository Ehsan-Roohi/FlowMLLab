"""Report safety for the exercise auditor; semantic checks live in its report."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from qa import audit_transformer_exercises as audit


class AuditReportSafety(unittest.TestCase):
    def test_default_prints_without_opening_output(self):
        output = io.StringIO()
        with patch.object(audit,'build_report',return_value={'passed':True}), \
             patch.object(Path,'open',side_effect=AssertionError('unexpected file write')), \
             contextlib.redirect_stdout(output):
            self.assertEqual(audit.main([]),0)
        self.assertEqual(json.loads(output.getvalue()),{'passed':True})

    def test_new_report_is_exclusive_and_failed_audit_is_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            target=Path(temp)/'report.json'
            with patch.object(audit,'build_report',return_value={'passed':False}), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(['--report',str(target)]),1)
            self.assertEqual(json.loads(target.read_text()),{'passed':False})
            original=target.read_bytes()
            with patch.object(audit,'build_report') as build, \
                 contextlib.redirect_stderr(io.StringIO()), \
                 self.assertRaises(SystemExit) as exc:
                audit.main(['--report',str(target)])
            self.assertEqual(exc.exception.code,2)
            build.assert_not_called()
            self.assertEqual(target.read_bytes(),original)


if __name__=='__main__':
    unittest.main()
