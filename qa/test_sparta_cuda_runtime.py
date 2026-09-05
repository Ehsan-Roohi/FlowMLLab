"""Linux shell regression for toolkit resolution; does not allocate or use a GPU."""
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

JOB = Path(__file__).resolve().parents[1] / 'cases/sparta_step/gpu_job.sh'
FUNCTION = re.search(r'flowml_cuda_runtime_dir\(\) \{\n.*?\n\}',
                     JOB.read_text(), re.S).group()


@unittest.skipIf(os.name == 'nt', 'Batch runtime path resolution is Linux-specific')
class CudaRuntimeTests(unittest.TestCase):
    def probe(self, root):
        return subprocess.run(['bash', '-c', FUNCTION +
                               '\nflowml_cuda_runtime_dir "$1"', 'test',
                               str(root / 'bin/nvcc')], text=True, capture_output=True)

    def test_missing_runtime_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'bin').mkdir(); (root / 'bin/nvcc').touch()
            result = self.probe(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('CUDA12_RUNTIME_NOT_IN_SELECTED_TOOLKIT', result.stderr)

    def test_toolkit_layouts_and_spaces(self):
        for layout in ('lib64', 'targets/x86_64-linux/lib'):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory(prefix='CUDA toolkit ') as tmp:
                root = Path(tmp); (root / 'bin').mkdir(); (root / 'bin/nvcc').touch()
                library = root / layout; library.mkdir(parents=True)
                (library / 'libcudart.so.12').touch()
                result = self.probe(root)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(Path(result.stdout.strip()), library.resolve())

    def test_external_library_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'toolkit'; (root / 'bin').mkdir(parents=True)
            (root / 'bin/nvcc').touch(); (root / 'lib64').mkdir()
            external = Path(tmp) / 'wrong-library'; external.touch()
            (root / 'lib64/libcudart.so.12').symlink_to(external)
            self.assertNotEqual(self.probe(root).returncode, 0)

    def test_in_toolkit_symlink_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'bin').mkdir(); (root / 'bin/nvcc').touch()
            library = root / 'targets/x86_64-linux/lib'; library.mkdir(parents=True)
            (library / 'libcudart.so.12').touch()
            (root / 'lib64').symlink_to(library, target_is_directory=True)
            result = self.probe(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(Path(result.stdout.strip()), library.resolve())

    def test_stub_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'bin').mkdir(); (root / 'bin/nvcc').touch()
            stub = root / 'lib64/stubs'; stub.mkdir(parents=True)
            (stub / 'libcudart.so.12').touch()
            (stub.parent / 'libcudart.so.12').symlink_to(stub / 'libcudart.so.12')
            self.assertNotEqual(self.probe(root).returncode, 0)


def check_binary(binary):
    """Read ldd on a trusted saved SPARTA binary; never execute the solver."""
    nvcc = shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('Load the declared CUDA module in a child shell first')
    directory = subprocess.check_output(['bash', '-c', FUNCTION +
        '\nflowml_cuda_runtime_dir "$1"', 'test', nvcc], text=True).strip()
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = directory + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    linked = subprocess.check_output(['ldd', str(Path(binary).resolve(strict=True))], env=env, text=True)
    if 'not found' in linked:
        raise RuntimeError(linked)
    match = re.search(r'libcudart\.so\.12\s+=>\s+(\S+)', linked)
    if not match or Path(match[1]).resolve() != (Path(directory) / 'libcudart.so.12').resolve():
        raise RuntimeError('Linked CUDA runtime differs from the selected toolkit')
    print(json.dumps({'runtime': str(Path(match[1]).resolve()),
                      'unresolved_libraries': False, 'solver_executed': False}))


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--check-binary':
        check_binary(sys.argv[2])
    else:
        unittest.main()
