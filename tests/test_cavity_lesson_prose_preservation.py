import hashlib

import nbformat

from qa.validate_cavity_expansion import preserved_cell_source


def test_exact_prose_substitution_does_not_admit_unrelated_edits():
    old = nbformat.v4.new_markdown_cell('The added exercise compares the fields.', id='lesson')
    new = nbformat.v4.new_markdown_cell('This exercise compares the fields.', id='lesson')
    ledger = {'lesson.ipynb': {'lesson': {
        'before_sha256': hashlib.sha256(old.source.encode()).hexdigest(),
        'after_sha256': hashlib.sha256(new.source.encode()).hexdigest(),
    }}}
    assert preserved_cell_source('lesson.ipynb', old, new, ledger)
    assert not preserved_cell_source('other.ipynb', old, new, ledger)
    new.source += ' An unrecorded claim.'
    assert not preserved_cell_source('lesson.ipynb', old, new, ledger)


def test_prose_ledger_cannot_exempt_executable_source():
    old = nbformat.v4.new_code_cell('x = 1', id='lesson')
    new = nbformat.v4.new_code_cell('x = 2', id='lesson')
    ledger = {'lesson.ipynb': {'lesson': {
        'before_sha256': hashlib.sha256(old.source.encode()).hexdigest(),
        'after_sha256': hashlib.sha256(new.source.encode()).hexdigest(),
    }}}
    assert not preserved_cell_source('lesson.ipynb', old, new, ledger)
    assert preserved_cell_source('lesson.ipynb', old, old, {})
