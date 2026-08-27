"""Static guard: a name read at runtime must actually be bound somewhere.

WHY THIS EXISTS
---------------
`glowstar/training/retrain.py` built its summary dict with `info.n_test`. `info`
never existed - it was left behind when the gate moved to the rolling production
horizon and `gate_split` started returning `(train, test, origin)`.

The line sat at the very END of a four-minute nightly job, so nothing reached it
until 02:34 in production, and every symptom pointed AWAY from the cause:

  Saved model 20260826T023415 (test MAE=1.863, promoted=True)
  NameError: name 'info' is not defined

The model really was trained, gated and promoted. But the process exited
non-zero, so systemd failed the unit and silently skipped every later step: the
drift report, the nightly DATABASE BACKUP, and the ExecStartPost that restarts
the API onto the model it had just promoted. Two nights of backups were lost
and the client kept being served the previous day's model, while the log line
everyone reads said "promoted=True".

The whole 280-test suite passed throughout, because no test runs a full retrain.
A type checker or pyflakes would have caught it instantly, but neither is
installed and this must not add a dependency to the client's server - so the
check is built on the standard library's `symtable`, which does real scope
resolution (params, locals, closures, comprehensions, global/nonlocal).
"""

from __future__ import annotations

import builtins
import symtable
from pathlib import Path

PKG = Path(__file__).resolve().parents[1] / "glowstar"
_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__",
                                  "__spec__", "__loader__", "__builtins__"}


def _module_bindings(top: symtable.SymbolTable) -> set[str]:
    return {s.get_name() for s in top.get_symbols()
            if s.is_assigned() or s.is_imported() or s.is_parameter()}


def _walk(table: symtable.SymbolTable, module_names: set[str],
          out: list[tuple[str, str, int]], path: str) -> None:
    for sym in table.get_symbols():
        name = sym.get_name()
        # is_global() means the compiler resolved this to module/builtin scope.
        # If it is bound at neither, the read WILL raise NameError when reached.
        if sym.is_global() and not sym.is_assigned():
            # Skip compiler-injected dunders. CPython 3.14 synthesises names like
            # `__conditional_annotations__` (PEP 649) into every module that uses
            # `from __future__ import annotations`; they are never source-visible.
            if name.startswith("__") and name.endswith("__"):
                continue
            if name not in module_names and name not in _BUILTINS:
                out.append((path, name, table.get_lineno()))
    for child in table.get_children():
        _walk(child, module_names, out, path)


def test_no_name_is_read_without_ever_being_bound():
    findings: list[tuple[str, str, int]] = []
    files = sorted(PKG.rglob("*.py"))
    assert files, "found no source files to check - the path is wrong"

    for f in files:
        src = f.read_text(encoding="utf-8")
        top = symtable.symtable(src, str(f), "exec")
        _walk(top, _module_bindings(top), findings,
              str(f.relative_to(PKG.parent)))

    assert not findings, "undefined name(s) — these raise NameError when reached:\n" + \
        "\n".join(f"  {p}: '{n}' (scope near line {ln})" for p, n, ln in findings)
