"""Runtime implementation provenance against synthetic checkouts and wheels.

Every test builds its own temporary Git repository or installed-wheel tree, so
the verdicts do not depend on the state of the repository running the tests.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.util
import json
import marshal
import os
import py_compile
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import pytest

from btc_quarter_hour_engine.research import prospective_provenance as provenance
from btc_quarter_hour_engine.research.prospective_provenance import (
    ImplementationProvenanceError,
    VerifiedImplementation,
)

PACKAGE = provenance.PACKAGE_NAME
GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is required for checkout provenance tests")

SOURCES = {
    f"{PACKAGE}/__init__.py": b'"""Synthetic package."""\n',
    f"{PACKAGE}/research/__init__.py": b"",
    f"{PACKAGE}/research/runner.py": b"VALUE = 35\n",
    f"{PACKAGE}/data/sample.csv": b"a,b\n1,2\n",
}
MODULES = (
    (PACKAGE, "__init__.py", True),
    (f"{PACKAGE}.research", "research/__init__.py", True),
    (f"{PACKAGE}.research.runner", "research/runner.py", False),
)
OTHER_SHA = "f" * 40


def _write(root: Path, files: dict[str, bytes]) -> None:
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _git(root: Path, *args: str) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    completed = subprocess.run(
        [GIT, "-C", str(root), *args], check=True, capture_output=True, env=environment,
    )
    return completed.stdout.decode("utf-8").strip()


def _make_repo(root: Path, files: dict[str, bytes] = SOURCES) -> str:
    root.mkdir(parents=True)
    _git(root, "init", "-q")
    for key, value in (
        ("core.autocrlf", "false"), ("core.safecrlf", "false"), ("commit.gpgsign", "false"),
        ("user.name", "Provenance Test"), ("user.email", "provenance@example.invalid"),
    ):
        _git(root, "config", key, value)
    _write(root, files)
    (root / ".gitignore").write_bytes(b"__pycache__/\n*.pyc\n*.log\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--no-verify", "-m", "synthetic implementation")
    return _git(root, "rev-parse", "HEAD")


def _load(name: str, path: Path, *, package: bool = False) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_modules(package_root: Path) -> dict[str, types.ModuleType]:
    """Import the synthetic package with the real source loader (not into sys.modules)."""

    return {
        name: _load(name, package_root / relative, package=is_package)
        for name, relative, is_package in MODULES
    }


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "checkout"
    head = _make_repo(root)
    return root, root / PACKAGE, head


@pytest.fixture
def plain_package(tmp_path):
    root = tmp_path / "plain"
    _write(root, SOURCES)
    return root / PACKAGE


def _source_identity(package_root: Path, git_sha: str = "a" * 40) -> VerifiedImplementation:
    files = provenance.snapshot_package_files(package_root)
    return VerifiedImplementation(
        mode=provenance.MODE_GIT_CHECKOUT, git_sha=git_sha, package_root=package_root,
        files=files, package_content_sha256=provenance.package_content_digest(files),
        source_root=package_root.parent,
    )


def _record_hash(data: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")


def _build_wheel(
    directory: Path,
    files: dict[str, bytes] = SOURCES,
    *,
    version: str = "0.1.0",
    filename: str | None = None,
    extra: dict[str, bytes] | None = None,
    corrupt_record: bool = False,
) -> Path:
    dist_info = f"{PACKAGE}-{version}.dist-info"
    entries = dict(files)
    entries[f"{dist_info}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: btc-quarter-hour-engine\nVersion: {version}\n"
    ).encode("ascii")
    entries[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: provenance-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    entries.update(extra or {})
    rows = [f"{path},{_record_hash(data)},{len(data)}" for path, data in entries.items()]
    if corrupt_record:
        rows[0] = rows[0].rsplit(",", 1)[0] + ",999"
    rows.append(f"{dist_info}/RECORD,,")
    entries[f"{dist_info}/RECORD"] = ("\n".join(rows) + "\n").encode("ascii")
    directory.mkdir(parents=True, exist_ok=True)
    wheel = directory / (filename or f"{PACKAGE}-{version}-py3-none-any.whl")
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, data in entries.items():
            archive.writestr(path, data)
    return wheel


class Artifact(types.SimpleNamespace):
    pass


@pytest.fixture
def artifact(tmp_path, plain_package):
    """A wheel built from the verified source, attested, then installed into a site directory."""

    wheel = _build_wheel(tmp_path / "dist")
    report = provenance.build_implementation_attestation(
        wheel, tmp_path / "attestation.json", source=_source_identity(plain_package),
    )
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    return Artifact(
        wheel=wheel, site=site, package=site / PACKAGE, report=report,
        path=Path(report["attestation_path"]), sha256=report["attestation_sha256"],
        dist_info=site / f"{PACKAGE}-0.1.0.dist-info",
    )


def _verify_artifact(artifact, *, expected: str = "a" * 40, path=None, sha256=None, modules=None):
    return provenance.verify_installed_artifact(
        artifact.package, expected, artifact.path if path is None else path,
        artifact.sha256 if sha256 is None else sha256,
        modules=_load_modules(artifact.package) if modules is None else modules,
        search_path=[str(artifact.site)],
    )


def _attestation_payload(**changes) -> bytes:
    files = {path: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)} for path, data in SOURCES.items()}
    document = {
        "schema": provenance.ATTESTATION_SCHEMA,
        "distribution": provenance.DISTRIBUTION_NAME,
        "version": "0.1.0",
        "package": PACKAGE,
        "git_sha": "a" * 40,
        "wheel_filename": f"{PACKAGE}-0.1.0-py3-none-any.whl",
        "wheel_sha256": "b" * 64,
        "files": files,
        "package_content_sha256": provenance.package_content_digest(
            (path, entry["sha256"], entry["size"]) for path, entry in files.items()
        ),
    }
    document.update(changes)
    return json.dumps(document).encode("utf-8")


# ---------------------------------------------------------------- Git checkout


@needs_git
def test_clean_checkout_identity_is_the_actual_head(checkout):
    root, package, head = checkout
    identity = provenance.verify_git_checkout(package, head, modules=_load_modules(package))
    assert identity.mode == provenance.MODE_GIT_CHECKOUT
    assert identity.git_sha == head
    assert os.path.samefile(identity.source_root, root)
    assert [path for path, _, _ in identity.files] == sorted(SOURCES)
    assert identity.package_content_sha256 == provenance.package_content_digest(identity.files)
    assert identity.as_dict()["package_file_count"] == len(SOURCES)


@needs_git
def test_runtime_dispatch_uses_git_checkout_without_attestation(checkout):
    _, package, head = checkout
    identity = provenance.verify_runtime_implementation(head, modules=_load_modules(package))
    assert identity.mode == provenance.MODE_GIT_CHECKOUT and identity.git_sha == head


@needs_git
def test_unexpected_commit_is_rejected(checkout):
    root, package, head = checkout
    modules = _load_modules(package)
    with pytest.raises(ImplementationProvenanceError, match="does not match expected implementation"):
        provenance.verify_git_checkout(package, OTHER_SHA, modules=modules)
    (package / "research" / "runner.py").write_bytes(b"VALUE = 36\n")
    _git(root, "commit", "-q", "--no-verify", "-am", "later implementation")
    with pytest.raises(ImplementationProvenanceError, match=f"HEAD [0-9a-f]{{40}} does not match expected implementation {head}"):
        provenance.verify_git_checkout(package, head, modules=_load_modules(package))


@needs_git
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("modified", "Git working tree is not clean"),
        ("staged", "Git working tree is not clean"),
        ("deleted", "Git working tree is not clean"),
        ("untracked_in_package", "Git working tree is not clean"),
        ("untracked_outside_package", "Git working tree is not clean"),
        ("ignored_in_package", "package contains files not in"),
        ("ignored_bytecode_outside_cache", "package contains files not in"),
        ("assume_unchanged", "assume-unchanged/skip-worktree"),
        ("skip_worktree", "assume-unchanged/skip-worktree"),
    ],
)
def test_checkout_modifications_are_rejected(checkout, change, message):
    root, package, head = checkout
    modules = _load_modules(package)
    runner = package / "research" / "runner.py"
    relative = f"{PACKAGE}/research/runner.py"
    if change == "modified":
        runner.write_bytes(b"VALUE = 0\n")
    elif change == "staged":
        runner.write_bytes(b"VALUE = 0\n")
        _git(root, "add", relative)
    elif change == "deleted":
        runner.unlink()
    elif change == "untracked_in_package":
        (package / "research" / "extra.py").write_bytes(b"EXTRA = 1\n")
    elif change == "untracked_outside_package":
        (root / "notes.txt").write_bytes(b"runtime data must live outside the checkout\n")
    elif change == "ignored_in_package":
        (package / "research" / "debug.log").write_bytes(b"ignored but inside the package\n")
    elif change == "ignored_bytecode_outside_cache":
        (package / "research" / "runner.pyc").write_bytes(b"\0" * 16)
    elif change == "assume_unchanged":
        _git(root, "update-index", "--assume-unchanged", relative)
        runner.write_bytes(b"VALUE = 0\n")
    else:
        _git(root, "update-index", "--skip-worktree", relative)
        runner.write_bytes(b"VALUE = 0\n")
    with pytest.raises(ImplementationProvenanceError, match=message):
        provenance.verify_git_checkout(package, head, modules=modules)


@needs_git
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("modified", "package files differ from"),
        ("missing", "package is missing files from"),
        ("extra", "package contains files not in"),
    ],
)
def test_package_content_is_rehashed_even_if_git_status_reports_clean(checkout, monkeypatch, change, message):
    _, package, head = checkout
    modules = _load_modules(package)
    real_run_git = provenance._run_git

    def status_says_clean(git, root, *arguments):
        return b"" if arguments[0] == "status" else real_run_git(git, root, *arguments)

    monkeypatch.setattr(provenance, "_run_git", status_says_clean)
    if change == "modified":
        (package / "research" / "runner.py").write_bytes(b"VALUE = 0\n")
    elif change == "missing":
        (package / "data" / "sample.csv").unlink()
    else:
        (package / "research" / "extra.py").write_bytes(b"EXTRA = 1\n")
    with pytest.raises(ImplementationProvenanceError, match=message):
        provenance.verify_git_checkout(package, head, modules=modules)


@needs_git
def test_crlf_checkout_of_the_same_commit_is_accepted(checkout):
    root, package, head = checkout
    _git(root, "config", "core.autocrlf", "true")
    shutil.rmtree(package)
    _git(root, "checkout", "--", ".")
    assert b"\r\n" in (package / "data" / "sample.csv").read_bytes()
    identity = provenance.verify_git_checkout(package, head, modules=_load_modules(package))
    assert identity.git_sha == head


@needs_git
def test_git_environment_overrides_cannot_redirect_verification(checkout, tmp_path, monkeypatch):
    _, package, head = checkout
    other = tmp_path / "other"
    other_head = _make_repo(other, {**SOURCES, f"{PACKAGE}/research/runner.py": b"VALUE = 99\n"})
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    identity = provenance.verify_git_checkout(package, head, modules=_load_modules(package))
    assert identity.git_sha == head != other_head
    with pytest.raises(ImplementationProvenanceError, match="does not match expected implementation"):
        provenance.verify_git_checkout(package, other_head, modules=_load_modules(package))


@needs_git
def test_unavailable_git_fails_closed(checkout, tmp_path, monkeypatch):
    _, package, head = checkout
    modules = _load_modules(package)
    with pytest.raises(ImplementationProvenanceError, match="could not run"):
        provenance.verify_git_checkout(
            package, head, modules=modules, git_executable=str(tmp_path / "missing-git"),
        )
    monkeypatch.setattr(provenance.shutil, "which", lambda name: None)
    with pytest.raises(ImplementationProvenanceError, match="git executable is unavailable"):
        provenance.verify_git_checkout(package, head, modules=modules)


def test_package_outside_a_checkout_is_unverifiable(plain_package):
    modules = _load_modules(plain_package)
    with pytest.raises(ImplementationProvenanceError, match="not the top-level package of a Git checkout"):
        provenance.verify_git_checkout(plain_package, "a" * 40, modules=modules)
    with pytest.raises(ImplementationProvenanceError, match="identity is unverifiable"):
        provenance.verify_runtime_implementation("a" * 40, modules=modules)


@needs_git
def test_nested_package_in_a_checkout_is_not_top_level(tmp_path):
    root = tmp_path / "nested"
    head = _make_repo(root, {f"src/{path}": data for path, data in SOURCES.items()})
    package = root / "src" / PACKAGE
    with pytest.raises(ImplementationProvenanceError, match="not the top-level package"):
        provenance.verify_git_checkout(package, head, modules=_load_modules(package))


# ------------------------------------------------------------ loaded modules


def test_loaded_modules_must_come_from_verified_source_files(plain_package):
    files = provenance.snapshot_package_files(plain_package)
    modules = _load_modules(plain_package)
    assert provenance.imported_package_root(modules) == Path(os.path.abspath(plain_package))
    assert provenance.verify_loaded_modules(plain_package, files, modules=modules) == len(MODULES)
    # A package module executed as __main__ is checked under its real name.
    assert provenance.verify_loaded_modules(
        plain_package, files, modules={**modules, "__main__": modules[f"{PACKAGE}.research.runner"]},
    ) == len(MODULES) + 1


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("foreign_origin", "originates outside the verified package"),
        ("renamed_module", "originates from unverified file"),
        ("unverified_file", "originates from unverified file"),
        ("sourceless_loader", "was not imported from a source file"),
        ("file_mismatch", "__file__ does not match its origin"),
        ("altered_search_path", "altered search path"),
        ("unchecked_hash_bytecode", "unchecked hash-based bytecode cache"),
        ("custom_source_loader", "was not imported from a source file"),
        ("hidden_cache_location", "inconsistent bytecode cache location"),
        ("redirected_cache_location", "inconsistent bytecode cache location"),
        ("missing_root", "is not among the loaded modules"),
    ],
)
def test_unverified_loaded_modules_are_rejected(plain_package, tmp_path, case, message):
    files = provenance.snapshot_package_files(plain_package)
    modules = _load_modules(plain_package)
    runner_name = f"{PACKAGE}.research.runner"
    if case == "foreign_origin":
        foreign = tmp_path / "elsewhere" / "rogue.py"
        _write(foreign.parent, {"rogue.py": b"ROGUE = 1\n"})
        modules[f"{PACKAGE}.research.rogue"] = _load(f"{PACKAGE}.research.rogue", foreign)
    elif case == "renamed_module":
        modules[f"{PACKAGE}.research.other"] = _load(
            f"{PACKAGE}.research.other", plain_package / "research" / "runner.py",
        )
    elif case == "unverified_file":
        late = plain_package / "research" / "late.py"
        late.write_bytes(b"LATE = 1\n")
        modules[f"{PACKAGE}.research.late"] = _load(f"{PACKAGE}.research.late", late)
    elif case == "sourceless_loader":
        name = f"{PACKAGE}.research.compiled"
        origin = str(plain_package / "research" / "compiled.pyc")
        spec = importlib.machinery.ModuleSpec(
            name, importlib.machinery.SourcelessFileLoader(name, origin), origin=origin,
        )
        modules[name] = importlib.util.module_from_spec(spec)
    elif case == "file_mismatch":
        modules[runner_name].__file__ = str(plain_package / "__init__.py")
    elif case == "altered_search_path":
        modules[f"{PACKAGE}.research"].__path__ = [str(tmp_path)]
    elif case == "unchecked_hash_bytecode":
        spec = modules[runner_name].__spec__
        py_compile.compile(
            spec.origin, cfile=spec.cached, doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
        )
    elif case == "custom_source_loader":
        class SubstitutedLoader(importlib.machinery.SourceFileLoader):
            def get_code(self, fullname):
                return compile("VALUE = 99", self.path, "exec", dont_inherit=True)

        module = modules[runner_name]
        module.__spec__.loader = SubstitutedLoader(runner_name, module.__file__)
        module.__spec__.loader.exec_module(module)
        assert module.VALUE == 99
    elif case == "hidden_cache_location":
        modules[runner_name].__cached__ = None
    elif case == "redirected_cache_location":
        modules[runner_name].__spec__.cached = str(tmp_path / "unrelated.pyc")
    else:
        del modules[PACKAGE]
    with pytest.raises(ImplementationProvenanceError, match=message):
        provenance.verify_loaded_modules(plain_package, files, modules=modules)


def test_checked_hash_bytecode_is_accepted(plain_package):
    files = provenance.snapshot_package_files(plain_package)
    modules = _load_modules(plain_package)
    spec = modules[f"{PACKAGE}.research.runner"].__spec__
    py_compile.compile(
        spec.origin, cfile=spec.cached, doraise=True,
        invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
    )
    assert provenance.verify_loaded_modules(plain_package, files, modules=modules) == len(MODULES)


def _substitute_cached_code(package_root, invalidation_mode):
    source = package_root / "research" / "runner.py"
    cached = Path(importlib.util.cache_from_source(str(source)))
    py_compile.compile(str(source), doraise=True, invalidation_mode=invalidation_mode)
    header = cached.read_bytes()[:16]
    replacement = compile(b"VALUE = 99\n", str(source), "exec", dont_inherit=True)
    cached.write_bytes(header + marshal.dumps(replacement))
    return cached


@needs_git
@pytest.mark.parametrize("invalidation_mode", [
    py_compile.PycInvalidationMode.TIMESTAMP,
    py_compile.PycInvalidationMode.CHECKED_HASH,
])
def test_cached_payload_cannot_override_verified_checkout(checkout, invalidation_mode):
    root, package, head = checkout
    _substitute_cached_code(package, invalidation_mode)
    modules = _load_modules(package)
    assert modules[f"{PACKAGE}.research.runner"].VALUE == 99
    assert (package / "research" / "runner.py").read_bytes() == SOURCES[f"{PACKAGE}/research/runner.py"]
    assert _git(root, "status", "--porcelain") == ""
    with pytest.raises(ImplementationProvenanceError, match="bytecode"):
        provenance.verify_git_checkout(package, head, modules=modules)


@pytest.mark.parametrize("invalidation_mode", [
    py_compile.PycInvalidationMode.TIMESTAMP,
    py_compile.PycInvalidationMode.CHECKED_HASH,
])
def test_cached_payload_cannot_override_attested_install(artifact, invalidation_mode):
    _substitute_cached_code(artifact.package, invalidation_mode)
    modules = _load_modules(artifact.package)
    assert modules[f"{PACKAGE}.research.runner"].VALUE == 99
    with pytest.raises(ImplementationProvenanceError, match="bytecode"):
        _verify_artifact(artifact, modules=modules)


def test_unimported_module_cache_is_verified_before_it_can_be_used(plain_package):
    files = provenance.snapshot_package_files(plain_package)
    modules = _load_modules(plain_package)
    del modules[f"{PACKAGE}.research.runner"]
    _substitute_cached_code(plain_package, py_compile.PycInvalidationMode.TIMESTAMP)
    with pytest.raises(ImplementationProvenanceError, match="bytecode cache payload differs"):
        provenance.verify_loaded_modules(plain_package, files, modules=modules)


def test_boundary_recheck_detects_cache_drift_without_source_changes(plain_package):
    identity = _source_identity(plain_package)
    modules = _load_modules(plain_package)
    assert provenance.recheck_runtime_implementation(identity, modules=modules) is identity
    _substitute_cached_code(plain_package, py_compile.PycInvalidationMode.TIMESTAMP)
    assert provenance.snapshot_package_files(plain_package) == identity.files
    with pytest.raises(ImplementationProvenanceError, match="bytecode cache payload differs"):
        provenance.recheck_runtime_implementation(identity, modules=modules)


def test_external_pycache_prefix_is_verified(plain_package, tmp_path, monkeypatch):
    prefix = str(tmp_path / "external-cache")
    if os.name == "nt":
        # Prefixes mirror absolute source paths, easily exceeding MAX_PATH.
        prefix = "\\\\?\\" + prefix
    monkeypatch.setattr(sys, "pycache_prefix", prefix)
    files = provenance.snapshot_package_files(plain_package)
    modules = _load_modules(plain_package)
    assert provenance.verify_loaded_modules(plain_package, files, modules=modules) == len(MODULES)
    cached = _substitute_cached_code(plain_package, py_compile.PycInvalidationMode.TIMESTAMP)
    assert not cached.is_relative_to(plain_package)
    with pytest.raises(ImplementationProvenanceError, match="bytecode cache payload differs"):
        provenance.verify_loaded_modules(plain_package, files, modules=modules)


def test_installer_staging_filename_does_not_change_valid_code(plain_package, tmp_path):
    source = plain_package / "research" / "runner.py"
    py_compile.compile(
        str(source), dfile=str(tmp_path / "old-install-staging" / "runner.py"), doraise=True,
    )
    modules = _load_modules(plain_package)
    files = provenance.snapshot_package_files(plain_package)
    assert provenance.verify_loaded_modules(plain_package, files, modules=modules) == len(MODULES)


def test_nested_code_constants_are_verified(plain_package):
    source = plain_package / "research" / "runner.py"
    source.write_bytes(b"VALUE = 35\ndef answer():\n    return 35\n")
    files = provenance.snapshot_package_files(plain_package)
    py_compile.compile(str(source), doraise=True)
    cached = Path(importlib.util.cache_from_source(str(source)))
    replacement = compile(b"VALUE = 35\ndef answer():\n    return 99\n", str(source), "exec", dont_inherit=True)
    cached.write_bytes(cached.read_bytes()[:16] + marshal.dumps(replacement))
    modules = _load_modules(plain_package)
    assert modules[f"{PACKAGE}.research.runner"].VALUE == 35
    assert modules[f"{PACKAGE}.research.runner"].answer() == 99
    with pytest.raises(ImplementationProvenanceError, match="bytecode cache payload differs"):
        provenance.verify_loaded_modules(plain_package, files, modules=modules)


@pytest.mark.parametrize("level", ["module", "function"])
def test_cached_execution_metadata_is_verified_recursively(plain_package, level):
    source = plain_package / "research" / "runner.py"
    source.write_bytes(b"def answer():\n    return 35\n")
    files = provenance.snapshot_package_files(plain_package)
    modules = _load_modules(plain_package)
    cached = Path(importlib.util.cache_from_source(str(source)))
    original = cached.read_bytes()
    code = marshal.loads(original[16:])
    nested = next(value for value in code.co_consts if isinstance(value, types.CodeType))
    target = code if level == "module" else nested
    replacements = [
        target.replace(co_stacksize=target.co_stacksize + 1),
        target.replace(co_linetable=target.co_linetable + b"\0"),
    ]
    if level == "function":
        replacements.append(target.replace(co_filename="unverified-nested-origin"))
    if hasattr(target, "co_qualname"):
        replacements.append(target.replace(co_qualname="unverified_name"))
    if hasattr(target, "co_exceptiontable"):
        replacements.append(target.replace(co_exceptiontable=target.co_exceptiontable + b"\0"))
    for replacement in replacements:
        altered = replacement if level == "module" else code.replace(
            co_consts=tuple(replacement if value is nested else value for value in code.co_consts),
        )
        cached.write_bytes(original[:16] + marshal.dumps(altered))
        with pytest.raises(ImplementationProvenanceError, match="bytecode cache payload differs"):
            provenance.verify_loaded_modules(plain_package, files, modules=modules)


@pytest.mark.parametrize("damage", [
    "magic", "truncated_header", "invalid_flags", "non_code", "invalid_marshal",
    "truncated_payload", "trailing_payload", "wrong_source_hash",
])
def test_unverifiable_cache_is_rejected(plain_package, damage):
    files = provenance.snapshot_package_files(plain_package)
    modules = _load_modules(plain_package)
    cached = Path(modules[f"{PACKAGE}.research.runner"].__cached__)
    data = cached.read_bytes()
    if damage == "magic":
        data = b"\0" * 4 + data[4:]
    elif damage == "truncated_header":
        data = data[:10]
    elif damage == "invalid_flags":
        data = data[:4] + (2).to_bytes(4, "little") + data[8:]
    elif damage == "non_code":
        data = data[:16] + marshal.dumps(35)
    elif damage == "invalid_marshal":
        data = data[:16] + b"?"
    elif damage == "truncated_payload":
        data = data[:17]
    elif damage == "trailing_payload":
        data += b"unattested data"
    elif damage == "wrong_source_hash":
        data = data[:4] + (3).to_bytes(4, "little") + b"\0" * 8 + data[16:]
    cached.write_bytes(data)
    with pytest.raises(ImplementationProvenanceError, match="bytecode"):
        provenance.verify_loaded_modules(plain_package, files, modules=modules)


@pytest.mark.parametrize("optimization", [0, 1, 2])
def test_cache_compilation_matches_interpreter_optimization(plain_package, optimization):
    source = plain_package / "research" / "runner.py"
    source.write_bytes(
        b'"""Module docs."""\ndef answer():\n    """Function docs."""\n'
        b'    assert False, "kept without optimization"\n    return 35\n'
    )
    script = """
import importlib.util
from pathlib import Path
import py_compile
import sys
from tests.test_prospective_provenance import _load_modules, MODULES
from btc_quarter_hour_engine.research import prospective_provenance as p
root = Path(sys.argv[1])
files = p.snapshot_package_files(root)
modules = _load_modules(root)
if p.verify_loaded_modules(root, files, modules=modules) != len(MODULES):
    raise AssertionError('valid optimized bytecode was rejected')
source = root / 'research' / 'runner.py'
py_compile.compile(str(source), cfile=importlib.util.cache_from_source(str(source)),
                   optimize=1 if sys.flags.optimize == 0 else 0, doraise=True)
try:
    p.verify_loaded_modules(root, files, modules=_load_modules(root))
except p.ImplementationProvenanceError as exc:
    if 'bytecode cache payload differs' not in str(exc):
        raise AssertionError(str(exc))
else:
    raise AssertionError('wrong optimization bytecode was accepted')
"""
    result = subprocess.run(
        [sys.executable, *([] if optimization == 0 else ["-" + "O" * optimization]),
         "-c", script, str(plain_package)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_package_inventory_rejects_unexpected_cache_content(plain_package):
    cache = plain_package / "research" / "__pycache__"
    cache.mkdir()
    (cache / "runner.cpython-310.pyc").write_bytes(b"\0" * 16)
    assert [path for path, _, _ in provenance.snapshot_package_files(plain_package)] == sorted(SOURCES)
    (cache / "notes.txt").write_bytes(b"not bytecode")
    with pytest.raises(ImplementationProvenanceError, match="unexpected bytecode-cache entry"):
        provenance.snapshot_package_files(plain_package)


# ------------------------------------------------------- installed artifact


def test_attested_installed_wheel_is_verified_without_git(artifact):
    assert not (artifact.site / ".git").exists()
    identity = _verify_artifact(artifact)
    assert identity.mode == provenance.MODE_INSTALLED_ARTIFACT
    assert identity.git_sha == "a" * 40
    assert identity.attestation_sha256 == artifact.sha256
    assert identity.wheel_sha256 == hashlib.sha256(artifact.wheel.read_bytes()).hexdigest()
    assert identity.package_content_sha256 == artifact.report["package_content_sha256"]
    assert artifact.report["package_file_count"] == len(SOURCES)


def test_runtime_dispatch_uses_configured_attestation(artifact, monkeypatch):
    monkeypatch.syspath_prepend(str(artifact.site))
    identity = provenance.verify_runtime_implementation(
        "a" * 40, attestation_path=artifact.path, attestation_sha256=artifact.sha256,
        modules=_load_modules(artifact.package),
    )
    assert identity.mode == provenance.MODE_INSTALLED_ARTIFACT


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong_pinned_sha256", "does not match pinned"),
        ("missing_attestation", "is missing"),
        ("relative_attestation_path", "must be absolute"),
        ("unexpected_commit", "does not match expected"),
        ("modified_installed_file", "installed package files differ from attestation"),
        ("extra_installed_file", "installed package has unattested files"),
        ("missing_installed_file", "installed package is missing attested files"),
        ("version_mismatch", "version does not match the attestation"),
        ("record_hash_mismatch", "RECORD disagrees"),
        ("record_missing_entry", "RECORD does not match the attested files"),
        ("no_distribution_metadata", "found 0"),
        ("duplicate_distribution", "found 2"),
        ("editable_install", "editable installs must be verified in Git checkout mode"),
        ("archive_hash_mismatch", "installed archive hash does not match"),
    ],
)
def test_mismatched_installed_artifact_is_rejected(artifact, tmp_path, case, message):
    modules = _load_modules(artifact.package)
    kwargs = {"modules": modules}
    record = artifact.dist_info / "RECORD"
    runner = artifact.package / "research" / "runner.py"
    if case == "wrong_pinned_sha256":
        kwargs["sha256"] = "0" * 64
    elif case == "missing_attestation":
        kwargs["path"] = tmp_path / "absent.json"
    elif case == "relative_attestation_path":
        kwargs["path"] = Path("attestation.json")
    elif case == "unexpected_commit":
        kwargs["expected"] = OTHER_SHA
    elif case == "modified_installed_file":
        runner.write_bytes(b"VALUE = 0\n")
    elif case == "extra_installed_file":
        (artifact.package / "research" / "extra.py").write_bytes(b"EXTRA = 1\n")
    elif case == "missing_installed_file":
        (artifact.package / "data" / "sample.csv").unlink()
    elif case == "version_mismatch":
        metadata = artifact.dist_info / "METADATA"
        metadata.write_bytes(metadata.read_bytes().replace(b"Version: 0.1.0", b"Version: 0.2.0"))
    elif case == "record_hash_mismatch":
        text = record.read_text("ascii")
        record.write_text(text.replace(_record_hash(SOURCES[f"{PACKAGE}/research/runner.py"]), _record_hash(b"x")), "ascii")
    elif case == "record_missing_entry":
        rows = [row for row in record.read_text("ascii").splitlines() if not row.startswith(f"{PACKAGE}/data/")]
        record.write_text("\n".join(rows) + "\n", "ascii")
    elif case == "no_distribution_metadata":
        shutil.rmtree(artifact.dist_info)
    elif case == "duplicate_distribution":
        duplicate = artifact.site / f"{PACKAGE}-0.1.1.dist-info"
        shutil.copytree(artifact.dist_info, duplicate)
        metadata = duplicate / "METADATA"
        metadata.write_bytes(metadata.read_bytes().replace(b"Version: 0.1.0", b"Version: 0.1.1"))
    elif case == "editable_install":
        (artifact.dist_info / "direct_url.json").write_text(
            json.dumps({"url": "file:///src", "dir_info": {"editable": True}}), "utf-8",
        )
    else:
        (artifact.dist_info / "direct_url.json").write_text(
            json.dumps({"url": "file:///dist/x.whl", "archive_info": {"hashes": {"sha256": "0" * 64}}}), "utf-8",
        )
    with pytest.raises(ImplementationProvenanceError, match=message):
        _verify_artifact(artifact, **kwargs)


def test_matching_archive_hash_in_direct_url_is_accepted(artifact):
    wheel_sha256 = hashlib.sha256(artifact.wheel.read_bytes()).hexdigest()
    (artifact.dist_info / "direct_url.json").write_text(json.dumps({
        "url": "file:///dist/x.whl",
        "archive_info": {"hash": f"sha256={wheel_sha256}", "hashes": {"sha256": wheel_sha256}},
    }), "utf-8")
    assert _verify_artifact(artifact).wheel_sha256 == wheel_sha256


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b'{"schema": "a", "schema": "b"}', "duplicate key"),
        (b'{"size": NaN}', "non-finite"),
        (b"not json", "not valid JSON"),
        (b'{"schema": "x"}', "unexpected structure"),
        (_attestation_payload(schema="btc-qh-implementation-attestation/v0"), "schema is not supported"),
        (_attestation_payload(distribution="other"), "names a different package"),
        (_attestation_payload(git_sha="A" * 40), "attested implementation Git SHA"),
        (_attestation_payload(wheel_filename="other-0.1.0-py3-none-any.whl"), "wheel filename is invalid"),
        (_attestation_payload(package_content_sha256="0" * 64), "digest is inconsistent"),
        (
            _attestation_payload(files={f"{PACKAGE}/../evil.py": {"sha256": "0" * 64, "size": 1}}),
            "is invalid",
        ),
        (
            _attestation_payload(files={f"{PACKAGE}/x.py": {"sha256": "0" * 64, "size": True}}),
            "size for",
        ),
    ],
)
def test_attestation_parsing_is_strict(payload, message):
    with pytest.raises(ImplementationProvenanceError, match=message):
        provenance.parse_implementation_attestation(payload, hashlib.sha256(payload).hexdigest())


def test_attestation_is_checked_against_pin_before_parsing():
    payload = _attestation_payload()
    parsed = provenance.parse_implementation_attestation(payload, hashlib.sha256(payload).hexdigest())
    assert parsed.git_sha == "a" * 40 and len(parsed.files) == len(SOURCES)
    with pytest.raises(ImplementationProvenanceError, match="does not match pinned"):
        provenance.parse_implementation_attestation(b"not json", "0" * 64)


# ------------------------------------------------------- build attestation


def test_attestation_round_trip_is_exclusive_and_canonical(artifact, plain_package):
    payload = artifact.path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == artifact.sha256
    document = json.loads(payload)
    assert document["git_sha"] == "a" * 40
    assert document["wheel_filename"] == artifact.wheel.name
    assert set(document["files"]) == set(SOURCES)
    with pytest.raises(ImplementationProvenanceError, match="already exists"):
        provenance.build_implementation_attestation(
            artifact.wheel, artifact.path, source=_source_identity(plain_package),
        )
    assert artifact.path.read_bytes() == payload


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("artifact_mode_source", "only be built from a verified Git checkout"),
        ("different_wheel_content", "wheel package contents differ from verified checkout"),
        ("unexpected_top_level_entry", "wheel contains unexpected entry"),
        ("corrupt_record", "wheel RECORD disagrees"),
        ("filename_version_mismatch", "does not match its METADATA"),
        ("not_a_wheel", "not a valid archive"),
    ],
)
def test_attestation_builder_rejects_mismatched_wheels(tmp_path, plain_package, case, message):
    source = _source_identity(plain_package)
    dist = tmp_path / "dist"
    if case == "artifact_mode_source":
        wheel = _build_wheel(dist)
        source = VerifiedImplementation(**{**source.__dict__, "mode": provenance.MODE_INSTALLED_ARTIFACT})
    elif case == "different_wheel_content":
        wheel = _build_wheel(dist, {**SOURCES, f"{PACKAGE}/research/runner.py": b"VALUE = 0\n"})
    elif case == "unexpected_top_level_entry":
        wheel = _build_wheel(dist, extra={"sitecustomize.py": b"import os\n"})
    elif case == "corrupt_record":
        wheel = _build_wheel(dist, corrupt_record=True)
    elif case == "filename_version_mismatch":
        wheel = _build_wheel(dist, filename=f"{PACKAGE}-9.9.9-py3-none-any.whl")
    else:
        dist.mkdir()
        wheel = dist / f"{PACKAGE}-0.1.0-py3-none-any.whl"
        wheel.write_bytes(b"not a zip archive")
    output = tmp_path / "attestation.json"
    with pytest.raises(ImplementationProvenanceError, match=message):
        provenance.build_implementation_attestation(wheel, output, source=source)
    assert not output.exists()


@needs_git
def test_attestation_from_real_checkout_verifies_the_installed_wheel(checkout, tmp_path):
    _, package, head = checkout
    source = provenance.verify_git_checkout(package, head, modules=_load_modules(package))
    wheel = _build_wheel(tmp_path / "dist")
    report = provenance.build_implementation_attestation(wheel, tmp_path / "attestation.json", source=source)
    site = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(site)
    installed = provenance.verify_installed_artifact(
        site / PACKAGE, head, Path(report["attestation_path"]), report["attestation_sha256"],
        modules=_load_modules(site / PACKAGE), search_path=[str(site)],
    )
    assert installed.git_sha == head
    assert installed.package_content_sha256 == source.package_content_sha256


# ------------------------------------------------------ runtime entry points


def test_recheck_detects_drift_after_verification(plain_package, tmp_path):
    identity = _source_identity(plain_package)
    modules = _load_modules(plain_package)
    assert provenance.recheck_runtime_implementation(identity, modules=modules) is identity
    (plain_package / "research" / "runner.py").write_bytes(b"VALUE = 0\n")
    with pytest.raises(ImplementationProvenanceError, match="package files changed after verification"):
        provenance.recheck_runtime_implementation(identity, modules=modules)
    moved = tmp_path / "moved"
    _write(moved, SOURCES)
    with pytest.raises(ImplementationProvenanceError, match="location changed"):
        provenance.recheck_runtime_implementation(identity, modules=_load_modules(moved / PACKAGE))
    with pytest.raises(ImplementationProvenanceError, match="no verified implementation"):
        provenance.recheck_runtime_implementation(None)


def test_runtime_configuration_errors_fail_closed(plain_package, monkeypatch):
    modules = _load_modules(plain_package)
    with pytest.raises(ImplementationProvenanceError, match="40 lowercase"):
        provenance.verify_runtime_implementation("HEAD", modules=modules)
    with pytest.raises(ImplementationProvenanceError, match="configured together"):
        provenance.verify_runtime_implementation(
            "a" * 40, attestation_path=Path(os.path.abspath("attestation.json")), modules=modules,
        )
    with pytest.raises(ImplementationProvenanceError, match="configured together"):
        provenance.verify_runtime_implementation("a" * 40, attestation_sha256="0" * 64, modules=modules)
    with pytest.raises(ImplementationProvenanceError, match="is not imported"):
        provenance.verify_runtime_implementation("a" * 40, modules={})

    def broken(modules=None):
        raise OSError("simulated unreadable import location")

    monkeypatch.setattr(provenance, "imported_package_root", broken)
    with pytest.raises(ImplementationProvenanceError, match="unverifiable: simulated unreadable"):
        provenance.verify_runtime_implementation("a" * 40, modules=modules)


def test_environment_acknowledgement_alone_is_not_identity(plain_package, monkeypatch):
    monkeypatch.setenv("BTC_QH_IMPLEMENTATION_GIT_SHA", "a" * 40)
    with pytest.raises(ImplementationProvenanceError, match="unverifiable"):
        provenance.verify_runtime_implementation("a" * 40, modules=_load_modules(plain_package))


def test_executing_test_process_package_is_importable_and_inventoried():
    root = provenance.imported_package_root()
    files = provenance.snapshot_package_files(root)
    assert root.name == PACKAGE
    assert f"{PACKAGE}/research/prospective_provenance.py" in {path for path, _, _ in files}
    assert provenance.verify_loaded_modules(root, files) >= 2
