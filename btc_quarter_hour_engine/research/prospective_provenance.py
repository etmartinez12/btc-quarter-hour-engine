"""Fail-closed runtime implementation provenance for the prospective runner.

The prospective ledger stores ``implementation_git_sha`` as evidence of which
implementation produced each prediction. That value must describe the code
that is actually executing, not merely agree with operator-supplied
configuration. This module establishes the executing implementation's
identity locally (no network access) in one of two modes:

``git_checkout``
    The imported ``btc_quarter_hour_engine`` package is the top-level package
    directory of a Git working tree. ``HEAD`` must equal the expected commit,
    the working tree must be clean (no modified, staged, deleted, untracked or
    ignored package files, and no assume-unchanged/skip-worktree flags), and
    every on-disk package file must hash, in Python, to the blob recorded in
    ``HEAD``.

``installed_artifact``
    The package was installed from a wheel; Git metadata is not assumed. The
    operator supplies the implementation attestation produced at build time by
    :func:`build_implementation_attestation` from a verified checkout and pins
    its SHA256 in the reviewed configuration. Installed files, the owning
    distribution's ``RECORD`` and the attestation must agree exactly.

In both modes every already-imported package module must have been loaded by
the standard source loader from a file in the verified set. Every cache the
current interpreter could use for those sources is checked against a fresh
compilation of the verified source, not merely its timestamp or hash header.

This is self-verification by the running process. It detects an unexpected
commit, modified/extra/missing source, a mismatched installed package and an
unverifiable identity. It does not defend against an adversary able to modify
the interpreter, this verifier, or third-party dependencies; those remain host
access-control and environment-freeze responsibilities.
"""

from __future__ import annotations

import base64
import csv
import email.parser
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import marshal
import os
import re
import shutil
import stat
import subprocess
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

PACKAGE_NAME = "btc_quarter_hour_engine"
DISTRIBUTION_NAME = "btc-quarter-hour-engine"
ATTESTATION_SCHEMA = "btc-qh-implementation-attestation/v1"
MODE_GIT_CHECKOUT = "git_checkout"
MODE_INSTALLED_ARTIFACT = "installed_artifact"

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ATTESTATION_BYTES = 4 * 1024 * 1024
_MAX_WHEEL_BYTES = 256 * 1024 * 1024
_MAX_PACKAGE_FILES = 10_000
_GIT_TIMEOUT_SECONDS = 60.0
_ATTESTATION_KEYS = frozenset(
    {
        "schema",
        "distribution",
        "version",
        "package",
        "git_sha",
        "wheel_filename",
        "wheel_sha256",
        "files",
        "package_content_sha256",
    }
)

PackageFiles = tuple[tuple[str, str, int], ...]


class ImplementationProvenanceError(ValueError):
    """The executing implementation's identity could not be verified."""


@dataclass(frozen=True)
class VerifiedImplementation:
    """Independently verified identity of the executing implementation."""

    mode: str
    git_sha: str
    package_root: Path
    files: PackageFiles
    package_content_sha256: str
    source_root: Path | None = None
    attestation_sha256: str | None = None
    wheel_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "git_sha": self.git_sha,
            "package_root": str(self.package_root),
            "package_file_count": len(self.files),
            "package_content_sha256": self.package_content_sha256,
            "source_root": None if self.source_root is None else str(self.source_root),
            "attestation_sha256": self.attestation_sha256,
            "wheel_sha256": self.wheel_sha256,
        }


@dataclass(frozen=True)
class ImplementationAttestation:
    """Parsed, integrity-checked build attestation."""

    sha256: str
    version: str
    git_sha: str
    wheel_filename: str
    wheel_sha256: str
    files: PackageFiles
    package_content_sha256: str


def require_git_sha(value: object, label: str = "implementation Git SHA") -> str:
    if not isinstance(value, str) or not _GIT_SHA_RE.fullmatch(value):
        raise ImplementationProvenanceError(f"{label} must be 40 lowercase hexadecimal characters")
    return value


def require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ImplementationProvenanceError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def package_content_digest(files: Iterable[tuple[str, str, int]]) -> str:
    """Return the order-independent SHA256 of a package file inventory."""

    digest = hashlib.sha256()
    for path, sha256, size in sorted(files):
        digest.update(f"{path}\n{sha256}\n{size}\n".encode("utf-8"))
    return digest.hexdigest()


def _scan_package(package_root: Path) -> tuple[PackageFiles, dict[str, bytes]]:
    root = Path(package_root)
    if root.name != PACKAGE_NAME:
        raise ImplementationProvenanceError(f"package root {root} is not named {PACKAGE_NAME}")
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise ImplementationProvenanceError(f"package root {root} is unreadable: {exc}") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ImplementationProvenanceError(f"package root {root} is not a real directory")

    entries: list[tuple[str, str, int]] = []
    contents: dict[str, bytes] = {}

    def _raise(exc: OSError) -> None:
        raise ImplementationProvenanceError(f"package directory walk failed: {exc}") from exc

    try:
        for dirpath, dirnames, filenames in os.walk(root, onerror=_raise, followlinks=False):
            current = Path(dirpath)
            relative_parts = current.relative_to(root).parts
            in_cache = "__pycache__" in relative_parts
            if in_cache and dirnames:
                raise ImplementationProvenanceError(f"unexpected directory inside {current}")
            for name in dirnames:
                if not stat.S_ISDIR(os.lstat(current / name).st_mode):
                    raise ImplementationProvenanceError(f"package directory {current / name} is a link")
            dirnames.sort()
            for name in sorted(filenames):
                path = current / name
                file_stat = os.lstat(path)
                if not stat.S_ISREG(file_stat.st_mode):
                    raise ImplementationProvenanceError(f"package entry {path} is not a regular file")
                if in_cache:
                    if not name.endswith(".pyc"):
                        raise ImplementationProvenanceError(f"unexpected bytecode-cache entry {path}")
                    continue
                relative = PurePosixPath(PACKAGE_NAME, *relative_parts, name).as_posix()
                with open(path, "rb") as handle:
                    data = handle.read()
                entries.append((relative, hashlib.sha256(data).hexdigest(), len(data)))
                contents[relative] = data
                if len(entries) > _MAX_PACKAGE_FILES:
                    raise ImplementationProvenanceError("package file inventory exceeds the safety limit")
    except OSError as exc:
        raise ImplementationProvenanceError(f"package file inventory failed: {exc}") from exc
    if f"{PACKAGE_NAME}/__init__.py" not in contents:
        raise ImplementationProvenanceError("package root has no __init__.py")
    return tuple(sorted(entries)), contents


def snapshot_package_files(package_root: Path) -> PackageFiles:
    """Inventory every non-cache package file as ``(path, sha256, size)``."""

    return _scan_package(package_root)[0]


def _same_location(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def imported_package_root(modules: Mapping[str, object] | None = None) -> Path:
    """Return the directory the executing package was imported from."""

    module_map = sys.modules if modules is None else modules
    package = module_map.get(PACKAGE_NAME)
    if package is None and modules is None:
        package = importlib.import_module(PACKAGE_NAME)
    if package is None:
        raise ImplementationProvenanceError(f"{PACKAGE_NAME} is not imported")
    spec = getattr(package, "__spec__", None)
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    if spec is None or len(locations) != 1:
        raise ImplementationProvenanceError(f"{PACKAGE_NAME} must be imported from exactly one directory")
    root = Path(os.path.abspath(locations[0]))
    origin = getattr(spec, "origin", None)
    if root.name != PACKAGE_NAME or not origin or not _same_location(Path(origin), root / "__init__.py"):
        raise ImplementationProvenanceError(f"{PACKAGE_NAME} import location is inconsistent")
    return root


def _in_package(name: str) -> bool:
    return name == PACKAGE_NAME or name.startswith(PACKAGE_NAME + ".")


def _matches_compiled_code(cached: types.CodeType, expected: types.CodeType) -> bool:
    # Code equality omits stack size and some interpreter metadata. Check
    # those fields at every nested code object too, not only at module level.
    if cached != expected or any(
        getattr(cached, field, None) != getattr(expected, field, None)
        for field in ("co_stacksize", "co_filename", "co_qualname", "co_linetable", "co_exceptiontable")
    ):
        return False
    return all(
        not isinstance(value, types.CodeType)
        or (isinstance(reference, types.CodeType) and _matches_compiled_code(value, reference))
        for value, reference in zip(cached.co_consts, expected.co_consts)
    )


def _verify_bytecode_cache(source: Path, source_bytes: bytes) -> None:
    cached = importlib.util.cache_from_source(str(source))
    try:
        with open(cached, "rb") as handle:
            header = handle.read(16)
            if len(header) != 16 or header[:4] != importlib.util.MAGIC_NUMBER:
                raise ImplementationProvenanceError(f"invalid bytecode cache header for {source}")
            flags = int.from_bytes(header[4:8], "little")
            if flags == 1:
                raise ImplementationProvenanceError(
                    f"{source} has an unchecked hash-based bytecode cache that is not validated against source"
                )
            if flags not in (0, 3):
                raise ImplementationProvenanceError(f"unsupported bytecode cache flags for {source}")
            if flags == 3 and header[8:16] != importlib.util.source_hash(source_bytes):
                raise ImplementationProvenanceError(f"bytecode cache source hash differs for {source}")
            code = marshal.load(handle)
            if not isinstance(code, types.CodeType) or handle.read(1):
                raise ImplementationProvenanceError(f"invalid bytecode cache payload for {source}")
    except FileNotFoundError:
        return
    except ImplementationProvenanceError:
        raise
    except (OSError, EOFError, ValueError, TypeError) as exc:
        raise ImplementationProvenanceError(f"bytecode cache for {source} is unverifiable: {exc}") from exc
    try:
        # Installers may compile in a staging directory. SourceFileLoader fixes
        # co_filename on import; it does not verify the cached code's contents.
        expected = compile(
            source_bytes, code.co_filename, "exec", dont_inherit=True,
            optimize=sys.flags.optimize,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ImplementationProvenanceError(f"bytecode source cannot be compiled for {source}: {exc}") from exc
    if not _matches_compiled_code(code, expected):
        raise ImplementationProvenanceError(f"bytecode cache payload differs from verified source for {source}")


def _verify_module(name: str, module: object, package_root: Path, allowed: frozenset[str]) -> None:
    spec = getattr(module, "__spec__", None)
    if spec is None or spec.name != name:
        raise ImplementationProvenanceError(f"loaded module {name} has no consistent import spec")
    if type(spec.loader) is not importlib.machinery.SourceFileLoader:
        raise ImplementationProvenanceError(f"loaded module {name} was not imported from a source file")
    origin = spec.origin
    module_file = getattr(module, "__file__", None)
    if not origin or not module_file:
        raise ImplementationProvenanceError(f"loaded module {name} has no source location")
    origin_path = Path(os.path.abspath(origin))
    if os.path.normcase(os.path.abspath(module_file)) != os.path.normcase(str(origin_path)):
        raise ImplementationProvenanceError(f"loaded module {name} __file__ does not match its origin")
    try:
        relative = origin_path.relative_to(package_root.parent).as_posix()
    except ValueError as exc:
        raise ImplementationProvenanceError(
            f"loaded module {name} originates outside the verified package: {origin_path}"
        ) from exc
    is_package = spec.submodule_search_locations is not None
    parts = name.split(".")
    expected = "/".join(parts) + ("/__init__.py" if is_package else ".py")
    if relative != expected or relative not in allowed:
        raise ImplementationProvenanceError(f"loaded module {name} originates from unverified file {relative}")
    if is_package:
        search = [os.path.normcase(os.path.abspath(item)) for item in (getattr(module, "__path__", None) or [])]
        if search != [os.path.normcase(str(origin_path.parent))]:
            raise ImplementationProvenanceError(f"loaded package {name} has an altered search path")
    expected_cache = os.path.normcase(importlib.util.cache_from_source(str(origin_path)))
    if any(
        not cached or os.path.normcase(os.path.abspath(cached)) != expected_cache
        for cached in (getattr(spec, "cached", None), getattr(module, "__cached__", None))
    ):
        raise ImplementationProvenanceError(f"loaded module {name} has an inconsistent bytecode cache location")


def verify_loaded_modules(
    package_root: Path, files: PackageFiles, *, modules: Mapping[str, object] | None = None
) -> int:
    """Verify loaded origins and all active caches, including not-yet-imported modules."""

    module_map = sys.modules if modules is None else modules
    allowed = frozenset(path for path, _, _ in files if path.endswith(".py"))
    root = Path(os.path.abspath(package_root))
    for path, sha256, size in files:
        if path not in allowed:
            continue
        source = root.parent.joinpath(*PurePosixPath(path).parts)
        try:
            source_bytes = source.read_bytes()
        except OSError as exc:
            raise ImplementationProvenanceError(f"verified source {source} is unreadable: {exc}") from exc
        if len(source_bytes) != size or hashlib.sha256(source_bytes).hexdigest() != sha256:
            raise ImplementationProvenanceError(f"source changed after provenance inventory: {source}")
        _verify_bytecode_cache(source, source_bytes)
    checked = 0
    root_seen = False
    for key, module in list(module_map.items()):
        if module is None:
            continue
        name = key
        if key == "__main__":
            spec = getattr(module, "__spec__", None)
            if spec is None or not _in_package(spec.name):
                continue
            name = spec.name
        elif not _in_package(key):
            continue
        _verify_module(name, module, root, allowed)
        checked += 1
        root_seen = root_seen or name == PACKAGE_NAME
    if not root_seen:
        raise ImplementationProvenanceError(f"{PACKAGE_NAME} is not among the loaded modules")
    return checked


# --------------------------------------------------------------------------
# Git checkout identity
# --------------------------------------------------------------------------


def _git_environment() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("GIT_")}
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _run_git(git: str, root: Path, *arguments: str) -> bytes:
    command = [
        git,
        "--no-replace-objects",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-C",
        str(root),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_git_environment(),
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ImplementationProvenanceError(f"git {arguments[0]} could not run: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()[:500]
        raise ImplementationProvenanceError(
            f"git {arguments[0]} failed with exit status {completed.returncode}: {detail}"
        )
    return completed.stdout


def git_blob_sha1(data: bytes) -> str:
    """Return Git's SHA-1 object ID for a blob containing ``data``."""

    digest = hashlib.sha1(b"blob %d\0" % len(data), usedforsecurity=False)
    digest.update(data)
    return digest.hexdigest()


def _matches_blob(data: bytes, object_id: str) -> bool:
    if git_blob_sha1(data) == object_id:
        return True
    # A checkout with core.autocrlf converts LF blobs to CRLF on disk. Accept
    # only that exact normalisation; Git clean/smudge filters are not trusted.
    return b"\r\n" in data and git_blob_sha1(data.replace(b"\r\n", b"\n")) == object_id


def _tracked_package_blobs(git: str, checkout: Path) -> dict[str, str]:
    output = _run_git(git, checkout, "ls-tree", "-r", "-z", "--full-tree", "HEAD", "--", PACKAGE_NAME)
    tracked: dict[str, str] = {}
    for record in output.split(b"\0"):
        if not record:
            continue
        meta, separator, raw_path = record.partition(b"\t")
        fields = meta.split(b" ")
        if not separator or len(fields) != 3:
            raise ImplementationProvenanceError("unexpected git ls-tree output")
        mode, object_type, object_id = (field.decode("ascii", "replace") for field in fields)
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ImplementationProvenanceError("tracked package path is not valid UTF-8") from exc
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ImplementationProvenanceError(f"tracked package entry {path} is not a regular file blob")
        if not _GIT_SHA_RE.fullmatch(object_id):
            raise ImplementationProvenanceError("repository object format is not SHA-1")
        if not path.startswith(PACKAGE_NAME + "/") or path in tracked:
            raise ImplementationProvenanceError(f"unexpected tracked package path {path}")
        tracked[path] = object_id
    if f"{PACKAGE_NAME}/__init__.py" not in tracked:
        raise ImplementationProvenanceError("HEAD does not contain the package")
    return tracked


def _describe_paths(paths: Iterable[str]) -> str:
    ordered = sorted(paths)
    shown = ", ".join(ordered[:5])
    return shown + (f" (+{len(ordered) - 5} more)" if len(ordered) > 5 else "")


def verify_git_checkout(
    package_root: Path,
    expected_git_sha: str,
    *,
    modules: Mapping[str, object] | None = None,
    git_executable: str | None = None,
) -> VerifiedImplementation:
    """Verify the executing package is an unmodified checkout of a commit."""

    expected = require_git_sha(expected_git_sha)
    root = Path(os.path.abspath(package_root))
    checkout = root.parent
    if not os.path.lexists(checkout / ".git"):
        raise ImplementationProvenanceError(
            f"executing package {root} is not the top-level package of a Git checkout"
        )
    git = git_executable or shutil.which("git")
    if not git:
        raise ImplementationProvenanceError("git executable is unavailable; checkout identity is unverifiable")

    toplevel = _run_git(git, checkout, "rev-parse", "--show-toplevel").decode("utf-8", "replace").strip()
    if not toplevel or not _same_location(Path(toplevel), checkout):
        raise ImplementationProvenanceError(f"package checkout {checkout} is not the Git work-tree root")
    head = _run_git(git, checkout, "rev-parse", "--verify", "HEAD^{commit}").decode("ascii", "replace").strip()
    if head != expected:
        raise ImplementationProvenanceError(
            f"executing checkout HEAD {head} does not match expected implementation {expected}"
        )

    status = _run_git(
        git,
        checkout,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        changed = [entry.decode("utf-8", "replace") for entry in status.split(b"\0") if entry]
        raise ImplementationProvenanceError(f"Git working tree is not clean: {_describe_paths(changed)}")
    flagged = []
    for entry in _run_git(git, checkout, "ls-files", "-v", "-z").split(b"\0"):
        if entry and (entry[:1].islower() or entry[:1] == b"S"):
            flagged.append(entry[2:].decode("utf-8", "replace"))
    if flagged:
        raise ImplementationProvenanceError(
            f"index hides working-tree changes (assume-unchanged/skip-worktree): {_describe_paths(flagged)}"
        )

    tracked = _tracked_package_blobs(git, checkout)
    files, contents = _scan_package(root)
    extra = set(contents) - set(tracked)
    missing = set(tracked) - set(contents)
    if extra:
        raise ImplementationProvenanceError(f"package contains files not in {expected}: {_describe_paths(extra)}")
    if missing:
        raise ImplementationProvenanceError(f"package is missing files from {expected}: {_describe_paths(missing)}")
    modified = [path for path, object_id in tracked.items() if not _matches_blob(contents[path], object_id)]
    if modified:
        raise ImplementationProvenanceError(f"package files differ from {expected}: {_describe_paths(modified)}")

    verify_loaded_modules(root, files, modules=modules)
    return VerifiedImplementation(
        mode=MODE_GIT_CHECKOUT,
        git_sha=head,
        package_root=root,
        files=files,
        package_content_sha256=package_content_digest(files),
        source_root=checkout,
    )


# --------------------------------------------------------------------------
# Installed-artifact identity
# --------------------------------------------------------------------------


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise ImplementationProvenanceError(f"attestation contains duplicate key {key!r}")
        document[key] = value
    return document


def _reject_constant(value: str) -> None:
    raise ImplementationProvenanceError(f"attestation contains non-finite number {value}")


def _normalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _valid_package_path(path: object) -> bool:
    if not isinstance(path, str) or not path.startswith(PACKAGE_NAME + "/") or "\\" in path:
        return False
    parts = path.split("/")
    return all(part not in {"", ".", "..", "__pycache__"} for part in parts) and not path.endswith(".pyc")


def parse_implementation_attestation(payload: bytes, expected_sha256: str) -> ImplementationAttestation:
    """Verify an attestation's pinned SHA256, then parse it strictly."""

    pinned = require_sha256(expected_sha256, "implementation attestation SHA256")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != pinned:
        raise ImplementationProvenanceError(
            f"implementation attestation SHA256 {actual} does not match pinned {pinned}"
        )
    try:
        document = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ImplementationProvenanceError(f"implementation attestation is not valid JSON: {exc}") from exc
    if not isinstance(document, dict) or set(document) != _ATTESTATION_KEYS:
        raise ImplementationProvenanceError("implementation attestation has an unexpected structure")
    if document["schema"] != ATTESTATION_SCHEMA:
        raise ImplementationProvenanceError("implementation attestation schema is not supported")
    if document["distribution"] != DISTRIBUTION_NAME or document["package"] != PACKAGE_NAME:
        raise ImplementationProvenanceError("implementation attestation names a different package")
    version = document["version"]
    wheel_filename = document["wheel_filename"]
    if not isinstance(version, str) or not version or "/" in version or "\\" in version:
        raise ImplementationProvenanceError("implementation attestation version is invalid")
    if (
        not isinstance(wheel_filename, str)
        or not wheel_filename.startswith(f"{PACKAGE_NAME}-{version}-")
        or not wheel_filename.endswith(".whl")
        or "/" in wheel_filename
        or "\\" in wheel_filename
    ):
        raise ImplementationProvenanceError("implementation attestation wheel filename is invalid")
    git_sha = require_git_sha(document["git_sha"], "attested implementation Git SHA")
    wheel_sha256 = require_sha256(document["wheel_sha256"], "attested wheel SHA256")
    raw_files = document["files"]
    if not isinstance(raw_files, dict) or not raw_files or len(raw_files) > _MAX_PACKAGE_FILES:
        raise ImplementationProvenanceError("implementation attestation file inventory is invalid")
    files: list[tuple[str, str, int]] = []
    for path, entry in raw_files.items():
        if not _valid_package_path(path) or not isinstance(entry, dict) or set(entry) != {"sha256", "size"}:
            raise ImplementationProvenanceError(f"implementation attestation entry {path!r} is invalid")
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ImplementationProvenanceError(f"implementation attestation size for {path} is invalid")
        files.append((path, require_sha256(entry["sha256"], f"attested SHA256 for {path}"), size))
    inventory = tuple(sorted(files))
    if f"{PACKAGE_NAME}/__init__.py" not in raw_files:
        raise ImplementationProvenanceError("implementation attestation does not include the package root")
    content = require_sha256(document["package_content_sha256"], "attested package content SHA256")
    if package_content_digest(inventory) != content:
        raise ImplementationProvenanceError("implementation attestation package content digest is inconsistent")
    return ImplementationAttestation(
        sha256=actual,
        version=version,
        git_sha=git_sha,
        wheel_filename=wheel_filename,
        wheel_sha256=wheel_sha256,
        files=inventory,
        package_content_sha256=content,
    )


def load_implementation_attestation(path: Path, expected_sha256: str) -> ImplementationAttestation:
    """Read a local attestation file and verify it against its pinned SHA256."""

    location = Path(path)
    if not location.is_absolute():
        raise ImplementationProvenanceError("implementation attestation path must be absolute")
    try:
        file_stat = os.lstat(location)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ImplementationProvenanceError("implementation attestation must be a regular, non-link file")
        if file_stat.st_size > _MAX_ATTESTATION_BYTES:
            raise ImplementationProvenanceError("implementation attestation exceeds the size limit")
        with open(location, "rb") as handle:
            payload = handle.read(_MAX_ATTESTATION_BYTES + 1)
    except FileNotFoundError as exc:
        raise ImplementationProvenanceError(f"implementation attestation {location} is missing") from exc
    except OSError as exc:
        raise ImplementationProvenanceError(f"implementation attestation is unreadable: {exc}") from exc
    if len(payload) > _MAX_ATTESTATION_BYTES:
        raise ImplementationProvenanceError("implementation attestation exceeds the size limit")
    return parse_implementation_attestation(payload, expected_sha256)


def _record_digest(sha256_hex: str) -> str:
    return "sha256=" + base64.urlsafe_b64encode(bytes.fromhex(sha256_hex)).rstrip(b"=").decode("ascii")


def _parse_record(text: str) -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 3:
            raise ImplementationProvenanceError("distribution RECORD has a malformed row")
        path = row[0].replace("\\", "/")
        if path in entries:
            raise ImplementationProvenanceError(f"distribution RECORD lists {path} more than once")
        entries[path] = (row[1], row[2])
    return entries


def _owning_distribution(package_root: Path, search_path: list[str] | None) -> importlib.metadata.Distribution:
    init_file = package_root / "__init__.py"
    owners: dict[tuple[str, str], importlib.metadata.Distribution] = {}
    paths = list(sys.path if search_path is None else search_path)
    for distribution in importlib.metadata.distributions(path=paths):
        name = distribution.metadata["Name"] or ""
        if _normalize_distribution(name) != DISTRIBUTION_NAME:
            continue
        located = Path(str(distribution.locate_file(f"{PACKAGE_NAME}/__init__.py")))
        if _same_location(located, init_file):
            base = os.path.normcase(os.path.realpath(str(distribution.locate_file(""))))
            owners[(base, distribution.version)] = distribution
    if len(owners) != 1:
        raise ImplementationProvenanceError(
            f"expected exactly one installed {DISTRIBUTION_NAME} distribution owning {package_root}, "
            f"found {len(owners)}"
        )
    return next(iter(owners.values()))


def _verify_direct_url(distribution: importlib.metadata.Distribution, wheel_sha256: str) -> None:
    text = distribution.read_text("direct_url.json")
    if text is None:
        return
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ImplementationProvenanceError("installed direct_url.json is invalid") from exc
    if not isinstance(document, dict):
        raise ImplementationProvenanceError("installed direct_url.json is invalid")
    dir_info = document.get("dir_info")
    if isinstance(dir_info, dict) and dir_info.get("editable"):
        raise ImplementationProvenanceError("editable installs must be verified in Git checkout mode")
    archive_info = document.get("archive_info")
    if isinstance(archive_info, dict):
        hashes = archive_info.get("hashes")
        recorded = hashes.get("sha256") if isinstance(hashes, dict) else None
        legacy = archive_info.get("hash")
        if recorded is None and isinstance(legacy, str) and legacy.startswith("sha256="):
            recorded = legacy[len("sha256=") :]
        if recorded is not None and recorded != wheel_sha256:
            raise ImplementationProvenanceError("installed archive hash does not match the attested wheel")


def verify_installed_artifact(
    package_root: Path,
    expected_git_sha: str,
    attestation_path: Path,
    attestation_sha256: str,
    *,
    modules: Mapping[str, object] | None = None,
    search_path: list[str] | None = None,
) -> VerifiedImplementation:
    """Verify an installed package against a pinned build attestation."""

    expected = require_git_sha(expected_git_sha)
    attestation = load_implementation_attestation(attestation_path, attestation_sha256)
    if attestation.git_sha != expected:
        raise ImplementationProvenanceError(
            f"attested implementation {attestation.git_sha} does not match expected {expected}"
        )
    root = Path(os.path.abspath(package_root))
    files = snapshot_package_files(root)
    installed = {path: (sha256, size) for path, sha256, size in files}
    attested = {path: (sha256, size) for path, sha256, size in attestation.files}
    if set(installed) - set(attested):
        raise ImplementationProvenanceError(
            f"installed package has unattested files: {_describe_paths(set(installed) - set(attested))}"
        )
    if set(attested) - set(installed):
        raise ImplementationProvenanceError(
            f"installed package is missing attested files: {_describe_paths(set(attested) - set(installed))}"
        )
    changed = [path for path in attested if installed[path] != attested[path]]
    if changed:
        raise ImplementationProvenanceError(f"installed package files differ from attestation: {_describe_paths(changed)}")

    distribution = _owning_distribution(root, search_path)
    if distribution.version != attestation.version:
        raise ImplementationProvenanceError("installed distribution version does not match the attestation")
    record_text = distribution.read_text("RECORD")
    if record_text is None:
        raise ImplementationProvenanceError("installed distribution has no RECORD")
    record = {
        path: value
        for path, value in _parse_record(record_text).items()
        if path.startswith(PACKAGE_NAME + "/") and "/__pycache__/" not in path
    }
    if set(record) != set(attested):
        raise ImplementationProvenanceError("installed distribution RECORD does not match the attested files")
    for path, (digest, size) in record.items():
        sha256, expected_size = attested[path]
        if digest != _record_digest(sha256) or size != str(expected_size):
            raise ImplementationProvenanceError(f"installed distribution RECORD disagrees for {path}")
    _verify_direct_url(distribution, attestation.wheel_sha256)

    verify_loaded_modules(root, files, modules=modules)
    return VerifiedImplementation(
        mode=MODE_INSTALLED_ARTIFACT,
        git_sha=attestation.git_sha,
        package_root=root,
        files=files,
        package_content_sha256=package_content_digest(files),
        attestation_sha256=attestation.sha256,
        wheel_sha256=attestation.wheel_sha256,
    )


# --------------------------------------------------------------------------
# Runtime entry points
# --------------------------------------------------------------------------


def verify_runtime_implementation(
    expected_git_sha: str,
    *,
    attestation_path: Path | None = None,
    attestation_sha256: str | None = None,
    modules: Mapping[str, object] | None = None,
) -> VerifiedImplementation:
    """Independently establish which implementation is executing, or fail."""

    try:
        expected = require_git_sha(expected_git_sha)
        if (attestation_path is None) != (attestation_sha256 is None):
            raise ImplementationProvenanceError(
                "implementation attestation path and SHA256 must be configured together"
            )
        root = imported_package_root(modules)
        if attestation_path is not None:
            return verify_installed_artifact(
                root, expected, Path(attestation_path), str(attestation_sha256), modules=modules
            )
        if not os.path.lexists(root.parent / ".git"):
            raise ImplementationProvenanceError(
                "executing package is not in a Git checkout and no implementation attestation is "
                "configured; implementation identity is unverifiable"
            )
        return verify_git_checkout(root, expected, modules=modules)
    except ImplementationProvenanceError:
        raise
    except (OSError, ValueError, LookupError, RuntimeError) as exc:
        raise ImplementationProvenanceError(f"implementation identity is unverifiable: {exc}") from exc


def recheck_runtime_implementation(
    identity: VerifiedImplementation, *, modules: Mapping[str, object] | None = None
) -> VerifiedImplementation:
    """Cheaply confirm the verified package has not drifted since startup."""

    if not isinstance(identity, VerifiedImplementation):
        raise ImplementationProvenanceError("no verified implementation identity is available")
    try:
        root = imported_package_root(modules)
        if not _same_location(root, identity.package_root):
            raise ImplementationProvenanceError("executing package location changed after verification")
        files = snapshot_package_files(identity.package_root)
        if files != identity.files:
            before = {path: (sha256, size) for path, sha256, size in identity.files}
            after = {path: (sha256, size) for path, sha256, size in files}
            drift = {path for path in set(before) | set(after) if before.get(path) != after.get(path)}
            raise ImplementationProvenanceError(
                f"package files changed after verification: {_describe_paths(drift)}"
            )
        verify_loaded_modules(identity.package_root, files, modules=modules)
    except ImplementationProvenanceError:
        raise
    except (OSError, ValueError, LookupError, RuntimeError) as exc:
        raise ImplementationProvenanceError(f"implementation identity recheck failed: {exc}") from exc
    return identity


# --------------------------------------------------------------------------
# Build-time attestation
# --------------------------------------------------------------------------


def _wheel_inventory(payload: bytes, wheel_name: str) -> tuple[str, PackageFiles]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ImplementationProvenanceError(f"wheel is not a valid archive: {exc}") from exc
    with archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ImplementationProvenanceError("wheel contains duplicate entries")
        dist_info_dirs = {name.split("/", 1)[0] for name in names if name.split("/", 1)[0].endswith(".dist-info")}
        if len(dist_info_dirs) != 1:
            raise ImplementationProvenanceError("wheel must contain exactly one .dist-info directory")
        dist_info = dist_info_dirs.pop()
        for name in names:
            top = name.split("/", 1)[0]
            if top not in {PACKAGE_NAME, dist_info} or "\\" in name or ".." in name.split("/"):
                raise ImplementationProvenanceError(f"wheel contains unexpected entry {name}")
        contents = {name: archive.read(name) for name in names}

    metadata_name = f"{dist_info}/METADATA"
    if metadata_name not in contents:
        raise ImplementationProvenanceError("wheel has no METADATA")
    try:
        metadata = email.parser.HeaderParser().parsestr(contents[metadata_name].decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ImplementationProvenanceError("wheel METADATA is not UTF-8") from exc
    if _normalize_distribution(metadata.get("Name", "")) != DISTRIBUTION_NAME:
        raise ImplementationProvenanceError("wheel METADATA names a different distribution")
    version = metadata.get("Version", "")
    if not version or not wheel_name.startswith(f"{PACKAGE_NAME}-{version}-") or not wheel_name.endswith(".whl"):
        raise ImplementationProvenanceError("wheel filename does not match its METADATA")

    record_name = f"{dist_info}/RECORD"
    if record_name not in contents:
        raise ImplementationProvenanceError("wheel has no RECORD")
    record = _parse_record(contents[record_name].decode("utf-8"))
    if set(record) != set(contents):
        raise ImplementationProvenanceError("wheel RECORD does not list exactly the archive entries")
    for name, data in contents.items():
        if name == record_name:
            continue
        digest, size = record[name]
        if digest != _record_digest(hashlib.sha256(data).hexdigest()) or size != str(len(data)):
            raise ImplementationProvenanceError(f"wheel RECORD disagrees for {name}")

    package_files = []
    for name, data in contents.items():
        if not name.startswith(PACKAGE_NAME + "/"):
            continue
        if not _valid_package_path(name):
            raise ImplementationProvenanceError(f"wheel contains non-source package entry {name}")
        package_files.append((name, hashlib.sha256(data).hexdigest(), len(data)))
    return version, tuple(sorted(package_files))


def build_implementation_attestation(
    wheel_path: Path, output_path: Path, *, source: VerifiedImplementation
) -> dict[str, Any]:
    """Attest that a wheel contains exactly a verified checkout's package.

    ``source`` must come from :func:`verify_git_checkout`. The attestation is
    written with exclusive creation; its returned SHA256 is the value to pin
    as ``implementation_attestation_sha256`` in the reviewed configuration.
    """

    if not isinstance(source, VerifiedImplementation) or source.mode != MODE_GIT_CHECKOUT:
        raise ImplementationProvenanceError("attestations may only be built from a verified Git checkout")
    wheel = Path(os.path.abspath(wheel_path))
    output = Path(os.path.abspath(output_path))
    try:
        if not stat.S_ISREG(os.lstat(wheel).st_mode):
            raise ImplementationProvenanceError("wheel must be a regular, non-link file")
        with open(wheel, "rb") as handle:
            payload = handle.read(_MAX_WHEEL_BYTES + 1)
    except OSError as exc:
        raise ImplementationProvenanceError(f"wheel is unreadable: {exc}") from exc
    if len(payload) > _MAX_WHEEL_BYTES:
        raise ImplementationProvenanceError("wheel exceeds the size limit")
    try:
        version, files = _wheel_inventory(payload, wheel.name)
    except ImplementationProvenanceError:
        raise
    except (ValueError, LookupError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ImplementationProvenanceError(f"wheel could not be inspected: {exc}") from exc
    if files != source.files:
        wheel_map = {path: (sha256, size) for path, sha256, size in files}
        source_map = {path: (sha256, size) for path, sha256, size in source.files}
        drift = {path for path in set(wheel_map) | set(source_map) if wheel_map.get(path) != source_map.get(path)}
        raise ImplementationProvenanceError(
            f"wheel package contents differ from verified checkout {source.git_sha}: {_describe_paths(drift)}"
        )

    document = {
        "schema": ATTESTATION_SCHEMA,
        "distribution": DISTRIBUTION_NAME,
        "version": version,
        "package": PACKAGE_NAME,
        "git_sha": source.git_sha,
        "wheel_filename": wheel.name,
        "wheel_sha256": hashlib.sha256(payload).hexdigest(),
        "files": {path: {"sha256": sha256, "size": size} for path, sha256, size in files},
        "package_content_sha256": package_content_digest(files),
    }
    encoded = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    try:
        descriptor = os.open(output, flags, 0o444)
        try:
            view = memoryview(encoded)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError as exc:
        raise ImplementationProvenanceError(f"attestation output {output} already exists") from exc
    except OSError as exc:
        raise ImplementationProvenanceError(f"attestation could not be written: {exc}") from exc
    attestation_sha256 = hashlib.sha256(encoded).hexdigest()
    parse_implementation_attestation(encoded, attestation_sha256)
    return {
        "attestation_path": str(output),
        "attestation_sha256": attestation_sha256,
        "git_sha": source.git_sha,
        "version": version,
        "wheel_filename": wheel.name,
        "wheel_sha256": document["wheel_sha256"],
        "package_file_count": len(files),
        "package_content_sha256": document["package_content_sha256"],
    }
