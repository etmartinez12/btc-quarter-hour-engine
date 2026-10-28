from __future__ import annotations

from weakref import WeakKeyDictionary


_verified_official_bundles: WeakKeyDictionary[object, tuple[object, ...]] = (
    WeakKeyDictionary()
)


def _bundle_fingerprint(bundle: object) -> tuple[object, ...]:
    models = getattr(bundle, "models", {})
    hashes = getattr(bundle, "model_sha256", {})
    manifest = getattr(bundle, "manifest", {})
    return (
        getattr(bundle, "manifest_sha256", None),
        tuple(sorted((name, id(model)) for name, model in models.items())),
        tuple(sorted(hashes.items())),
        manifest.get("status") if isinstance(manifest, dict) else None,
    )


def register_official_bundle(bundle: object) -> None:
    _verified_official_bundles[bundle] = _bundle_fingerprint(bundle)


def is_verified_official_bundle(bundle: object) -> bool:
    try:
        expected = _verified_official_bundles.get(bundle)
        return expected is not None and _bundle_fingerprint(bundle) == expected
    except (AttributeError, TypeError):
        return False
