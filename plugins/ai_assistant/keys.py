"""API keys in the OS keychain (macOS Keychain, Windows Credential Locker, Secret Service on Linux).

Keys are never written to settings.toml or the plugin's JSON storage. If the system has no
secure keyring backend, saving is refused rather than silently falling back to a plaintext
file, and the user can supply the key through an environment variable instead.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

SERVICE = "ThetaIDE"

ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
}

# Backends that would store the secret unencrypted, or not at all.
_INSECURE_BACKENDS = (
    "keyring.backends.fail",
    "keyring.backends.null",
    "keyrings.alt.file",
    "keyrings.alt.Windows",
)


class KeyStoreError(Exception):
    pass


class KeyStore:
    """Per-provider API keys: the keychain first, then the provider's environment variable."""

    def __init__(self, backend: Any = None, env: Mapping[str, str] | None = None):
        self._backend = backend
        self._env = os.environ if env is None else env

    @property
    def backend(self) -> Any:
        if self._backend is None:
            try:
                import keyring
            except ImportError:
                return None
            self._backend = keyring
        return self._backend

    @staticmethod
    def username(provider: str) -> str:
        return f"{provider}-api-key"

    def is_secure(self) -> bool:
        """Whether a real OS keychain is available to save into."""
        backend = self.backend
        if backend is None:
            return False
        get = getattr(backend, "get_keyring", None)
        active = get() if callable(get) else backend
        module = type(active).__module__
        return not module.startswith(_INSECURE_BACKENDS)

    def _keychain_value(self, provider: str) -> str | None:
        if self.backend is None:
            return None
        try:
            return self.backend.get_password(SERVICE, self.username(provider)) or None
        except Exception:  # a locked or unreachable keychain must not crash the IDE
            return None

    def get(self, provider: str) -> str | None:
        return self._keychain_value(provider) or self._env.get(ENV_VARS.get(provider, ""), "") or None

    def source(self, provider: str) -> str | None:
        """Where the key in use comes from: 'keychain', 'environment', or None."""
        if self._keychain_value(provider):
            return "keychain"
        if self._env.get(ENV_VARS.get(provider, "")):
            return "environment"
        return None

    def set(self, provider: str, key: str) -> None:
        key = key.strip()
        if not key:
            raise KeyStoreError("The API key is empty.")
        if not self.is_secure():
            raise KeyStoreError(
                "No secure OS keychain is available, so the key was not saved. "
                f"Set the {ENV_VARS.get(provider, 'provider')} environment variable instead."
            )
        try:
            self.backend.set_password(SERVICE, self.username(provider), key)
        except Exception as exc:
            raise KeyStoreError(f"The keychain refused the key: {exc}") from exc

    def delete(self, provider: str) -> None:
        if self.backend is None:
            return
        try:
            self.backend.delete_password(SERVICE, self.username(provider))
        except Exception:
            pass  # already absent


def mask(key: str | None) -> str:
    """A key safe to show in the UI: its last four characters only."""
    if not key:
        return ""
    return "•" * 8 + key[-4:] if len(key) > 8 else "•" * len(key)
