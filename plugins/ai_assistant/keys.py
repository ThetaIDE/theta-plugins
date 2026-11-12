"""API keys in the OS keychain (macOS Keychain, Windows Credential Locker, Secret Service on Linux).

Keys are never written to settings.toml, the plugin's JSON storage, or any other file. On Linux
desktops that do not start a keychain themselves (i3, sway, a bare X session), the Secret Service
is usually installed but asleep; it is started over D-Bus on first use. If no secure keychain is
available at all, a key can be kept in memory for this session only; it is never written to disk.

Opening a keychain can prompt the user (KWallet and GNOME Keyring ask to unlock or create a wallet), so
the keychain is read only when a key is needed: sending a message or listing models. To know which
providers have a key without reading it, the store keeps an index of provider names (never keys) in the
plugin's settings.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

SERVICE = "ThetaIDE"

ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "ollama": "OLLAMA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Backends that would store the secret unencrypted, or not at all.
_INSECURE_BACKENDS = (
    "keyring.backends.fail",
    "keyring.backends.null",
    "keyrings.alt.file",
    "keyrings.alt.Windows",
)


# D-Bus names that start a Secret Service provider when nothing is serving org.freedesktop.secrets.
SECRET_SERVICES = ("org.freedesktop.secrets", "org.kde.kwalletd6", "org.kde.kwalletd5", "org.gnome.keyring")


class KeyStoreError(Exception):
    pass


def wake_secret_service(timeout: float = 3.0) -> bool:
    """Start the desktop's Secret Service (KWallet, GNOME Keyring) over D-Bus. True once one is serving."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        from jeepney import DBusAddress, new_method_call
        from jeepney.io.blocking import open_dbus_connection
        from jeepney.wrappers import unwrap_msg
    except ImportError:
        return False
    bus = DBusAddress("/org/freedesktop/DBus", bus_name="org.freedesktop.DBus", interface="org.freedesktop.DBus")
    try:
        connection = open_dbus_connection(bus="SESSION")
    except Exception:  # no session bus (SSH, containers)
        return False

    def serving() -> bool:
        reply = connection.send_and_get_reply(new_method_call(bus, "NameHasOwner", "s", (SECRET_SERVICES[0],)))
        return bool(unwrap_msg(reply)[0])

    with connection:
        try:
            if serving():
                return True
            for name in SECRET_SERVICES:
                try:
                    unwrap_msg(
                        connection.send_and_get_reply(
                            new_method_call(bus, "StartServiceByName", "su", (name, 0)), timeout=timeout
                        )
                    )
                except Exception:  # not installed, or not activatable
                    continue
                deadline = time.monotonic() + timeout  # KWallet registers the Secret Service name just after starting
                while time.monotonic() < deadline:
                    if serving():
                        return True
                    time.sleep(0.1)
        except Exception:
            return False
    return False


class KeyStore:
    """Per-provider API keys: one entered for this session, then the keychain, then the environment variable."""

    def __init__(
        self,
        backend: Any = None,
        env: Mapping[str, str] | None = None,
        waker: Callable[[], bool] | None = None,
    ):
        self._backend = backend
        self._env = os.environ if env is None else env
        self._session: dict[str, str] = {}  # memory only, gone when ThetaIDE closes
        # Only the real keyring is woken by default: a test's fake backend never reaches D-Bus, and neither
        # does a backend the user pinned with PYTHON_KEYRING_BACKEND.
        wakes = backend is None and not (env if env is not None else os.environ).get("PYTHON_KEYRING_BACKEND")
        self._waker = waker if waker is not None else (wake_secret_service if wakes else lambda: False)
        self._woken = False
        self._cache: dict[str, str] = {}  # keychain values already read this session
        self._load_index: Callable[[], Any] | None = None
        self._save_index: Callable[[list[str]], None] | None = None

    def attach_index(self, load: Callable[[], Any], save: Callable[[list[str]], None]) -> None:
        """Where to keep the names of providers whose key is in the keychain (not the keys)."""
        self._load_index, self._save_index = load, save

    def _index(self) -> set[str] | None:
        """Providers known to have a keychain key, or None when unknown (no index yet)."""
        if self._load_index is None:
            return None
        value = self._load_index()
        return None if value is None else {str(p) for p in value}

    def _record(self, provider: str, present: bool) -> None:
        if self._save_index is None:
            return
        index = self._index()
        if index is None and not present:
            return  # an unknown index stays unknown until a key is found or saved
        index = set(index or ())
        if (provider in index) != present:
            index.symmetric_difference_update({provider})
            self._save_index(sorted(index))
        elif self._index() is None:
            self._save_index(sorted(index))

    @property
    def backend(self) -> Any:
        if self._backend is None:
            try:
                import keyring
            except ImportError:
                return None
            self._backend = keyring
            if not self.is_secure():
                self.wake()  # a key saved last time is unreadable until the keychain runs
        return self._backend

    def wake(self) -> bool:
        """Try once to start the desktop keychain and re-detect keyring's backend. True if now secure."""
        if self._woken:
            return self.is_secure()
        self._woken = True
        if not self._waker():
            return False
        if self._backend is not None and hasattr(self._backend, "core"):
            try:
                self._backend.core.init_backend()
            except Exception:
                return False
        return self.is_secure()

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
        """Read the keychain (this may show its unlock prompt). Cached for the session."""
        if provider in self._cache:
            return self._cache[provider]
        if self.backend is None:
            return None
        try:
            value = self.backend.get_password(SERVICE, self.username(provider)) or None
        except Exception:  # a locked or unreachable keychain must not crash the IDE
            return None
        if value:
            self._cache[provider] = value
        self._record(provider, bool(value))
        return value

    def _in_keychain(self, provider: str) -> bool:
        """Whether a keychain key exists, without opening the keychain when the index can answer."""
        if provider in self._cache:
            return True
        index = self._index()
        if index is not None:
            return provider in index
        if self._load_index is not None:
            return False  # unknown: assume none rather than prompt; get() still looks when it matters
        return bool(self._keychain_value(provider))

    def get(self, provider: str) -> str | None:
        """The key to use. Reads the keychain if it may hold one, which can prompt to unlock it."""
        if self._session.get(provider):
            return self._session[provider]
        index = self._index()
        if index is None or provider in index or provider in self._cache:
            value = self._keychain_value(provider)
            if value:
                return value
        return self._env.get(ENV_VARS.get(provider, ""), "") or None

    def peek(self, provider: str) -> str | None:
        """The key if it is known without opening the keychain (for showing its last four characters)."""
        return (
            self._session.get(provider)
            or self._cache.get(provider)
            or (None if self._in_keychain(provider) else self._env.get(ENV_VARS.get(provider, ""), ""))
            or None
        )

    def source(self, provider: str) -> str | None:
        """Where the key in use comes from: 'session', 'keychain', 'environment', or None. Never prompts
        once the index exists."""
        if self._session.get(provider):
            return "session"
        if self._in_keychain(provider):
            return "keychain"
        if self._env.get(ENV_VARS.get(provider, "")):
            return "environment"
        return None

    def set(self, provider: str, key: str) -> None:
        key = key.strip()
        if not key:
            raise KeyStoreError("The API key is empty.")
        if not self.is_secure() and not self.wake():
            raise KeyStoreError(
                "No secure OS keychain is available (such as GNOME Keyring or KWallet), so the key was not saved. "
                f"You can set {ENV_VARS.get(provider, 'the provider environment variable')} instead."
            )
        try:
            self.backend.set_password(SERVICE, self.username(provider), key)
        except Exception as exc:
            raise KeyStoreError(f"The keychain refused the key: {exc}") from exc
        self._session.pop(provider, None)
        self._cache[provider] = key
        self._record(provider, True)

    def set_session(self, provider: str, key: str) -> None:
        """Use a key until ThetaIDE closes, without saving it anywhere."""
        key = key.strip()
        if not key:
            raise KeyStoreError("The API key is empty.")
        self._session[provider] = key

    def delete(self, provider: str) -> None:
        self._session.pop(provider, None)
        self._cache.pop(provider, None)
        self._record(provider, False)
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
