"""API keys in the OS keychain (macOS Keychain, Windows Credential Locker, Secret Service on Linux).

Keys are never written to settings.toml, the plugin's JSON storage, or any other file. On Linux
desktops that do not start a keychain themselves (i3, sway, a bare X session), the Secret Service
is usually installed but asleep; it is started over D-Bus on first use. If no secure keychain is
available at all, a key can be kept in memory for this session only; it is never written to disk.
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
        if self.backend is None:
            return None
        try:
            return self.backend.get_password(SERVICE, self.username(provider)) or None
        except Exception:  # a locked or unreachable keychain must not crash the IDE
            return None

    def get(self, provider: str) -> str | None:
        return (
            self._session.get(provider)
            or self._keychain_value(provider)
            or self._env.get(ENV_VARS.get(provider, ""), "")
            or None
        )

    def source(self, provider: str) -> str | None:
        """Where the key in use comes from: 'session', 'keychain', 'environment', or None."""
        if self._session.get(provider):
            return "session"
        if self._keychain_value(provider):
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

    def set_session(self, provider: str, key: str) -> None:
        """Use a key until ThetaIDE closes, without saving it anywhere."""
        key = key.strip()
        if not key:
            raise KeyStoreError("The API key is empty.")
        self._session[provider] = key

    def delete(self, provider: str) -> None:
        self._session.pop(provider, None)
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
