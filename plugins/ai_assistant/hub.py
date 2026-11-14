"""Component Hub access for the assistant: search the registry and install components.

These calls run on the assistant's worker thread because they wait on the network and on
installs that can take minutes. Anything that touches the window's HubClient state goes
through `on_gui`, and installs use the same HubClient.install_async path as the Hub UI, so the
Components panel, config tree and composed config all refresh exactly as after a manual install.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .tools import ToolError

INSTALL_TIMEOUT_S = 15 * 60
MAX_RESULTS = 25


def summary(component: Any) -> dict[str, Any]:
    deps = component.dependencies or {}
    return {
        "id": component.id,
        "name": component.name,
        "kind": component.kind,
        "version": component.version,
        "installed": component.installed_version if component.is_installed else False,
        "update_available": bool(getattr(component, "has_update", False)),
        "description": (component.description or "")[:240],
        "tags": list(component.tags or [])[:8],
        "author": getattr(component.author, "name", ""),
        "requires": {k: v for k, v in deps.items() if v} or None,
    }


class HubAccess:
    def __init__(
        self,
        window: Any,
        on_gui: Callable[[Callable[[], Any]], Any] = lambda fn: fn(),
        cancelled: Callable[[], bool] = lambda: False,
    ):
        self.window = window
        self.on_gui = on_gui
        self.cancelled = cancelled
        self._index_loaded = False

    @property
    def client(self) -> Any:
        client = getattr(self.window, "hub_client", None)
        if client is None:
            raise ToolError("The Component Hub is not available in this IDE session.")
        return client

    def _ensure_index(self, refresh: bool = False) -> None:
        """Fetch the registry once per session (ETag-cached), then publish it to the shared client."""
        if self._index_loaded and not refresh:
            return
        client = self.client
        components = client.fetch_index_sync(force=False, _publish=False)  # network: stays off the GUI thread
        if not components:
            raise ToolError("Could not load the Component Hub registry (offline and no cached copy).")

        def publish():
            client.components = components
            client.refresh_local_components()
            client.indexLoaded.emit(client.components)

        self.on_gui(publish)
        self._index_loaded = True

    def _component(self, component_id: str) -> Any:
        self._ensure_index()
        component = self.on_gui(lambda: self.client.get_component(component_id))
        if component is None:
            raise ToolError(f"No Hub component {component_id!r}. Use search_hub to find the right id.")
        return component

    def search(
        self,
        query: str = "",
        kind: str = "",
        installed: bool | None = None,
        refresh: bool = False,
    ) -> dict:
        self._ensure_index(refresh)
        found = self.on_gui(lambda: self.client.search(query, kind or None))
        if installed is not None:
            found = [c for c in found if bool(c.is_installed) == installed]
        return {
            "matches": len(found),
            "components": [summary(c) for c in found[:MAX_RESULTS]],
            **({"note": f"Showing the first {MAX_RESULTS}; narrow the query."} if len(found) > MAX_RESULTS else {}),
        }

    def describe_install(self, component_id: str, version: str | None = None) -> str:
        """The approval text. Raises ToolError when there is nothing to install."""
        component = self._component(component_id)
        target = version or component.version
        if component.is_installed and component.installed_version == target:
            raise ToolError(f"{component.name} {target} is already installed; nothing to do.")
        if version and component.releases and version not in component.releases:
            raise ToolError(f"{component.name} has no version {version}. Versions: {', '.join(component.releases)}")
        deps = component.dependencies or {}
        lines = [
            f"{component.name} ({component.id}) {target}",
            f"Kind: {component.kind}    Author: {getattr(component.author, 'name', 'unknown')}    "
            f"License: {component.license}",
        ]
        if component.is_installed:
            lines.append(f"Currently installed: {component.installed_version} (this replaces it)")
        if component.repository:
            lines.append(f"Source: {component.repository}")
        if deps.get("components"):
            lines.append("Also installs Hub components: " + ", ".join(map(str, deps["components"])))
        if deps.get("pip"):
            lines.append("Installs Python packages with pip: " + ", ".join(map(str, deps["pip"])))
        if component.description:
            lines += ["", component.description.strip()]
        lines += [
            "",
            "Hub components are third-party code that runs inside ThetaIDE and your training runs.",
        ]
        return "\n".join(lines)

    def install(self, component_id: str, version: str | None = None) -> str:
        component = self._component(component_id)
        client = self.client
        done = threading.Event()
        outcome: dict[str, Any] = {}

        def finished(cid: str, ok: bool, message: str) -> None:
            if cid == component.id:
                outcome.update(ok=ok, message=message)
                done.set()

        def start():
            client.installFinished.connect(finished)
            client.install_async(component, version)

        self.on_gui(start)
        try:
            deadline = time.monotonic() + INSTALL_TIMEOUT_S
            while not done.wait(0.25):
                if self.cancelled():
                    return f"Stopped waiting; {component.name} keeps installing in the background (see the Hub)."
                if time.monotonic() > deadline:
                    return (
                        f"{component.name} is still installing after {INSTALL_TIMEOUT_S // 60} minutes; check the Hub."
                    )
        finally:
            self.on_gui(lambda: _disconnect(client.installFinished, finished))
        if not outcome.get("ok"):
            raise ToolError(f"Installing {component.name} failed: {outcome.get('message') or 'unknown error'}")
        result = f"Installed {component.name} {version or component.version}."
        try:
            config = self.on_gui(lambda: client.local_components.installed_config(component, client.installer))
            relative = config.relative_to(client.workspace_dir / "in" / "config").as_posix()
            result += f" Its config is {relative}; read it before referencing it in an experiment."
        except Exception:
            pass
        return result


def _disconnect(signal: Any, slot: Callable) -> None:
    try:
        signal.disconnect(slot)
    except (TypeError, RuntimeError):
        pass
