from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


def decode_observation(payload: Any) -> list[list[Any]]:
    """Decode Voyager's intentionally double-encoded observation response."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, list):
        raise ValueError("Voyager observation must be an event list")
    return payload


def final_observation(events: list[list[Any]]) -> dict[str, Any]:
    for event_type, value in reversed(events):
        if event_type == "observe":
            return value
    raise ValueError("Voyager response contains no observe event")


@dataclass
class VoyagerHTTPClient:
    base_url: str = "http://127.0.0.1:3000"
    minecraft_port: int = 25565
    timeout_seconds: float = 120.0

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def reset(
        self,
        *,
        hard: bool = True,
        inventory: dict[str, int] | None = None,
        position: dict[str, float] | None = None,
        spread: bool = False,
        kill_on_hard_reset: bool = False,
        setup_commands: tuple[str, ...] = (),
        stuck_teleport: bool = False,
    ) -> list[list[Any]]:
        payload = {
            "port": self.minecraft_port,
            "reset": "hard" if hard else "soft",
            "inventory": inventory or {},
            "equipment": [None] * 6,
            "position": position,
            "spread": spread,
            "waitTicks": 5,
            "killOnHardReset": kill_on_hard_reset,
            "setupCommands": list(setup_commands),
            "stuckTeleport": stuck_teleport,
        }
        response = requests.post(
            f"{self.base_url}/start", json=payload, timeout=self.timeout_seconds
        )
        response.raise_for_status()
        return decode_observation(response.json())

    def step(self, *, code: str, programs: str = "") -> list[list[Any]]:
        response = requests.post(
            f"{self.base_url}/step",
            json={"code": code, "programs": programs},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return decode_observation(response.json())

    def stop(self) -> None:
        response = requests.post(f"{self.base_url}/stop", timeout=10)
        response.raise_for_status()
