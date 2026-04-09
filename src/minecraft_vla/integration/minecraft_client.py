from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MinecraftClientConfig:
    mode: str
    host: str
    port: int
    connect_timeout_sec: float
    read_timeout_sec: float
    username: str


class MinecraftClient:
    def connect(self) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> Dict[str, Any]:
        raise NotImplementedError

    def step(self, action_id: int) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MockMinecraftClient(MinecraftClient):
    def __init__(self, config: MinecraftClientConfig) -> None:
        self.config = config
        self.connected = False
        self.tick = 0

    def connect(self) -> Dict[str, Any]:
        self.connected = True
        self.tick = 0
        return {
            "connected": True,
            "mode": "mock",
            "host": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
        }

    def reset(self) -> Dict[str, Any]:
        if not self.connected:
            raise RuntimeError("Client is not connected")
        self.tick = 0
        return {
            "tick": self.tick,
            "health": 20,
            "hunger": 20,
            "inventory_slots": 36,
        }

    def step(self, action_id: int) -> Dict[str, Any]:
        if not self.connected:
            raise RuntimeError("Client is not connected")

        self.tick += 1
        reward = ((action_id % 5) - 2) * 0.05
        done = False
        return {
            "tick": self.tick,
            "reward": float(reward),
            "done": done,
            "observation": {
                "health": 20,
                "hunger": 20,
                "position": [self.tick * 0.1, 64.0, 0.0],
            },
        }

    def close(self) -> None:
        self.connected = False


class TcpMinecraftClient(MinecraftClient):
    def __init__(self, config: MinecraftClientConfig, dry_run: bool = True) -> None:
        self.config = config
        self.dry_run = dry_run
        self.sock: Optional[socket.socket] = None

    def connect(self) -> Dict[str, Any]:
        if self.dry_run:
            return {
                "connected": True,
                "mode": "tcp-dry-run",
                "host": self.config.host,
                "port": self.config.port,
                "username": self.config.username,
                "note": "dry_run enabled, network call skipped",
            }

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(self.config.connect_timeout_sec)
        start = time.time()
        self.sock.connect((self.config.host, self.config.port))
        elapsed_ms = int((time.time() - start) * 1000)

        return {
            "connected": True,
            "mode": "tcp",
            "host": self.config.host,
            "port": self.config.port,
            "username": self.config.username,
            "connect_latency_ms": elapsed_ms,
        }

    def reset(self) -> Dict[str, Any]:
        return {
            "tick": 0,
            "health": 20,
            "hunger": 20,
            "inventory_slots": 36,
        }

    def step(self, action_id: int) -> Dict[str, Any]:
        return {
            "tick": 1,
            "reward": 0.0,
            "done": False,
            "observation": {
                "action_echo": action_id,
            },
        }

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None



def create_client(config: MinecraftClientConfig, dry_run: bool = True) -> MinecraftClient:
    mode = config.mode.lower()
    if mode == "mock":
        return MockMinecraftClient(config)
    if mode == "tcp":
        return TcpMinecraftClient(config, dry_run=dry_run)
    raise ValueError(f"Unsupported minecraft client mode: {config.mode}")
