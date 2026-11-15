"""Windows audio capture via WASAPI (pyaudiowpatch).

Loopback records whatever the default speakers/headset play, i.e. the other participants,
independent of whether the call runs in Zoom, Teams or Meet.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


def _pyaudio():
    try:
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise RuntimeError(
            "pyaudiowpatch is missing. It only exists on Windows: pip install pyaudiowpatch"
        ) from exc
    return pyaudio


@dataclass
class Device:
    index: int
    name: str
    channels: int
    sample_rate: int

    @classmethod
    def from_info(cls, info: dict[str, Any]) -> "Device":
        return cls(
            index=int(info["index"]),
            name=str(info["name"]),
            channels=max(1, int(info["maxInputChannels"])),
            sample_rate=int(info["defaultSampleRate"]),
        )


def _wasapi_devices(pa) -> tuple[dict, list[dict]]:
    pyaudio = _pyaudio()
    wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
    devices = [
        pa.get_device_info_by_index(i)
        for i in range(pa.get_device_count())
        if pa.get_device_info_by_index(i)["hostApi"] == wasapi["index"]
    ]
    return wasapi, devices


def find_loopback(pa, name_hint: str = "") -> Device:
    wasapi, devices = _wasapi_devices(pa)
    loopbacks = [d for d in devices if d.get("isLoopbackDevice")]
    if name_hint:
        for d in loopbacks:
            if name_hint.lower() in d["name"].lower():
                return Device.from_info(d)
        raise RuntimeError(f"No loopback device matches '{name_hint}'. Run with --list-devices.")
    speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
    if speakers.get("isLoopbackDevice"):
        return Device.from_info(speakers)
    for d in loopbacks:
        if speakers["name"] in d["name"]:
            return Device.from_info(d)
    raise RuntimeError("No loopback device found for the default output device.")


def find_mic(pa, name_hint: str = "") -> Device:
    wasapi, devices = _wasapi_devices(pa)
    if name_hint:
        for d in devices:
            if not d.get("isLoopbackDevice") and d["maxInputChannels"] > 0 and name_hint.lower() in d["name"].lower():
                return Device.from_info(d)
        raise RuntimeError(f"No microphone matches '{name_hint}'. Run with --list-devices.")
    return Device.from_info(pa.get_device_info_by_index(wasapi["defaultInputDevice"]))


def list_devices() -> None:
    pyaudio = _pyaudio()
    with pyaudio.PyAudio() as pa:
        wasapi, devices = _wasapi_devices(pa)
        print("WASAPI devices (use a name fragment in config.toml [audio]):\n")
        for d in devices:
            kind = "LOOPBACK" if d.get("isLoopbackDevice") else ("INPUT" if d["maxInputChannels"] > 0 else "OUTPUT")
            default = ""
            if d["index"] == wasapi["defaultInputDevice"]:
                default = "  (default mic)"
            elif d["index"] == wasapi["defaultOutputDevice"]:
                default = "  (default speakers)"
            print(f"  [{kind:8}] {d['name']}  {int(d['defaultSampleRate'])} Hz{default}")


class AudioCapture:
    """Pushes mono int16 PCM chunks into an asyncio.Queue from PortAudio's callback thread."""

    def __init__(self, pa, device: Device, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, chunk_ms: int):
        self._pa = pa
        self.device = device
        self._loop = loop
        self._queue = queue
        self._chunk_frames = max(1, device.sample_rate * chunk_ms // 1000)
        self._stream = None
        self._level = 0.0
        self._level_ts = 0.0

    @property
    def level(self) -> float:
        """RMS of the latest chunk, 0..1. Loopback delivers no callbacks while nothing plays, hence the timeout."""
        return self._level if time.monotonic() - self._level_ts < 0.5 else 0.0

    def _push(self, data: bytes) -> None:
        if self._queue.full():  # STT is behind or reconnecting: drop the oldest audio, keep latency low
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._queue.put_nowait(data)

    def _callback(self, in_data, frame_count, time_info, status):
        pyaudio = _pyaudio()
        samples = np.frombuffer(in_data, dtype=np.int16)
        if self.device.channels > 1:
            samples = samples.reshape(-1, self.device.channels).mean(axis=1).astype(np.int16)
        if samples.size:
            self._level = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) / 32768.0
            self._level_ts = time.monotonic()
        self._loop.call_soon_threadsafe(self._push, samples.tobytes())
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        pyaudio = _pyaudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.device.channels,
            rate=self.device.sample_rate,
            input=True,
            input_device_index=self.device.index,
            frames_per_buffer=self._chunk_frames,
            stream_callback=self._callback,
        )
        log.info("Capturing %s (%d Hz, %d ch)", self.device.name, self.device.sample_rate, self.device.channels)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
