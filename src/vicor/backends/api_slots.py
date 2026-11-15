"""Optional cross-process concurrency limit for two parallel API game runs."""

import asyncio
import fcntl
import os
from contextlib import asynccontextmanager
from pathlib import Path


@asynccontextmanager
async def api_slot(directory: str | Path | None, slots: int = 2):
    if directory is None:
        yield
        return
    if not 1 <= slots <= 32:
        raise ValueError("API slots must be between 1 and 32")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    acquired = None
    try:
        while acquired is None:
            for index in range(slots):
                fd = os.open(folder / f"slot-{index}.lock",
                             os.O_CREAT | os.O_RDWR, 0o600)
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(fd)
                    continue
                acquired = fd
                break
            if acquired is None:
                await asyncio.sleep(0.075)
        yield
    finally:
        if acquired is not None:
            fcntl.flock(acquired, fcntl.LOCK_UN)
            os.close(acquired)
