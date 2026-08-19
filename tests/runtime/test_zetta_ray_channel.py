from __future__ import annotations

import asyncio
import uuid

import pytest

pytestmark = pytest.mark.ray


@pytest.fixture(scope="module", autouse=True)
def _ray_cluster():
    ray = pytest.importorskip("ray")
    if not ray.is_initialized():
        ray.init(
            namespace="zetta-runtime",
            include_dashboard=False,
            ignore_reinit_error=True,
        )
    yield ray
    ray.shutdown()


def test_named_channel_connects_and_isolates_keys() -> None:
    from zetta.runtime.ray.channel import ZettaChannel

    async def scenario() -> None:
        name = f"channel-contract-{uuid.uuid4().hex}"
        owner = ZettaChannel.create(name, maxsize=1)
        peer = ZettaChannel.connect(name)
        try:
            await owner.put({"value": 1}, key="a")
            await peer.put({"value": 2}, key="b")
            assert await peer.get(key="a") == {"value": 1}
            assert await owner.get(key="b") == {"value": 2}
        finally:
            owner.shutdown()

    asyncio.run(scenario())


def test_named_channel_reports_queue_full() -> None:
    from zetta.runtime.ray.channel import ZettaChannel

    async def scenario() -> None:
        name = f"channel-full-{uuid.uuid4().hex}"
        channel = ZettaChannel.create(name, maxsize=1)
        try:
            await channel.put_nowait(1, key="same")
            with pytest.raises(asyncio.QueueFull):
                await channel.put_nowait(2, key="same")
        finally:
            channel.shutdown()

    asyncio.run(scenario())
