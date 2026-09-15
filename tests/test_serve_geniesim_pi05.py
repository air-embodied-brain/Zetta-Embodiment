# Copyright (c) 2026 Zetta Contributors
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from scripts.deployment.serve_geniesim_pi05 import (
    MODEL_SOURCE_REVISION,
    SeededPolicyService,
    model_identity,
)


def packet(seed=17):
    return msgpack.packb(
        {
            "method": "infer",
            "params": {
                "policy_rng": seed,
                "robot_type": "G2_omnipicker",
                "task_name": "pick_block_color",
            },
        },
        use_bin_type=True,
    )


def test_service_applies_each_seed_without_reapplying_absolute_transform(tmp_path):
    observed = []
    policy = SimpleNamespace(_rng=None)

    def infer(observation):
        observed.append(policy._rng)
        return {
            "actions": np.concatenate(
                [
                    np.full((2, 14), policy._rng[1], dtype=np.float32),
                    np.full((2, 2), -0.25, dtype=np.float32),
                ],
                axis=1,
            )
        }

    def adapt(request):
        return {**request["params"], "state": np.asarray([1.0] * 14 + [0.0] * 7)}

    policy.infer = infer
    service = SeededPolicyService(
        policy=policy,
        adapt_observation=adapt,
        build_response=lambda action: {
            "result": {"actions": action["actions"].tolist()}
        },
        random_key=lambda seed: ("jax-key", seed),
        model_version="weights-sha256",
        output_root=tmp_path,
    )
    for seed in (17, 19, 17):
        response = msgpack.unpackb(service.infer(packet(seed)), raw=False)["result"]
        assert response["policy_rng"] == seed
        assert response["model_version"] == "weights-sha256"
        assert response["actions"][0] == [float(seed)] * 14 + [-0.25, -0.25]
    assert observed == [("jax-key", 17), ("jax-key", 19), ("jax-key", 17)]
    assert (tmp_path / "first-request.msgpack").read_bytes() == packet(17)
    assert len((tmp_path / "inferences.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("seed", [None, True, -1, 2**32, 0.5])
def test_service_rejects_invalid_rng_before_inference(tmp_path, seed):
    service = SeededPolicyService(
        policy=None,
        adapt_observation=None,
        build_response=None,
        random_key=None,
        model_version="weights-sha256",
        output_root=tmp_path,
    )
    with pytest.raises(ValueError, match="policy_rng"):
        service.infer(packet(seed))
    assert not list(tmp_path.iterdir())


def test_service_rejects_malformed_model_output_without_acknowledging(tmp_path):
    service = SeededPolicyService(
        policy=SimpleNamespace(infer=lambda _: {"actions": np.zeros((1, 21))}),
        adapt_observation=lambda request: request["params"],
        build_response=None,
        random_key=lambda seed: seed,
        model_version="weights-sha256",
        output_root=tmp_path,
    )
    with pytest.raises(ValueError, match="invalid G2 joint actions"):
        service.infer(packet())
    assert not list(tmp_path.iterdir())


def test_identity_rejects_nonofficial_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.deployment.serve_geniesim_pi05.subprocess.check_output",
        lambda *args, **kwargs: MODEL_SOURCE_REVISION,
    )
    monkeypatch.setattr(
        "scripts.deployment.serve_geniesim_pi05.subprocess.run",
        lambda *args, **kwargs: None,
    )
    (tmp_path / "params").mkdir()
    (tmp_path / "assets").mkdir()
    (tmp_path / "params" / "weights").write_bytes(b"wrong weights")
    (tmp_path / "assets" / "norm_stats.json").write_text("{}")
    (tmp_path / "_CHECKPOINT_METADATA").write_text("{}")
    with pytest.raises(ValueError, match="pinned official weights"):
        model_identity(tmp_path, tmp_path)
