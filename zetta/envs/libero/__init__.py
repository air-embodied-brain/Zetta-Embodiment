"""LIBERO environment utilities owned by Zetta."""

__all__ = ["LiberoEnv", "ShArray", "ZettaLiberoEnv"]


def __getattr__(name: str):
    if name in {"LiberoEnv", "ZettaLiberoEnv"}:
        from zetta.envs.libero.environment import LiberoEnv

        return LiberoEnv
    if name == "ShArray":
        from zetta.envs.libero.vector_env import ShArray

        return ShArray
    raise AttributeError(name)
