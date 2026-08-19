"""ManiSkill environment integration owned by Zetta."""

__all__ = ["ManiskillEnv"]


def __getattr__(name: str):
    if name == "ManiskillEnv":
        from zetta.envs.maniskill.environment import ManiskillEnv

        return ManiskillEnv
    raise AttributeError(name)
