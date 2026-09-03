"""RoboTwin 2.0 environment integration owned by Zetta."""

__all__ = ["RoboTwinEnv"]


def __getattr__(name: str):
    """Import ``RoboTwinEnv`` lazily.

    The module pulls in ``gymnasium`` and, at construction time, the ``robotwin``
    package, neither of which exists in the minimal test environment. Keeping
    the import lazy means ``import zetta.envs.robotwin`` stays cheap and safe.

    Args:
        name: Attribute being looked up.

    Returns:
        The ``RoboTwinEnv`` class.

    Raises:
        AttributeError: Any other attribute name.
    """
    if name == "RoboTwinEnv":
        from zetta.envs.robotwin.environment import RoboTwinEnv

        return RoboTwinEnv
    raise AttributeError(name)
