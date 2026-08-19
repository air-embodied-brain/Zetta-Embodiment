"""OpenPI action model extracted from the RLinf OpenPI distribution."""

from zetta.policies.openpi.factory import build_openpi_model

__all__ = ["build_openpi_model", "get_model"]


def get_model(config, torch_dtype=None):
    from zetta.policies.openpi._factory_impl import get_model as implementation

    return implementation(config, torch_dtype=torch_dtype)
