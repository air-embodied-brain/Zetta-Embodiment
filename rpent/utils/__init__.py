"""Utility helpers: config, logging, path resolution, templates."""

from rpent.utils.logging import get_logger, get_output_dir, init_output_dir
from rpent.utils.rpc import RpcClient, RpcError, parse_endpoint
from rpent.utils.socket_rpc import (
    SocketRpcClient,
    SocketRpcServer,
)
from rpent.utils.templates import (
    default_variables,
    substitute,
    substitute_text,
)

__all__ = [
    "RpcClient",
    "RpcError",
    "SocketRpcClient",
    "SocketRpcServer",
    "parse_endpoint",
    "default_variables",
    "get_logger",
    "get_output_dir",
    "init_output_dir",
    "substitute",
    "substitute_text",
]
