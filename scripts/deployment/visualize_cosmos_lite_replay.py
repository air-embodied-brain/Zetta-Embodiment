# Copyright (c) 2026 Zetta Contributors
"""Render a self-contained HTML report from a Cosmos-Lite replay artifact."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_COLORS = (
    "#63e6be",
    "#74c0fc",
    "#b197fc",
    "#ffd43b",
    "#ff8787",
    "#4dabf7",
    "#69db7c",
    "#ffa94d",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Cosmos-Lite Model Replay")
    return parser


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.1f} ms"


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _points(
    values: Sequence[float | None],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    minimum: float,
    maximum: float,
) -> str:
    span = maximum - minimum or 1.0
    denominator = max(len(values) - 1, 1)
    points: list[str] = []
    for index, value in enumerate(values):
        if value is None:
            continue
        x = left + width * index / denominator
        y = top + height * (maximum - value) / span
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _latency_chart(
    client: Sequence[float],
    server: Sequence[float | None],
    *,
    start_index: int,
    title: str,
) -> str:
    visible_client = list(client[start_index:])
    visible_server = list(server[start_index:])
    numeric = visible_client + [value for value in visible_server if value is not None]
    if not numeric:
        return ""
    chart_width = 960.0
    chart_height = 300.0
    left = 66.0
    top = 30.0
    plot_width = 860.0
    plot_height = 210.0
    maximum = max(numeric) * 1.08
    minimum = 0.0
    grid: list[str] = []
    for tick in range(5):
        ratio = tick / 4
        y = top + plot_height * ratio
        value = maximum * (1.0 - ratio)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" '
            'y2="{y:.2f}" class="grid" />'
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'class="axis-label">{value:.0f}</text>'
        )
    client_points = _points(
        visible_client,
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
        minimum=minimum,
        maximum=maximum,
    )
    server_points = _points(
        visible_server,
        left=left,
        top=top,
        width=plot_width,
        height=plot_height,
        minimum=minimum,
        maximum=maximum,
    )
    request_start = start_index + 1
    request_end = start_index + len(visible_client)
    return f"""
      <section class="panel chart-panel">
        <div class="section-heading">
          <div><span class="eyebrow">LATENCY</span><h2>{_escape(title)}</h2></div>
          <div class="legend"><span class="client-dot"></span>端到端
          <span class="server-dot"></span>服务端推理</div>
        </div>
        <svg viewBox="0 0 {chart_width:.0f} {chart_height:.0f}" role="img"
             aria-label="{_escape(title)}">
          {"".join(grid)}
          <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}"
                y2="{top + plot_height}" class="axis" />
          <polyline points="{server_points}" class="server-line" />
          <polyline points="{client_points}" class="client-line" />
          <text x="{left}" y="{top + plot_height + 28}" class="axis-label">
            请求 {request_start}</text>
          <text x="{left + plot_width}" y="{top + plot_height + 28}"
                text-anchor="end" class="axis-label">请求 {request_end}</text>
          <text x="18" y="{top + plot_height / 2}" text-anchor="middle"
                transform="rotate(-90 18 {top + plot_height / 2})"
                class="axis-label">毫秒</text>
        </svg>
      </section>
    """


def _action_labels(dimension: int) -> list[str]:
    if dimension == 8:
        return [*(f"Joint {index + 1}" for index in range(7)), "Gripper"]
    return [f"Action {index + 1}" for index in range(dimension)]


def _validate_actions(value: Any) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError("request.actions must be a non-empty matrix")
    rows: list[list[float]] = []
    width: int | None = None
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or not row:
            raise ValueError(f"request.actions[{row_index}] must be a non-empty list")
        parsed = [
            _number(item, f"request.actions[{row_index}][{column_index}]")
            for column_index, item in enumerate(row)
        ]
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise ValueError("request.actions must be rectangular")
        rows.append(parsed)
    return rows


def _action_charts(actions: list[list[float]] | None, request_count: int) -> str:
    if actions is None:
        return """
          <section class="panel empty-state">
            <span class="eyebrow">ACTION CHUNK</span>
            <h2>动作轨迹未写入旧版 Replay</h2>
            <p>重新运行 smoke 时增加 <code>--include-actions</code> 即可生成轨迹图。</p>
          </section>
        """
    dimension = len(actions[0])
    labels = _action_labels(dimension)
    cards: list[str] = []
    for column, label in enumerate(labels):
        values = [row[column] for row in actions]
        minimum = min(values)
        maximum = max(values)
        padding = max((maximum - minimum) * 0.1, 1e-6)
        points = _points(
            values,
            left=24.0,
            top=18.0,
            width=292.0,
            height=104.0,
            minimum=minimum - padding,
            maximum=maximum + padding,
        )
        zero = ""
        if minimum <= 0 <= maximum:
            zero_y = 18.0 + 104.0 * (maximum + padding) / (
                maximum - minimum + 2 * padding
            )
            zero = (
                f'<line x1="24" y1="{zero_y:.2f}" x2="316" '
                f'y2="{zero_y:.2f}" class="zero-line" />'
            )
        cards.append(
            f"""
            <article class="action-card">
              <div class="action-title"><strong>{_escape(label)}</strong>
                <span>{minimum:.4f} → {maximum:.4f}</span></div>
              <svg viewBox="0 0 340 150" role="img"
                   aria-label="{_escape(label)} action trajectory">
                {zero}
                <polyline points="{points}" class="action-line"
                          style="stroke:{_COLORS[column % len(_COLORS)]}" />
                <text x="24" y="143" class="axis-label">step 1</text>
                <text x="316" y="143" text-anchor="end" class="axis-label">
                  step {len(values)}</text>
              </svg>
            </article>
            """
        )
    return f"""
      <section class="panel">
        <div class="section-heading">
          <div><span class="eyebrow">ACTION CHUNK</span><h2>{len(actions)} 步动作轨迹</h2></div>
          <p class="muted">展示首个请求；{request_count} 次请求的一致性由 hash 和最大误差校验。</p>
        </div>
        <div class="action-grid">{"".join(cards)}</div>
      </section>
    """


def _identity_table(
    identity: dict[str, Any], model_version: str, action_sha256: str
) -> str:
    rows = [
        ("Model version", model_version),
        ("Action SHA256", action_sha256),
        ("Model family", identity.get("model_family", "—")),
        ("Strategy", identity.get("strategy", "—")),
        ("Profile", identity.get("profile", "—")),
        ("Artifact", identity.get("artifact", "—")),
        ("Repository revision", identity.get("repository_revision", "—")),
        ("Manifest SHA256", identity.get("manifest_sha256", "—")),
        ("Resolved config SHA256", identity.get("resolved_config_sha256", "—")),
    ]
    body = "".join(
        f"<tr><th>{_escape(label)}</th><td><code>{_escape(value)}</code></td></tr>"
        for label, value in rows
    )
    return f"""
      <section class="panel">
        <span class="eyebrow">VERIFIED IDENTITY</span><h2>部署身份</h2>
        <div class="table-wrap"><table>{body}</table></div>
      </section>
    """


def _request_values(
    requests: Sequence[dict[str, Any]],
) -> tuple[list[float], list[float | None]]:
    client: list[float] = []
    server: list[float | None] = []
    for index, request in enumerate(requests):
        client.append(
            _number(request.get("latency_ms"), f"requests[{index}].latency_ms")
        )
        auxiliary = request.get("auxiliary_outputs")
        timing = auxiliary.get("server_timing") if isinstance(auxiliary, dict) else None
        infer_ms = timing.get("infer_ms") if isinstance(timing, dict) else None
        server.append(
            None
            if infer_ms is None
            else _number(infer_ms, f"requests[{index}].server_timing.infer_ms")
        )
    return client, server


def _unique_nonempty(values: Iterable[Any]) -> list[str]:
    return sorted({str(value) for value in values if value not in (None, "")})


def render_report(payload: dict[str, Any], *, title: str, source: str) -> str:
    """Render one validated replay payload as standalone HTML."""
    requests_value = payload.get("requests")
    if not isinstance(requests_value, list) or not requests_value:
        raise ValueError("replay report must contain at least one request")
    if not all(isinstance(item, dict) for item in requests_value):
        raise ValueError("replay requests must be objects")
    requests: list[dict[str, Any]] = requests_value
    client, server = _request_values(requests)
    warm_client = client[1:] or client
    warm_server = [value for value in server[1:] if value is not None]
    if not warm_server:
        warm_server = [value for value in server if value is not None]
    hashes = _unique_nonempty(request.get("action_sha256") for request in requests)
    versions = _unique_nonempty(request.get("model_version") for request in requests)
    first = requests[0]
    shape = first.get("shape", "—")
    shape_text = (
        "×".join(str(value) for value in shape)
        if isinstance(shape, list)
        else str(shape)
    )
    deterministic = payload.get("deterministic") is True
    max_diff = _number(payload.get("max_action_abs_diff", 0.0), "max_action_abs_diff")
    action_data = first.get("actions")
    actions = None if action_data is None else _validate_actions(action_data)
    auxiliary = first.get("auxiliary_outputs")
    identity_value = (
        auxiliary.get("cosmos_lite_identity") if isinstance(auxiliary, dict) else None
    )
    identity = identity_value if isinstance(identity_value, dict) else {}
    model_version = versions[0] if len(versions) == 1 else f"{len(versions)} versions"
    input_value = payload.get("input")
    replay_input = input_value if isinstance(input_value, dict) else {}
    instruction = replay_input.get("instruction", "—")
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cold_latency = client[0]
    p50_client = statistics.median(warm_client)
    p95_client = _percentile(warm_client, 0.95)
    p50_server = statistics.median(warm_server) if warm_server else None
    status_class = "pass" if deterministic and len(hashes) == 1 else "fail"
    status_text = "PASS · 完全一致" if status_class == "pass" else "CHECK · 存在差异"
    cards = (
        ("请求数", str(len(requests)), "连续真实推理"),
        ("确定性", status_text, f"最大误差 {max_diff:.2g}"),
        (
            "动作输出",
            f"{shape_text} · {first.get('dtype', '—')}",
            f"{len(hashes)} 个唯一 hash",
        ),
        ("冷启动", _fmt_ms(cold_latency), "包含首次编译"),
        ("热态 P50", _fmt_ms(p50_client), f"P95 {_fmt_ms(p95_client)}"),
        ("服务端 P50", _fmt_ms(p50_server), "排除首次请求"),
    )
    cards_html = "".join(
        f"""
        <article class="metric"><span>{_escape(label)}</span>
          <strong class="{status_class if label == "确定性" else ""}">{_escape(value)}</strong>
          <small>{_escape(note)}</small></article>
        """
        for label, value, note in cards
    )
    action_html = _action_charts(actions, len(requests))
    all_latency = _latency_chart(client, server, start_index=0, title="全量请求延迟")
    warm_latency = (
        _latency_chart(
            client, server, start_index=1, title="热态请求延迟（排除首次编译）"
        )
        if len(client) > 1
        else ""
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape(title)}</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#101d2d; --line:#263b52;
      --text:#eef6ff; --muted:#9db0c5; --teal:#63e6be; --blue:#74c0fc;
      --red:#ff8787; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:radial-gradient(circle at 15% 0,#17314b 0,#07111f 42%);
      color:var(--text); font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1180px,calc(100% - 32px)); margin:0 auto; padding:48px 0 72px; }}
    header {{ margin-bottom:28px; }}
    h1 {{ margin:5px 0 8px; font-size:clamp(30px,5vw,54px); letter-spacing:-.035em; }}
    h2 {{ margin:4px 0 12px; font-size:22px; }}
    p {{ margin:0; }} .muted, .meta {{ color:var(--muted); }}
    .eyebrow {{ color:var(--teal); font-size:12px; font-weight:800; letter-spacing:.14em; }}
    .metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin:24px 0; }}
    .metric,.panel,.action-card {{ background:linear-gradient(145deg,rgba(18,34,52,.96),rgba(11,24,39,.96));
      border:1px solid var(--line); border-radius:16px; box-shadow:0 16px 45px rgba(0,0,0,.18); }}
    .metric {{ padding:18px; min-height:126px; }} .metric span,.metric small {{ display:block;color:var(--muted); }}
    .metric strong {{ display:block;margin:8px 0 5px;font-size:22px; }}
    .metric strong.pass {{ color:var(--teal); }} .metric strong.fail {{ color:var(--red); }}
    .panel {{ padding:22px; margin-top:16px; }} .chart-panel svg {{ width:100%; min-height:260px; }}
    .section-heading {{ display:flex;justify-content:space-between;gap:20px;align-items:flex-end; }}
    .legend {{ color:var(--muted);white-space:nowrap; }}
    .client-dot,.server-dot {{ display:inline-block;width:9px;height:9px;border-radius:50%;margin:0 6px 0 14px; }}
    .client-dot {{ background:var(--teal); }} .server-dot {{ background:var(--blue); }}
    .grid {{ stroke:#20344a;stroke-width:1; }} .axis {{ stroke:#51667d;stroke-width:1; }}
    .axis-label {{ fill:#8298ae;font-size:12px; }}
    .client-line,.server-line,.action-line {{ fill:none;stroke-linecap:round;stroke-linejoin:round; }}
    .client-line {{ stroke:var(--teal);stroke-width:3; }} .server-line {{ stroke:var(--blue);stroke-width:2;opacity:.8; }}
    .action-grid {{ display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:18px; }}
    .action-card {{ padding:14px;background:#0b1827; }} .action-card svg {{ width:100%; }}
    .action-title {{ display:flex;justify-content:space-between;color:var(--muted); }}
    .action-title strong {{ color:var(--text); }} .action-line {{ stroke-width:3; }}
    .zero-line {{ stroke:#42566b;stroke-width:1;stroke-dasharray:4 4; }}
    .table-wrap {{ overflow:auto; }} table {{ width:100%;border-collapse:collapse; }}
    th,td {{ padding:10px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top; }}
    th {{ width:190px;color:var(--muted);font-weight:500; }} code {{ color:#c5f6fa;overflow-wrap:anywhere; }}
    .empty-state {{ border-style:dashed;text-align:center;padding:42px; }}
    footer {{ color:var(--muted);margin-top:22px;font-size:13px; }}
    @media (max-width:760px) {{ .metrics,.action-grid {{ grid-template-columns:1fr; }}
      .section-heading {{ align-items:flex-start;flex-direction:column; }} .legend {{ white-space:normal; }} }}
  </style>
</head>
<body><main>
  <header><span class="eyebrow">ZETTA × NVIDIA COSMOS-LITE</span>
    <h1>{_escape(title)}</h1>
    <p class="meta">输入指令：{_escape(instruction)} · 生成时间：{_escape(generated_at)}</p>
  </header>
  <section class="metrics">{cards_html}</section>
  {all_latency}
  {warm_latency}
  {action_html}
  {_identity_table(identity, model_version, hashes[0] if len(hashes) == 1 else "—")}
  <footer>数据源：<code>{_escape(source)}</code> · 报告为自包含 HTML，无外部资源。</footer>
</main></body></html>
"""


def main() -> int:
    args = _parser().parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("replay report root must be an object")
    rendered = render_report(payload, title=args.title, source=str(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
