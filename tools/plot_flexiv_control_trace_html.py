#!/usr/bin/env python3
"""Generate a standalone interactive HTML from a Flexiv control trace."""

from __future__ import annotations

import argparse
import html
import json
import webbrowser
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from plotly.offline import get_plotlyjs, get_plotlyjs_version
from plotly.utils import PlotlyJSONEncoder

from plot_flexiv_control_trace import (
    AXIS_NAMES,
    SIDES,
    TraceData,
    _command_key,
    _command_label,
    _median_rate_hz,
    _nearest_indices,
    _relative_rotvec,
    _rotation_error_deg,
    _time_origin_ns,
    _times,
    _vectors,
    load_trace,
)


COLORS = ("#d62728", "#2ca02c", "#1f77b4")
STAGE_DASHES = {"policy": "dash", "applied": "solid", "target": "dot", "command": "solid", "TCP": "dash"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an interactive Flexiv control-chain HTML. Use the chart "
            "selector, click legend entries to hide curves, wheel to zoom Y, "
            "and Shift+wheel to zoom X."
        )
    )
    parser.add_argument("--log", type=Path, required=True, help="*_control_trace.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML (default: <log-stem>_interactive.html)",
    )
    parser.add_argument(
        "--plotly-js",
        choices=("inline", "cdn"),
        default="inline",
        help="inline creates an offline standalone HTML; cdn creates a smaller online HTML",
    )
    parser.add_argument("--open", action="store_true", help="Open the generated HTML")
    return parser.parse_args()


def _line_trace(
    x: np.ndarray,
    y: np.ndarray,
    name: str,
    *,
    color: str | None = None,
    dash: str = "solid",
    hover_unit: str = "",
    webgl: bool = True,
) -> dict[str, Any]:
    suffix = f" {hover_unit}" if hover_unit else ""
    return {
        "type": "scattergl" if webgl else "scatter",
        "mode": "lines",
        "x": x,
        "y": y,
        "name": name,
        "line": {"color": color, "dash": dash, "width": 1.5},
        "hovertemplate": f"t=%{{x:.4f}} s<br>{html.escape(name)}=%{{y:.6g}}{suffix}<extra></extra>",
    }


def _chart(
    *,
    group: str,
    label: str,
    title: str,
    y_title: str,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "group": group,
        "label": label,
        "data": traces,
        "layout": {
            "title": {"text": title, "x": 0.02, "xanchor": "left"},
            "xaxis": {
                "title": "time since first control event (s)",
                "showgrid": True,
                "gridcolor": "rgba(100,116,139,0.20)",
                "zeroline": False,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikedash": "dot",
                "spikecolor": "#64748b",
            },
            "yaxis": {
                "title": y_title,
                "showgrid": True,
                "gridcolor": "rgba(100,116,139,0.20)",
                "zerolinecolor": "rgba(100,116,139,0.35)",
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikedash": "dot",
                "spikecolor": "#64748b",
            },
            "hovermode": "closest",
            "hoverdistance": 40,
            "spikedistance": -1,
            "dragmode": "pan",
            "legend": {
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0.0,
                "itemclick": "toggle",
                "itemdoubleclick": "toggleothers",
            },
            "margin": {"l": 82, "r": 28, "t": 110, "b": 70},
            "paper_bgcolor": "#ffffff",
            "plot_bgcolor": "#ffffff",
            "font": {"family": "Inter, system-ui, sans-serif", "size": 13},
        },
    }


def _axis_stage_traces(
    x_by_stage: dict[str, np.ndarray],
    values_by_stage: dict[str, np.ndarray],
    *,
    multiplier: float,
    unit: str,
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for stage, values in values_by_stage.items():
        for axis, color in enumerate(COLORS):
            traces.append(
                _line_trace(
                    x_by_stage[stage],
                    values[:, axis] * multiplier,
                    f"{stage} {AXIS_NAMES[axis]}",
                    color=color,
                    dash=STAGE_DASHES[stage],
                    hover_unit=unit,
                )
            )
    return traces


def build_charts(trace: TraceData) -> OrderedDict[str, dict[str, Any]]:
    origin_ns = _time_origin_ns(trace)
    policy_t = _times(trace.policy, origin_ns)
    servo_t = _times(trace.servo, origin_ns)
    feedback_t = _times(trace.feedback, origin_ns)
    charts: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for side in SIDES:
        side_label = side.capitalize()
        group = f"{side_label} arm"
        policy_delta = _vectors(trace.policy, "policy_delta_pose6", side, 6)
        applied_delta = _vectors(trace.policy, "applied_delta_pose6", side, 6)
        target = _vectors(trace.servo, "accumulated_target_pose7", side, 7)
        command = _vectors(trace.servo, _command_key(trace), side, 7)
        actual = _vectors(trace.feedback, "actual_tcp_pose7", side, 7)
        reference = actual[0]

        charts[f"{side}_translation_delta"] = _chart(
            group=group,
            label="Translation delta",
            title=f"{side_label}: policy delta vs applied delta",
            y_title="translation delta (mm)",
            traces=_axis_stage_traces(
                {"policy": policy_t, "applied": policy_t},
                {"policy": policy_delta[:, :3], "applied": applied_delta[:, :3]},
                multiplier=1000.0,
                unit="mm",
            ),
        )
        charts[f"{side}_tcp_position"] = _chart(
            group=group,
            label="Target / command / TCP position",
            title=(
                f"{side_label}: accumulated target -> {_command_label(trace)} "
                "-> actual TCP"
            ),
            y_title="position relative to first TCP sample (mm)",
            traces=_axis_stage_traces(
                {"target": servo_t, "command": servo_t, "TCP": feedback_t},
                {
                    "target": target[:, :3] - reference[:3],
                    "command": command[:, :3] - reference[:3],
                    "TCP": actual[:, :3] - reference[:3],
                },
                multiplier=1000.0,
                unit="mm",
            ),
        )
        charts[f"{side}_rotation_delta"] = _chart(
            group=group,
            label="Rotation delta",
            title=f"{side_label}: policy rotation delta vs applied rotation delta",
            y_title="rotation delta (mrad)",
            traces=_axis_stage_traces(
                {"policy": policy_t, "applied": policy_t},
                {"policy": policy_delta[:, 3:], "applied": applied_delta[:, 3:]},
                multiplier=1000.0,
                unit="mrad",
            ),
        )
        charts[f"{side}_orientation"] = _chart(
            group=group,
            label="Target / command / TCP orientation",
            title=f"{side_label}: target -> command -> actual TCP orientation",
            y_title="rotvec relative to first TCP sample (deg)",
            traces=_axis_stage_traces(
                {"target": servo_t, "command": servo_t, "TCP": feedback_t},
                {
                    "target": _relative_rotvec(target, reference),
                    "command": _relative_rotvec(command, reference),
                    "TCP": _relative_rotvec(actual, reference),
                },
                multiplier=180.0 / np.pi,
                unit="deg",
            ),
        )

        nearest = _nearest_indices(servo_t, feedback_t)
        nearest_target = target[nearest]
        nearest_command = command[nearest]
        charts[f"{side}_translation_error"] = _chart(
            group=group,
            label="Translation tracking error",
            title=f"{side_label}: translation tracking error",
            y_title="error norm (mm)",
            traces=[
                _line_trace(
                    feedback_t,
                    np.linalg.norm(nearest_target[:, :3] - nearest_command[:, :3], axis=1)
                    * 1000.0,
                    "target-command",
                    color="#8b5cf6",
                    hover_unit="mm",
                ),
                _line_trace(
                    feedback_t,
                    np.linalg.norm(nearest_command[:, :3] - actual[:, :3], axis=1)
                    * 1000.0,
                    "command-TCP",
                    color="#f59e0b",
                    hover_unit="mm",
                ),
                _line_trace(
                    feedback_t,
                    np.linalg.norm(nearest_target[:, :3] - actual[:, :3], axis=1)
                    * 1000.0,
                    "target-TCP",
                    color="#0f766e",
                    hover_unit="mm",
                ),
            ],
        )
        charts[f"{side}_rotation_error"] = _chart(
            group=group,
            label="Rotation tracking error",
            title=f"{side_label}: rotation tracking error",
            y_title="error angle (deg)",
            traces=[
                _line_trace(
                    feedback_t,
                    _rotation_error_deg(nearest_target, nearest_command),
                    "target-command",
                    color="#8b5cf6",
                    hover_unit="deg",
                ),
                _line_trace(
                    feedback_t,
                    _rotation_error_deg(nearest_command, actual),
                    "command-TCP",
                    color="#f59e0b",
                    hover_unit="deg",
                ),
                _line_trace(
                    feedback_t,
                    _rotation_error_deg(nearest_target, actual),
                    "target-TCP",
                    color="#0f766e",
                    hover_unit="deg",
                ),
            ],
        )
        gripper = np.asarray(
            [record["gripper_command"][side] for record in trace.policy],
            dtype=float,
        )
        gripper_trace = _line_trace(
            policy_t,
            gripper,
            "gripper command",
            color="#2563eb",
            webgl=False,
        )
        gripper_trace["line"]["shape"] = "hv"
        charts[f"{side}_gripper"] = _chart(
            group=group,
            label="Gripper command",
            title=f"{side_label}: normalized gripper command",
            y_title="normalized command",
            traces=[gripper_trace],
        )

    interval_traces: list[dict[str, Any]] = []
    for label, times, color in (
        ("policy action", policy_t, "#2563eb"),
        ("servo command", servo_t, "#dc2626"),
        ("TCP feedback", feedback_t, "#059669"),
    ):
        if len(times) > 1:
            interval_traces.append(
                _line_trace(
                    times[1:],
                    np.diff(times) * 1000.0,
                    label,
                    color=color,
                    hover_unit="ms",
                )
            )
    charts["timing_intervals"] = _chart(
        group="Timing",
        label="Event intervals",
        title="Policy / servo / TCP feedback intervals",
        y_title="interval (ms)",
        traces=interval_traces,
    )
    charts["timing_servo"] = _chart(
        group="Timing",
        label="Servo execution cost",
        title=f"{_command_label(trace).capitalize()} send and producer cost",
        y_title="duration (ms)",
        traces=[
            _line_trace(
                servo_t,
                np.asarray([record["send_ms"] for record in trace.servo]),
                "dual-arm send",
                color="#dc2626",
                hover_unit="ms",
            ),
            _line_trace(
                servo_t,
                np.asarray([record["loop_ms"] for record in trace.servo]),
                "servo loop",
                color="#7c3aed",
                hover_unit="ms",
            ),
        ],
    )
    charts["timing_feedback"] = _chart(
        group="Timing",
        label="TCP feedback read cost",
        title="Dual-arm TCP feedback sampling cost",
        y_title="duration (ms)",
        traces=[
            _line_trace(
                feedback_t,
                np.asarray([record["read_ms"] for record in trace.feedback]),
                "TCP feedback read",
                color="#059669",
                hover_unit="ms",
            )
        ],
    )
    return charts


def _plotly_script(mode: str) -> str:
    if mode == "inline":
        return f"<script>{get_plotlyjs()}</script>"
    version = get_plotlyjs_version()
    return f'<script src="https://cdn.plot.ly/plotly-{version}.min.js"></script>'


def _build_html(
    trace_path: Path,
    trace: TraceData,
    charts: OrderedDict[str, dict[str, Any]],
    *,
    plotly_js: str,
) -> str:
    chart_json = json.dumps(
        charts,
        cls=PlotlyJSONEncoder,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    summary = {
        "policy": {"count": len(trace.policy), "hz": _median_rate_hz(trace.policy)},
        "servo": {"count": len(trace.servo), "hz": _median_rate_hz(trace.servo)},
        "feedback": {"count": len(trace.feedback), "hz": _median_rate_hz(trace.feedback)},
        "dropped": int(trace.footer.get("dropped_records", 0)),
    }
    summary_json = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    title = html.escape(f"Flexiv DP3 control trace — {trace_path.name}")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  {_plotly_script(plotly_js)}
  <style>
    :root {{ color-scheme: light; font-family: Inter, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f8fafc; color: #0f172a; overflow: hidden; }}
    header {{
      height: 104px; padding: 12px 18px; background: white;
      border-bottom: 1px solid #e2e8f0; display: flex; gap: 18px;
      align-items: center; flex-wrap: wrap;
    }}
    .title {{ min-width: 340px; flex: 1; }}
    .title h1 {{ margin: 0 0 5px; font-size: 18px; }}
    .title p {{ margin: 0; color: #475569; font-size: 12px; }}
    .controls {{ display: flex; align-items: end; gap: 9px; flex-wrap: wrap; }}
    label {{ display: flex; flex-direction: column; gap: 4px; font-size: 12px; color: #475569; }}
    select, button {{
      height: 34px; border: 1px solid #cbd5e1; border-radius: 7px;
      background: white; color: #0f172a; padding: 0 10px; font-size: 13px;
    }}
    select {{ min-width: 290px; }}
    button {{ cursor: pointer; }}
    button:hover {{ background: #f1f5f9; }}
    #status {{ font: 12px ui-monospace, SFMono-Regular, monospace; color: #334155; }}
    #plot {{ width: 100vw; height: calc(100vh - 104px); background: white; }}
    .warning {{ color: #b91c1c; font-weight: 600; }}
  </style>
</head>
<body>
  <header>
    <div class="title">
      <h1>{title}</h1>
      <p>悬停：显示数值；左键拖动：平移；左键单击数据点：固定 Data Tip。图例单击：隐藏曲线。滚轮：仅缩放 Y 轴；Shift + 滚轮：仅缩放 X 轴。</p>
    </div>
    <div class="controls">
      <label>选择图表<select id="chart-select"></select></label>
      <button id="show-all" type="button">显示全部曲线</button>
      <button id="hide-all" type="button">隐藏全部曲线</button>
      <button id="clear-tips" type="button">清除数据标记</button>
      <button id="reset-axes" type="button">复位坐标轴</button>
      <span id="status"></span>
    </div>
  </header>
  <div id="plot"></div>
  <script>
    const charts = {chart_json};
    const summary = {summary_json};
    const graph = document.getElementById('plot');
    const selector = document.getElementById('chart-select');
    let currentKey = Object.keys(charts)[0];

    const groups = new Map();
    for (const [key, chart] of Object.entries(charts)) {{
      if (!groups.has(chart.group)) groups.set(chart.group, []);
      groups.get(chart.group).push([key, chart.label]);
    }}
    for (const [group, entries] of groups.entries()) {{
      const optgroup = document.createElement('optgroup');
      optgroup.label = group;
      for (const [key, label] of entries) {{
        const option = document.createElement('option');
        option.value = key;
        option.textContent = label;
        optgroup.appendChild(option);
      }}
      selector.appendChild(optgroup);
    }}

    const plotConfig = {{
      responsive: true,
      displaylogo: false,
      scrollZoom: false,
      doubleClick: 'reset+autosize',
      modeBarButtonsToRemove: ['lasso2d', 'select2d']
    }};

    function saveCurrentState() {{
      if (!graph.data || !charts[currentKey]) return;
      graph.data.forEach((trace, index) => {{
        charts[currentKey].data[index].visible = trace.visible;
      }});
      const full = graph._fullLayout;
      if (full && full.xaxis && full.yaxis) {{
        charts[currentKey].savedRanges = {{
          x: [...full.xaxis.range],
          y: [...full.yaxis.range]
        }};
      }}
    }}

    function renderChart(key) {{
      const chart = charts[key];
      const layout = Object.assign({{}}, chart.layout, {{uirevision: key}});
      layout.xaxis = Object.assign({{}}, chart.layout.xaxis);
      layout.yaxis = Object.assign({{}}, chart.layout.yaxis);
      if (chart.savedRanges) {{
        layout.xaxis.range = chart.savedRanges.x;
        layout.xaxis.autorange = false;
        layout.yaxis.range = chart.savedRanges.y;
        layout.yaxis.autorange = false;
      }}
      currentKey = key;
      selector.value = key;
      const renderPromise = Plotly.react(graph, chart.data, layout, plotConfig);
      const droppedClass = summary.dropped ? 'warning' : '';
      document.getElementById('status').innerHTML =
        `policy ${{summary.policy.hz.toFixed(1)}} Hz · servo ${{summary.servo.hz.toFixed(1)}} Hz · ` +
        `feedback ${{summary.feedback.hz.toFixed(1)}} Hz · ` +
        `<span class="${{droppedClass}}">dropped=${{summary.dropped}}</span>`;
      return renderPromise;
    }}

    selector.addEventListener('change', () => {{
      saveCurrentState();
      renderChart(selector.value);
    }});
    document.getElementById('show-all').addEventListener('click', () => {{
      Plotly.restyle(graph, {{visible: true}});
    }});
    document.getElementById('hide-all').addEventListener('click', () => {{
      Plotly.restyle(graph, {{visible: 'legendonly'}});
    }});
    document.getElementById('clear-tips').addEventListener('click', () => {{
      Plotly.relayout(graph, {{annotations: []}});
    }});
    document.getElementById('reset-axes').addEventListener('click', () => {{
      delete charts[currentKey].savedRanges;
      Plotly.relayout(graph, {{'xaxis.autorange': true, 'yaxis.autorange': true}});
    }});

    function scaledRange(range, center, factor) {{
      return [center + (range[0] - center) * factor,
              center + (range[1] - center) * factor];
    }}

    graph.addEventListener('wheel', (event) => {{
      const full = graph._fullLayout;
      if (!full || !full._size || !full.xaxis || !full.yaxis) return;
      const rect = graph.getBoundingClientRect();
      const size = full._size;
      const px = event.clientX - rect.left - size.l;
      const py = event.clientY - rect.top - size.t;
      if (px < 0 || px > size.w || py < 0 || py > size.h) return;
      event.preventDefault();
      event.stopPropagation();
      const wheelDelta = Math.abs(event.deltaY) >= Math.abs(event.deltaX)
        ? event.deltaY : event.deltaX;
      const factor = Math.exp(Math.max(-120, Math.min(120, wheelDelta)) * 0.0025);
      if (event.shiftKey) {{
        const range = full.xaxis.range.map(Number);
        const fraction = px / size.w;
        const center = range[0] + fraction * (range[1] - range[0]);
        Plotly.relayout(graph, {{
          'xaxis.range': scaledRange(range, center, factor),
          'xaxis.autorange': false
        }});
      }} else {{
        const range = full.yaxis.range.map(Number);
        const fraction = 1 - py / size.h;
        const center = range[0] + fraction * (range[1] - range[0]);
        Plotly.relayout(graph, {{
          'yaxis.range': scaledRange(range, center, factor),
          'yaxis.autorange': false
        }});
      }}
    }}, {{passive: false, capture: true}});

    renderChart(currentKey).then(() => {{
      graph.on('plotly_click', (eventData) => {{
        if (!eventData || !eventData.points || !eventData.points.length) return;
        const point = eventData.points[0];
        const traceName = point.data && point.data.name ? point.data.name : 'curve';
        const xText = Number(point.x).toFixed(5);
        const yText = Number(point.y).toPrecision(7);
        const current = (graph.layout.annotations || []).slice();
        current.push({{
          x: point.x,
          y: point.y,
          xref: 'x',
          yref: 'y',
          text: `${{traceName}}<br>t=${{xText}} s<br>value=${{yText}}`,
          showarrow: true,
          arrowhead: 2,
          ax: 38,
          ay: -45,
          bgcolor: 'rgba(255,255,255,0.94)',
          bordercolor: '#64748b',
          borderwidth: 1,
          font: {{size: 11}}
        }});
        Plotly.relayout(graph, {{annotations: current}});
      }});
    }});
  </script>
</body>
</html>
"""


def write_html(
    trace_path: Path,
    output: Path,
    *,
    plotly_js: str = "inline",
) -> tuple[TraceData, OrderedDict[str, dict[str, Any]]]:
    trace_path = Path(trace_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    trace = load_trace(trace_path)
    charts = build_charts(trace)
    document = _build_html(
        trace_path,
        trace,
        charts,
        plotly_js=plotly_js,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return trace, charts


def main() -> int:
    args = parse_args()
    trace_path = args.log.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else trace_path.with_name(f"{trace_path.stem}_interactive.html")
    )
    trace, charts = write_html(
        trace_path,
        output,
        plotly_js=args.plotly_js,
    )
    print(f"interactive HTML: {output}")
    print(
        f"charts={len(charts)} policy={len(trace.policy)} "
        f"servo={len(trace.servo)} feedback={len(trace.feedback)} "
        f"dropped={int(trace.footer.get('dropped_records', 0))}"
    )
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
