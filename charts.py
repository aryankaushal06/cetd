from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import DATASET_COLORS, FORECAST_COLORS, OnsetParams
from detection import detect_onset, detect_wet_spell, wet_spell_start


def doy_to_label(doy: float) -> str:
    try:
        return (
            pd.Timestamp("2001-01-01") + pd.Timedelta(days=int(doy) - 1)
        ).strftime("%d %b")
    except Exception:
        return str(doy)


def ts_xaxis_range(year: int, clip: bool, params: OnsetParams):
    if not clip:
        return None
    try:
        x0 = pd.Timestamp(
            year=year, month=params.start_month, day=params.start_day
        )
        x1 = pd.Timestamp(year=year, month=params.end_month, day=params.end_day)
        pad = pd.Timedelta(days=3)
        return [x0 - pad, x1 + pad]
    except Exception:
        return None


def ts_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rainfall (mm/day)",
        legend=dict(
            orientation="v",
            x=1.01,
            xanchor="left",
            y=1,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="white",
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            font=dict(color="white", size=13),
        ),
        hovermode="x unified",
        height=500,
        margin=dict(r=280),
        yaxis2=dict(
            overlaying="y",
            range=[0, 1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
    )
    return fig


def add_rain(fig, times, rain_vals, label, color, show_markers=True):
    fig.add_trace(
        go.Scatter(
            x=list(times),
            y=list(rain_vals),
            mode="lines+markers" if show_markers else "lines",
            name=label,
            line=dict(color=color, width=1.4),
            marker=dict(size=3, color=color) if show_markers else dict(),
            hovertemplate="%{y:.2f} mm<extra>" + label + "</extra>",
        )
    )


def add_markers(fig, df_ts, dataset_key, label_prefix, params: OnsetParams):
    """Add wet-spell start and onset as toggleable vertical line traces."""
    colors = DATASET_COLORS.get(
        dataset_key,
        FORECAST_COLORS.get(dataset_key, {"wet": "gray", "onset": "black"}),
    )
    times = df_ts["time"]
    rain = df_ts["rain"].values
    if not np.any(np.isfinite(rain)):
        return
    _, wi, si_ws = detect_wet_spell(times, rain, params)
    ws = wet_spell_start(times, rain, wi, params.wet_day_mm, si_ws)
    od, _ = detect_onset(times, rain, params)
    if ws is not None:
        fig.add_trace(
            go.Scatter(
                x=[ws, ws],
                y=[0, 1],
                mode="lines",
                name=f"{label_prefix} wet spell ({ws.strftime('%d %b')})",
                line=dict(color=colors["wet"], width=2, dash="dash"),
                yaxis="y2",
                hovertemplate=f"Wet spell start: {ws.date()}<extra></extra>",
                showlegend=True,
            )
        )
    if od is not None:
        fig.add_trace(
            go.Scatter(
                x=[od, od],
                y=[0, 1],
                mode="lines",
                name=f"{label_prefix} onset ({od.strftime('%d %b')})",
                line=dict(color=colors["onset"], width=2.5, dash="solid"),
                yaxis="y2",
                hovertemplate=f"Onset: {od.date()}<extra></extra>",
                showlegend=True,
            )
        )


def add_threshold(fig, wet_thresh: float, times):
    t0 = times.iloc[0] if hasattr(times, "iloc") else times[0]
    t1 = times.iloc[-1] if hasattr(times, "iloc") else times[-1]
    fig.add_trace(
        go.Scatter(
            x=[t0, t1],
            y=[wet_thresh, wet_thresh],
            mode="lines",
            name=f"Wet threshold ({wet_thresh:g} mm/day)",
            legendgroup="wet_threshold",
            showlegend=True,
            line=dict(color="gray", width=1.2, dash="dot"),
            hoverinfo="skip",
        )
    )


def station_map(meta: pd.DataFrame, selected: Optional[str]) -> go.Figure:
    eth_path = Path("data/ethiopia.geojson")
    with open(eth_path) as f:
        eth_geojson = json.load(f)

    lats = meta["lat"].tolist()
    lons = meta["lon"].tolist()
    names = meta.index.tolist()
    fig = go.Figure()

    geom = eth_geojson["features"][0]["geometry"]
    polys = (
        [geom["coordinates"]]
        if geom["type"] == "Polygon"
        else geom["coordinates"]
    )
    for poly in polys:
        rings = [poly] if not isinstance(poly[0][0], list) else poly
        for ring in rings:
            xs = [pt[0] for pt in ring] + [None]
            ys = [pt[1] for pt in ring] + [None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="black", width=2),
                    fill="toself",
                    fillcolor="white",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    unsel = [
        (n, lats[i], lons[i]) for i, n in enumerate(names) if n != selected
    ]
    if unsel:
        un, ul, ulo = zip(*unsel)
        fig.add_trace(
            go.Scatter(
                x=list(ulo),
                y=list(ul),
                text=list(un),
                customdata=list(un),
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=9,
                    color="red",
                    line=dict(color="white", width=0.6),
                ),
                textposition="top center",
                textfont=dict(size=9, color="#888888"),
                hovertemplate="<b>%{text}</b><br>%{y:.3f}°N, %{x:.3f}°E<extra></extra>",
                showlegend=False,
            )
        )

    if selected and selected in meta.index:
        si = names.index(selected)
        fig.add_trace(
            go.Scatter(
                x=[lons[si]],
                y=[lats[si]],
                text=[selected],
                customdata=[selected],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=11,
                    color="red",
                    line=dict(color="white", width=0.8),
                ),
                textposition="top center",
                textfont=dict(size=10, color="black", family="Arial Black"),
                hovertemplate=f"<b>{selected}</b><br>{lats[si]:.3f}°N, {lons[si]:.3f}°E<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis=dict(
            range=[32.5, 48.0],
            tickvals=[34, 36, 38, 40, 42, 44, 46],
            ticktext=[f"{v}°E" for v in [34, 36, 38, 40, 42, 44, 46]],
            tickfont=dict(size=11, color="black"),
            title=dict(text="Lon", font=dict(size=12, color="black")),
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        yaxis=dict(
            range=[3.2, 15.2],
            tickvals=[4, 6, 8, 10, 12, 14],
            ticktext=[f"{v}°N" for v in [4, 6, 8, 10, 12, 14]],
            tickfont=dict(size=11, color="black"),
            title=dict(text="Lat", font=dict(size=12, color="black")),
            showgrid=False,
            zeroline=False,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        margin=dict(l=60, r=20, t=40, b=50),
        height=500,
        title=dict(
            text="Rainfall Station Locations — click to select",
            font=dict(size=13, color="black"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        dragmode=False,
    )
    return fig


def add_ensemble_members(
    fig,
    times,
    rain_matrix: np.ndarray,
    label: str,
    color: str,
) -> None:
    """Faded individual member traces (hidden from legend)."""
    times_list = list(times)
    for i in range(rain_matrix.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=times_list,
                y=list(rain_matrix[:, i]),
                mode="lines",
                line=dict(color=color, width=0.7),
                opacity=0.18,
                showlegend=False,
                hoverinfo="skip",
            )
        )


def add_ensemble_fan(
    fig,
    times,
    rain_matrix: np.ndarray,
    label: str,
    color: str,
    fan_color: str,
) -> None:
    """P10–P90 shaded band + bold median line."""
    p10 = np.nanpercentile(rain_matrix, 10, axis=1)
    p90 = np.nanpercentile(rain_matrix, 90, axis=1)
    med = np.nanmedian(rain_matrix, axis=1)
    times_list = list(times)
    fig.add_trace(
        go.Scatter(
            x=times_list + times_list[::-1],
            y=list(p90) + list(p10[::-1]),
            fill="toself",
            fillcolor=fan_color,
            line=dict(color="rgba(255,255,255,0)"),
            name=f"{label} P10–P90",
            showlegend=True,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times_list,
            y=list(med),
            mode="lines",
            name=f"{label} median",
            line=dict(color=color, width=2.5),
            hovertemplate="%{y:.2f} mm<extra>" + label + " median</extra>",
        )
    )


def add_ensemble_onset_markers(
    fig,
    df_fc: pd.DataFrame,
    model_key: str,
    label_prefix: str,
    params: OnsetParams,
) -> None:
    """
    Run onset/wet-spell detection on every ensemble member.
    Draw faded per-member vertical lines and a bold median line.
    df_fc must have columns: valid_date, rain, member.
    """
    colors = FORECAST_COLORS.get(
        model_key, {"wet": "gray", "onset": "black"}
    )
    members = sorted(df_fc["member"].unique())
    ws_dates: list = []
    od_dates: list = []

    for m in members:
        dfm = df_fc[df_fc["member"] == m].sort_values("valid_date")
        times_m = pd.Series(dfm["valid_date"].values)
        rain_m = dfm["rain"].values.astype(float)
        if not np.any(np.isfinite(rain_m)):
            continue
        _, wi, si_ws = detect_wet_spell(times_m, rain_m, params)
        ws = wet_spell_start(times_m, rain_m, wi, params.wet_day_mm, si_ws)
        od, _ = detect_onset(times_m, rain_m, params)
        if ws is not None:
            ws_dates.append(ws)
            fig.add_trace(
                go.Scatter(
                    x=[ws, ws],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color=colors["wet"], width=1, dash="dash"),
                    yaxis="y2",
                    opacity=0.25,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if od is not None:
            od_dates.append(od)
            fig.add_trace(
                go.Scatter(
                    x=[od, od],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color=colors["onset"], width=1),
                    yaxis="y2",
                    opacity=0.25,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    if ws_dates:
        ws_med = pd.Timestamp(
            int(np.median([t.value for t in ws_dates]))
        )
        fig.add_trace(
            go.Scatter(
                x=[ws_med, ws_med],
                y=[0, 1],
                mode="lines",
                name=f"{label_prefix} wet spell ({ws_med.strftime('%d %b')})",
                line=dict(color=colors["wet"], width=2.5, dash="dash"),
                yaxis="y2",
                showlegend=True,
                hovertemplate=(
                    f"Wet spell median: {ws_med.date()}<extra></extra>"
                ),
            )
        )
    if od_dates:
        od_med = pd.Timestamp(
            int(np.median([t.value for t in od_dates]))
        )
        fig.add_trace(
            go.Scatter(
                x=[od_med, od_med],
                y=[0, 1],
                mode="lines",
                name=f"{label_prefix} onset ({od_med.strftime('%d %b')})",
                line=dict(color=colors["onset"], width=3),
                yaxis="y2",
                showlegend=True,
                hovertemplate=(
                    f"Onset median: {od_med.date()}<extra></extra>"
                ),
            )
        )
