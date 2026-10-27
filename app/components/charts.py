"""Small Plotly helpers so each page labels its axes the same way."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def _layout(fig, title: str, x_label: str, y_label: str, height: int) -> None:
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=height,
        margin=dict(l=16, r=16, t=56, b=16),
    )


def line_chart(frame: pd.DataFrame, x: str, y: str, title: str, x_label: str, y_label: str, markers: bool = False):
    fig = px.line(frame, x=x, y=y, markers=markers, color_discrete_sequence=["#4C78A8"])
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=frame[x].tolist())
    _layout(fig, title, x_label, y_label, 380)
    return fig


def bar_chart(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    x_label: str,
    y_label: str,
    horizontal: bool = False,
    category_order: list[str] | None = None,
):
    if horizontal:
        fig = px.bar(frame, x=y, y=x, orientation="h", color_discrete_sequence=["#4C78A8"])
        height = max(320, 28 * len(frame) + 80)
        _layout(fig, title, y_label, x_label, height)
        if category_order:
            fig.update_yaxes(categoryorder="array", categoryarray=category_order)
    else:
        fig = px.bar(frame, x=x, y=y, color_discrete_sequence=["#4C78A8"])
        _layout(fig, title, x_label, y_label, 380)
        if category_order:
            fig.update_xaxes(type="category", categoryorder="array", categoryarray=category_order)
    return fig


def _column_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "text"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    values = series.dropna().head(8).tolist()
    if not values:
        return "text"
    if all(hasattr(value, "year") and hasattr(value, "month") for value in values):
        return "date"
    if all(
        isinstance(value, str) and len(value) >= 7 and value[:4].isdigit() and value[4] == "-"
        for value in values
    ):
        return "date"
    return "text"


def chart_for_rows(rows: list[dict] | None):
    """Bar for one category and one number. Line for one date and one number."""
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    if frame.shape[1] < 2:
        return None
    kinds = {column: _column_kind(frame[column]) for column in frame.columns}
    numbers = [column for column, kind in kinds.items() if kind == "number"]
    dates = [column for column, kind in kinds.items() if kind == "date"]
    text = [column for column, kind in kinds.items() if kind == "text"]
    def label(name: str) -> str:
        return str(name).replace("_", " ")

    if len(numbers) == 1 and len(dates) == 1 and not text:
        return line_chart(
            frame,
            dates[0],
            numbers[0],
            f"{label(numbers[0])} by {label(dates[0])}",
            label(dates[0]),
            label(numbers[0]),
            markers=len(frame) <= 24,
        )
    if len(numbers) == 1 and len(text) == 1 and not dates:
        ordered = frame.sort_values(numbers[0], ascending=False)
        return bar_chart(
            ordered,
            text[0],
            numbers[0],
            f"{label(numbers[0])} by {label(text[0])}",
            label(text[0]),
            label(numbers[0]),
        )
    return None


def scatter_chart(frame: pd.DataFrame, x: str, y: str, color: str, title: str, x_label: str, y_label: str):
    fig = px.scatter(frame, x=x, y=y, color=color, opacity=0.65, color_discrete_sequence=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title="Segment",
        height=520,
        margin=dict(l=16, r=16, t=56, b=16),
    )
    fig.update_traces(marker=dict(size=7))
    return fig
