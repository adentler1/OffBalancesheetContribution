#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Off-Balance-Sheet Contributions: standardized showcase with 8 archetypical graphs.

- OBSVizStyle: central style rules (fonts, footer, grid)
- Graphs: plotting archetypes (scatter, tornado, ranking, bucket, slope, multiline, small multiples, lorenz)
- Modules: prepare domain-specific data (players, teams, positions, seasons) and call Graphs
"""
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sqlite3
import datetime
from datetime import date
from typing import Iterable, Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.patches import FancyArrowPatch, Rectangle
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator, MaxNLocator


class OffBalanceShowcase:
    """
    Off-Balance-Sheet Contributions: standardized, centralized, factorized plotting toolkit.

    - OBSVizStyle: one place for fonts, colors, footer, spacings
    - Graphs: plotting archetypes (scatter, tornado, ranking, slope, bucket horizontal, grouped vertical,
              multiline, small multiples (scatter + lines), lorenz) built on shared helpers
    - Showcase: data-agnostic shell (choose scope + load externally; call mod_ functions elsewhere)
    """

    # ---------------------- Style Subclass ----------------------
    class OBSVizStyle:
        def __init__(self):
            # Typography
            self.title_fontsize = 15
            self.title_fontweight = "bold"
            self.subtitle_fontsize = 11
            self.label_fontsize = 12
            self.tick_fontsize = 10
            self.legend_fontsize = 10

            # Footer
            self.footer_text = "Off-Balance-Sheet Contributions"
            self.footer_fontsize = 12
            self.footer_fontweight = "bold"
            self.footer_lift = 0.15  # uniform bottom margin across all charts

            # Grid & palette
            self.grid_alpha = 0.25
            self.colors = {
                "top": "orange",
                "bottom": "steelblue",
                "line": "black",
                "fill": "lightgrey",
            }

        def apply_footer(self, fig: plt.Figure) -> None:
            # left bottom: date
            fig.text(
                0.01, 0.01,
                f"Created on {datetime.date.today().isoformat()}",
                ha="left", va="bottom",
                fontsize=self.footer_fontsize,
            )
            # right bottom: label
            fig.text(
                0.99, 0.01,
                self.footer_text,
                ha="right", va="bottom",
                fontsize=self.footer_fontsize,
                fontweight=self.footer_fontweight
            )

    # ---------------------- Graph Archetypes ----------------------
    class Graphs:
        """All drawing primitives. Uses standardized helpers for consistent look & feel."""

        def __init__(self, style: "OffBalanceShowcase.OBSVizStyle", parent: "OffBalanceShowcase"):
            self.s = style
            self._parent = parent  # access to defaults (e.g., figsize, palettes)

        # ---------- Shared helpers (centralize) ----------
        def _default_figsize(self) -> Tuple[float, float]:
            return getattr(self._parent, "figsize", (9, 6))

        def _new_fig_ax(self, title: str = "", subtitle: str = "", figsize: Tuple[float, float] | None = None) -> Tuple[plt.Figure, plt.Axes]:
            """Create a single-axes figure with standardized spacing & titles."""
            if figsize is None:
                figsize = self._default_figsize()
            fig, ax = plt.subplots(figsize=figsize)
            fig.subplots_adjust(bottom=self.s.footer_lift)
            if title:
                fig.suptitle(title, fontsize=self.s.title_fontsize, fontweight=self.s.title_fontweight)
            if subtitle:
                ax.set_title(subtitle, fontsize=self.s.subtitle_fontsize, pad=10)
            return fig, ax

        def _new_fig_axes(self, nrows: int, ncols: int,
                          title: str = "", subtitle: str = "",
                          sharex: bool | str = False, sharey: bool | str = False,
                          figsize: Tuple[float, float] | None = None) -> Tuple[plt.Figure, np.ndarray]:
            """Create a multi-axes figure with standardized spacing & titles."""
            if figsize is None:
                # scale default size by grid
                base_w, base_h = self._default_figsize()
                figsize = (max(base_w, 4.0) * ncols * 0.6, max(base_h, 3.8) * nrows * 0.6)
            fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
            fig.subplots_adjust(bottom=self.s.footer_lift)
            if title:
                fig.suptitle(title, fontsize=self.s.title_fontsize, fontweight=self.s.title_fontweight)
            if subtitle:
                fig.text(0.5, 0.93, subtitle, ha="center", fontsize=self.s.subtitle_fontsize)
            return fig, axes

        def _set_labels(self, ax: plt.Axes, xlabel: Optional[str] = None, ylabel: Optional[str] = None) -> None:
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=self.s.label_fontsize)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=self.s.label_fontsize)

        def _set_grid(self, ax: plt.Axes, axis: str | None = None) -> None:
            if axis in (None, "both", True):
                ax.grid(True, alpha=self.s.grid_alpha)
            elif axis in ("x", "y"):
                ax.grid(True, axis=axis, alpha=self.s.grid_alpha)

        def _hide_spines(self, ax: plt.Axes, which: Iterable[str] = ("top", "right")) -> None:
            for sp in which:
                if sp in ax.spines:
                    ax.spines[sp].set_visible(False)

        def _integer_xticks(self, ax: plt.Axes, x_values: Optional[Iterable[float]] = None) -> None:
            """Force integer ticks at given x-values (or autodetect ints)."""
            if x_values is not None:
                ints = sorted({int(round(v)) for v in x_values if np.isfinite(v)})
            else:
                # fallback: try to infer from current limits (less strict)
                ints = None
            if ints:
                ax.xaxis.set_major_locator(FixedLocator(ints))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        def _legend_if(self, ax: plt.Axes, labels: List[str]) -> None:
            uniq = [l for l in dict.fromkeys(labels) if l]
            if len(uniq) > 1:
                ncol = max(1, (len(uniq) + 4) // 5)
                ax.legend(frameon=False, ncol=ncol, fontsize=self.s.legend_fontsize)

        # ---------- Scatter (FTE vs Contribution, with highlights & regression) ----------
        def scatter_fte(
            self,
            sources,                     # dict OR list[dict] of {"x","y","labels","color","name","marker","size","alpha"}
            title: str = "",
            subtitle: str = "",
            xlabel: str | None = None,
            ylabel: str | None = None,
            show_legend: bool = True,
            figsize: Tuple[float, float] | None = None,
            highlights: List[Dict] | None = None,   # [{"x","y","text","side","color"}]
            *,
            hline: str | None = "zero",             # None | "zero" | "mean"
            regline: bool = False,                  # draw constant-trend OLS line
            reg_ci: str | None = None               # None | "mean" | "prediction"
        ):
            # Normalize sources
            if isinstance(sources, dict):
                sources = [sources]

            fig, ax = self._new_fig_ax(title, subtitle, figsize)

            # Build set of highlighted coordinates
            highlight_pts = set()
            if highlights:
                for h in highlights:
                    try:
                        hx = float(h["x"]); hy = float(h["y"])
                        if np.isfinite(hx) and np.isfinite(hy):
                            highlight_pts.add((hx, hy))
                    except Exception:
                        continue

            # Plot sources (skip highlighted duplicates)
            legend_labels = []
            for src in sources:
                x = np.asarray(src.get("x", []), dtype=float)
                y = np.asarray(src.get("y", []), dtype=float)
                if x.size == 0 or y.size == 0:
                    continue
                mask = np.isfinite(x) & np.isfinite(y)
                x, y = x[mask], y[mask]
                if highlight_pts:
                    keep = np.array([(float(xi), float(yi)) not in highlight_pts for xi, yi in zip(x, y)], dtype=bool)
                    x, y = x[keep], y[keep]
                if x.size == 0:
                    continue

                ax.scatter(
                    x, y,
                    s=float(src.get("size", 40.0)),
                    alpha=float(src.get("alpha", 0.70)),
                    color=src.get("color", None),   # let MPL cycle if None
                    marker=src.get("marker", "o"),
                    label=src.get("name", None),
                    zorder=2
                )
                if src.get("name"):
                    legend_labels.append(src["name"])

            # Highlights
            if highlights:
                side2ha_va_offset = {
                    "left":   ("right", "center", (-8,  0)),
                    "right":  ("left",  "center", ( 8,  0)),
                    "top":    ("center","bottom",( 0,  8)),
                    "bottom": ("center","top",   ( 0, -8)),
                }
                for h in highlights:
                    try:
                        hx = float(h["x"]); hy = float(h["y"])
                        txt  = str(h.get("text", ""))
                        side = str(h.get("side", "right")).lower()
                        color = h.get("color", "red")
                        if not (np.isfinite(hx) and np.isfinite(hy)):
                            continue
                        ha, va, (dx, dy) = side2ha_va_offset.get(side, ("left","center",(8,0)))
                        ax.scatter([hx], [hy], s=90, color=color, zorder=4)
                        ax.annotate(
                            txt, xy=(hx, hy), xycoords="data",
                            xytext=(dx, dy), textcoords="offset points",
                            ha=ha, va=va, fontsize=self.s.tick_fontsize, color=color,
                            zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85)
                        )
                    except Exception:
                        continue

            # Horizontal lines
            if hline == "zero":
                ax.axhline(0, color=self.s.colors.get("line", "black"), lw=0.8, zorder=1)
            elif hline == "mean":
                all_y = []
                for src in sources:
                    xi = np.asarray(src.get("x", []), dtype=float)
                    yi = np.asarray(src.get("y", []), dtype=float)
                    m = np.isfinite(xi) & np.isfinite(yi)
                    yi = yi[m]
                    if highlight_pts and yi.size:
                        keep = np.array([(float(a), float(b)) not in highlight_pts for a, b in zip(xi[m], yi)], dtype=bool)
                        yi = yi[keep]
                    if yi.size:
                        all_y.append(yi)
                if all_y:
                    m = float(np.nanmean(np.concatenate(all_y)))
                    if np.isfinite(m):
                        ax.axhline(m, color=self.s.colors.get("line", "black"), lw=0.8, linestyle="--", alpha=0.8, zorder=1)

            # Regression (closed-form OLS)
            if regline:
                Xs, Ys = [], []
                for src in sources:
                    xi = np.asarray(src.get("x", []), dtype=float)
                    yi = np.asarray(src.get("y", []), dtype=float)
                    m = np.isfinite(xi) & np.isfinite(yi)
                    xi, yi = xi[m], yi[m]
                    if highlight_pts and xi.size:
                        keep = np.array([(float(a), float(b)) not in highlight_pts for a, b in zip(xi, yi)], dtype=bool)
                        xi, yi = xi[keep], yi[keep]
                    if xi.size:
                        Xs.append(xi); Ys.append(yi)

                if Xs:
                    X = np.concatenate(Xs)
                    Y = np.concatenate(Ys)
                    m = np.isfinite(X) & np.isfinite(Y)
                    X, Y = X[m], Y[m]
                    if X.size >= 2:
                        xbar = X.mean(); ybar = Y.mean()
                        Sxx = np.sum((X - xbar) ** 2)
                        if Sxx > 1e-12:
                            Sxy = np.sum((X - xbar) * (Y - ybar))
                            b1 = Sxy / Sxx
                            b0 = ybar - b1 * xbar
                            xs = np.linspace(np.nanmin(X), np.nanmax(X), 200)
                            ys = b0 + b1 * xs
                            ax.plot(xs, ys, color=self.s.colors.get("line", "black"), lw=1.4, alpha=0.9, zorder=3)

                            if reg_ci in ("mean", "prediction") and X.size >= 3:
                                yhat = b0 + b1 * X
                                rss  = np.sum((Y - yhat) ** 2)
                                dof  = max(1, X.size - 2)
                                s    = np.sqrt(rss / dof)
                                Sxx  = max(Sxx, 1e-12)
                                if reg_ci == "mean":
                                    se = s * np.sqrt(1.0 / X.size + (xs - xbar) ** 2 / Sxx)
                                else:
                                    se = s * np.sqrt(1.0 + 1.0 / X.size + (xs - xbar) ** 2 / Sxx)
                                tcrit = 1.96 if X.size >= 30 else 2.0
                                ax.fill_between(xs, ys - tcrit * se, ys + tcrit * se,
                                                color=self.s.colors.get("line", "black"),
                                                alpha=0.08, edgecolor="none", zorder=2.5)

            self._set_labels(ax, xlabel, ylabel)
            self._set_grid(ax, axis=None)
            self._legend_if(ax, legend_labels) if show_legend else None
            self._hide_spines(ax, ("top", "right"))

            self.s.apply_footer(fig)
            return fig

        # ---------- Tornado ----------
        def tornado(self, items, title: str = "", subtitle: str = "", xlabel: Optional[str] = None,
                    symmetric: bool = True, figsize: Tuple[float, float] | None = None, show_values: bool = False):
            """
            items: list of dicts {"name","value","color"}
            If show_values=True, prints numeric at bar end (right for +, left for -) with anti-overlap.
            """
            names  = [str(it["name"]) for it in items]
            vals   = [float(it["value"]) for it in items]
            colors = [it.get("color", self.s.colors.get("fill", "lightgrey")) for it in items]

            fig, ax = self._new_fig_ax(title, subtitle, figsize)

            ax.barh(names, vals, color=colors, zorder=2)
            ax.invert_yaxis()
            ax.axvline(0, color=self.s.colors.get("line", "black"), lw=0.8, zorder=1)

            # x-limits
            if symmetric:
                lim_data = max(abs(min(vals) if vals else 0.0), abs(max(vals) if vals else 0.0))
                lim_data = lim_data if lim_data > 0 else 1.0
                span = 2.0 * lim_data
                extra = 0.06 * span if show_values else 0.0
                lim = lim_data + extra
                ax.set_xlim(-lim, lim)
            else:
                xmin = min(0.0, min(vals) if vals else 0.0)
                xmax = max(0.0, max(vals) if vals else 0.0)
                span = (xmax - xmin) if xmax > xmin else 1.0
                pad  = 0.06 * span if show_values else 0.02 * span
                ax.set_xlim(xmin - pad, xmax + pad)

            # keep numeric labels away from center labels
            xmin, xmax = ax.get_xlim()
            span = xmax - xmin
            min_from_zero = 0.03 * span
            edge_pad      = 0.01 * span

            # centered names at x=0
            for i, it in enumerate(items):
                ax.text(0, i, str(it["name"]),
                        ha="center", va="center",
                        fontsize=self.s.tick_fontsize,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.90),
                        zorder=4)

            # values outside bars
            if show_values:
                for i, it in enumerate(items):
                    v = float(it["value"])
                    lab = f"{v:.2f}"
                    if v >= 0:
                        x_target = max(v + edge_pad, min_from_zero); ha = "left"
                    else:
                        x_target = min(v - edge_pad, -min_from_zero); ha = "right"
                    ax.text(x_target, i, lab,
                            ha=ha, va="center", fontsize=self.s.tick_fontsize, color="black",
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.90),
                            zorder=5, clip_on=False)

            ax.tick_params(axis="y", left=False, labelleft=False)
            self._hide_spines(ax, ("left", "right", "top"))
            self._set_labels(ax, xlabel=xlabel, ylabel=None)
            self._set_grid(ax, axis="x")

            self.s.apply_footer(fig)
            return fig

        # ---------- Ranking ----------
        def ranking(self, categories, values, title: str = "", subtitle: str = "",
                    figsize: Tuple[float, float] | None = None):
            fig, ax = self._new_fig_ax(title, subtitle, figsize)
            ax.barh(categories, values, color="steelblue", zorder=2)
            ax.axvline(0, color=self.s.colors.get("line", "black"), lw=0.8, zorder=1)
            self._set_labels(ax, xlabel="Off-Balance-Sheet Contribution", ylabel=None)
            ax.tick_params(axis="y", labelsize=self.s.tick_fontsize)
            self._set_grid(ax, axis="x")
            self._hide_spines(ax, ("top", "right"))
            self.s.apply_footer(fig)
            return fig

        # ---------- Slope (arrows) ----------
        def slope(
            self,
            items,                                 # list[dict] OR dict-of-arrays
            title: str = "",
            subtitle: str = "",
            *,
            sort_start_ascending: bool = True,     # sort by start_value
            xlabel: str = "Off-Balance-Sheet Contribution",
            figsize: Tuple[float, float] | None = None,
            y_pad_bottom: float = 0.15,
            y_pad_top: float = 0.05,
        ):
            """
            Minimal slope chart:
              items rows = {start_value, final_value, label, flag_to_write_value=False, color=None, linewidth=1.8}
              - Sort by start_value
              - Label is placed left (inc) or right (dec) of START value
              - Δ shown near final (sign chooses side), 3 significant digits
              - Arrow from start→final, dot at start
            """
            try:
                default_color = mpl.rcParams["axes.prop_cycle"].by_key()["color"][0]
            except Exception:
                default_color = "tab:blue"

            # Normalize to list[dict]
            def _broadcast(val, n):
                if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                    if len(val) == n:
                        return list(val)
                return [val] * n

            norm = []
            if isinstance(items, dict):
                starts = np.asarray(items.get("start_value", []), dtype=float)
                finals = np.asarray(items.get("final_value", []), dtype=float)
                labels = list(items.get("label", []))
                n = min(len(starts), len(finals), len(labels))
                flags = _broadcast(items.get("flag_to_write_value", False), n)
                colors = _broadcast(items.get("color", default_color), n)
                lws    = _broadcast(items.get("linewidth", 1.8), n)
                for i in range(n):
                    sv, fv = float(starts[i]), float(finals[i])
                    if np.isfinite(sv) and np.isfinite(fv):
                        norm.append({
                            "start_value": sv, "final_value": fv, "label": str(labels[i]),
                            "flag_to_write_value": bool(flags[i]),
                            "color": colors[i] if colors[i] is not None else default_color,
                            "linewidth": float(lws[i]),
                        })
            else:
                for it in items:
                    sv = float(it.get("start_value", np.nan))
                    fv = float(it.get("final_value", np.nan))
                    if np.isfinite(sv) and np.isfinite(fv):
                        norm.append({
                            "start_value": sv, "final_value": fv, "label": str(it.get("label", "")),
                            "flag_to_write_value": bool(it.get("flag_to_write_value", False)),
                            "color": it.get("color", default_color) or default_color,
                            "linewidth": float(it.get("linewidth", 1.8)),
                        })
            if not norm:
                raise ValueError("No valid rows for slope().")

            norm.sort(key=lambda d: d["start_value"], reverse=not sort_start_ascending)

            fig, ax = self._new_fig_ax(title, subtitle, figsize)
            n = len(norm)

            starts = np.array([d["start_value"] for d in norm], dtype=float)
            finals = np.array([d["final_value"] for d in norm], dtype=float)
            X = np.r_[starts, finals]
            x_min = float(np.nanmin(X)); x_max = float(np.nanmax(X))
            if x_max <= x_min: x_max = x_min + 1.0
            span = x_max - x_min

            needs_left  = any(d["final_value"] > d["start_value"] for d in norm) \
                          or any(d["flag_to_write_value"] and (d["final_value"] - d["start_value"] < 0) for d in norm)
            needs_right = any(d["final_value"] < d["start_value"] for d in norm) \
                          or any(d["flag_to_write_value"] and (d["final_value"] - d["start_value"] > 0) for d in norm)

            left_pad  = (0.06 + (0.14 if needs_left  else 0.0)) * span
            right_pad = (0.06 + (0.14 if needs_right else 0.0)) * span
            ax.set_xlim(x_min - left_pad, x_max + right_pad)
            ax.set_ylim(-0.5 - float(y_pad_bottom), n - 0.5 + float(y_pad_top))
            text_dx = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])

            for yi, d in enumerate(norm):
                b = d["start_value"]; a = d["final_value"]
                delta = a - b
                color = d["color"]; lw = d["linewidth"]

                arr = FancyArrowPatch(
                    posA=(b, yi), posB=(a, yi),
                    arrowstyle="-|>", mutation_scale=12.0,
                    lw=lw, color=color, zorder=2, shrinkA=0.0, shrinkB=0.0
                )
                ax.add_patch(arr)
                ax.scatter([b], [yi], s=28, facecolor="white", edgecolor=color, linewidth=1.2, zorder=3)

                if a > b:
                    x_label, ha = b - text_dx, "right"
                else:
                    x_label, ha = b + text_dx, "left"

                ax.text(x_label, yi, d["label"],
                        ha=ha, va="center", fontsize=self.s.tick_fontsize,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.90),
                        zorder=4, clip_on=False)

                if d["flag_to_write_value"]:
                    try:
                        delta_txt = f"{delta:+.3g}"
                    except Exception:
                        delta_txt = f"{delta:+.2f}"
                    if delta > 0:
                        x_d, ha_d = a + text_dx, "left"
                    elif delta < 0:
                        x_d, ha_d = a - text_dx, "right"
                    else:
                        x_d, ha_d = a + text_dx, "left"
                    ax.text(x_d, yi, delta_txt,
                            ha=ha_d, va="center", fontsize=self.s.tick_fontsize, color=color,
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
                            zorder=4, clip_on=False)

            ax.set_yticks([])
            self._set_labels(ax, xlabel=xlabel, ylabel=None)
            self._set_grid(ax, axis="x")
            self._hide_spines(ax, ("top", "right", "left"))
            self.s.apply_footer(fig)
            return fig

        # ---------- Horizontal Buckets (tornado-style) ----------
        def bucket_horizontal(
            self,
            items,                           # list[{"name": str, "values": array_like, "color": Optional[str]}]
            title: str = "",
            subtitle: str = "",
            xlabel: str | None = "Off-Balance-Sheet Contribution",
            *,
            symmetric: bool = True,          # force symmetric x-limits around 0
            showfliers: bool = True,
            figsize: Tuple[float, float] | None = None,
            box_width: float = 0.55,         # thin box thickness
            band_scale: float = 1.25,        # thick IQR band height multiplier
            band_alpha: float = 0.90,        # opacity of the IQR band (under the centered label)
        ):
            names, data, colors = [], [], []
            default_face = self.s.colors.get("fill", "lightgrey")

            for it in items:
                nm = str(it.get("name", ""))
                vals = np.asarray(list(it.get("values", [])), dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                names.append(nm)
                data.append(vals)
                colors.append(it.get("color", default_face))
            if not data:
                raise ValueError("bucket_horizontal: no non-empty series in 'items'.")

            fig, ax = self._new_fig_ax(title, subtitle, figsize)

            # x-limits
            minv = float(min(np.nanmin(d) for d in data))
            maxv = float(max(np.nanmax(d) for d in data))
            if symmetric:
                lim = max(abs(minv), abs(maxv))
                lim = 1.0 if not np.isfinite(lim) or lim == 0 else lim
                ax.set_xlim(-lim * 1.05, lim * 1.05)
            else:
                span = maxv - minv if maxv > minv else 1.0
                pad  = 0.05 * span
                ax.set_xlim(minv - pad, maxv + pad)

            pos = np.arange(len(data))
            bp = ax.boxplot(
                data, vert=False, positions=pos, widths=float(box_width),
                patch_artist=True, showfliers=bool(showfliers), zorder=2,
            )

            # style thin boxes
            whisk_pairs = [bp["whiskers"][i:i+2] for i in range(0, len(bp["whiskers"]), 2)]
            cap_pairs   = [bp["caps"][i:i+2]     for i in range(0, len(bp["caps"]),     2)]
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor(colors[i]); box.set_alpha(0.50); box.set_edgecolor("black")
            for med in bp["medians"]:
                med.set_color("black"); med.set_linewidth(1.4)
            for pair in whisk_pairs:
                for w in pair: w.set_color("black"); w.set_linewidth(1.0)
            for pair in cap_pairs:
                for c in pair: c.set_color("black"); c.set_linewidth(1.0)

            # thick IQR band (under label)
            xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
            band_h = min(0.95, float(box_width) * float(band_scale))
            for y, vals, col in zip(pos, data, colors):
                if vals.size < 2:
                    med = float(np.nanmedian(vals))
                    tiny = 0.01 * xspan
                    rect = Rectangle((med - tiny/2, y - band_h/2), tiny, band_h,
                                     facecolor=col, edgecolor="none", alpha=band_alpha, zorder=3)
                    ax.add_patch(rect)
                else:
                    q1, q3 = np.nanpercentile(vals, [25, 75])
                    if not np.isfinite(q1) or not np.isfinite(q3) or q3 <= q1:
                        med = float(np.nanmedian(vals))
                        tiny = 0.01 * xspan
                        rect = Rectangle((med - tiny/2, y - band_h/2), tiny, band_h,
                                         facecolor=col, edgecolor="none", alpha=band_alpha, zorder=3)
                    else:
                        rect = Rectangle((q1, y - band_h/2), q3 - q1, band_h,
                                         facecolor=col, edgecolor="none", alpha=band_alpha, zorder=3)
                    ax.add_patch(rect)

            ax.axvline(0, color=self.s.colors.get("line", "black"), lw=0.8, zorder=1)
            ax.set_yticks([])
            for y, nm in zip(pos, names):
                ax.text(0, y, nm, ha="center", va="center",
                        fontsize=self.s.tick_fontsize,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.90),
                        zorder=4)

            ax.invert_yaxis()
            self._set_labels(ax, xlabel=xlabel, ylabel=None)
            self._set_grid(ax, axis="x")
            self._hide_spines(ax, ("left", "right", "top"))
            self.s.apply_footer(fig)
            return fig

        # ---------- Grouped Vertical Buckets ----------
        def bucket_grouped_vertical(
            self,
            data_map,                          # dict[str top_group] -> dict[str sub_group] -> array_like
            *,
            group_order: List[str],
            sub_order: List[str],
            title: str = "",
            subtitle: str = "",
            ylabel: str = "Off-Balance-Sheet Contribution",
            show_legend: bool = True,
            figsize: Tuple[float, float] | None = None,
        ):
            nG = len(group_order)
            nS = len(sub_order)
            if nG == 0 or nS == 0:
                raise ValueError("bucket_grouped_vertical: empty group_order or sub_order.")

            palette = getattr(self._parent, "default_colors", [
                "tab:blue","tab:orange","tab:green","tab:red","tab:purple",
                "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"
            ])
            col = {sg: palette[i % len(palette)] for i, sg in enumerate(sub_order)}

            gap = 1.0
            width = 0.65
            positions = []
            box_data  = []
            box_color = []
            centers   = []

            x = 1.0
            for g in group_order:
                centers.append(x + (nS - 1) * 0.5)
                for sg in sub_order:
                    vals = np.asarray(list((data_map.get(g, {}) or {}).get(sg, [])), dtype=float)
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        vals = np.array([np.nan], dtype=float)
                    positions.append(x)
                    box_data.append(vals)
                    box_color.append(col[sg])
                    x += 1.0
                x += gap

            fig, ax = self._new_fig_ax(title, subtitle, figsize)

            bp = ax.boxplot(
                box_data, positions=positions, widths=width,
                patch_artist=True, showfliers=True, vert=True, zorder=2
            )
            for i, box in enumerate(bp["boxes"]):
                box.set_facecolor(box_color[i]); box.set_alpha(0.70); box.set_edgecolor("black")
            for k in ("medians", "whiskers", "caps"):
                for el in bp[k]:
                    el.set_color("black")

            ax.set_xticks(centers)
            ax.set_xticklabels(group_order, fontsize=self.s.tick_fontsize)
            self._set_labels(ax, xlabel=None, ylabel=ylabel)
            self._set_grid(ax, axis="y")
            self._hide_spines(ax, ("top", "right"))

            if show_legend and nS > 1:
                handles = [mpatches.Patch(facecolor=col[sg], edgecolor="black", label=sg) for sg in sub_order]
                ax.legend(handles=handles, frameon=False, fontsize=self.s.legend_fontsize,
                          ncol=min(nS, 4), loc="upper right")

            self.s.apply_footer(fig)
            return fig

        # ---------- Multiline ----------
        def multiline(self, x, ys, labels, title: str = "", subtitle: str = "",
                      figsize: Tuple[float, float] | None = None, integer_xticks: bool = False):
            x = np.asarray(list(x), dtype=float)
            fig, ax = self._new_fig_ax(title, subtitle, figsize)
            for y, l in zip(ys, labels):
                y = np.asarray(list(y), dtype=float)
                ax.plot(x, y, marker="o", label=l)
            self._set_labels(ax, xlabel="Season / Age", ylabel="Off-Balance-Sheet Contribution")
            ax.legend(frameon=False, fontsize=self.s.legend_fontsize)
            self._set_grid(ax, axis=None)
            if integer_xticks:
                self._integer_xticks(ax, x)
            self.s.apply_footer(fig)
            return fig

        # ---------- Small Multiples (time lines with faint context) ----------
        def small_multiples_lines(
            self,
            x,                     # 1D sequence of times (e.g., seasons)
            panels,                # list[{"title": str, "y": array_like, "color": Optional[str]}]
            *,
            layout: Tuple[int, int] | None = None,  # if None → pick by N: 2→(2,1), 3→(3,1), 4→(2,2), 6→(2,3), 8→(2,4)
            title: str = "",
            subtitle: str = "",
            figsize: Tuple[float, float] | None = None,
        ):
            x = np.asarray(list(x), dtype=float)
            N = len(panels)
            allowed = {2: (2, 1), 3: (3, 1), 4: (2, 2), 6: (2, 3), 8: (2, 4)}
            if layout is None:
                if N not in allowed:
                    raise ValueError(f"small_multiples_lines: N={N} not supported; allowed {sorted(allowed)}.")
                rows, cols = allowed[N]
            else:
                rows, cols = layout

            if figsize is None:
                figsize = (max(10, 4 * cols), max(4.0, 3.2 * rows))

            Ys = []; titles = []; colors = []
            for p in panels:
                y = np.asarray(list(p.get("y", [])), dtype=float)
                Ys.append(y); titles.append(str(p.get("title", ""))); colors.append(p.get("color", None))

            allY = np.concatenate([np.asarray(y, dtype=float) for y in Ys if len(y) > 0]) if Ys else np.array([])
            ymin = float(np.nanmin(allY)) if allY.size else 0.0
            ymax = float(np.nanmax(allY)) if allY.size else 1.0
            if not np.isfinite(ymin): ymin = 0.0
            if not np.isfinite(ymax): ymax = 1.0
            if ymax <= ymin: ymax = ymin + 1.0

            fig, axes = self._new_fig_axes(rows, cols, title=title, subtitle=subtitle, sharey=True, figsize=figsize)
            axes = np.array(axes).ravel()

            # faint background lines for context
            for ax in axes[:N]:
                for y in Ys:
                    ax.plot(x, y, alpha=0.15, linewidth=1.2, color="gray")

            # focal subject
            for ax, y, t, c in zip(axes[:N], Ys, titles, colors):
                ax.plot(x, y, marker="o", linewidth=2.2, alpha=0.95, color=c)
                ax.set_title(t, fontsize=self.s.tick_fontsize + 1)
                ax.set_ylim(ymin, ymax)
                self._set_grid(ax, axis=None)
                self._hide_spines(ax, ("top", "right"))
                # integer ticks if x looks like integers (seasons)
                self._integer_xticks(ax, x)

            # hide unused axes (if any)
            for ax in axes[N:]:
                ax.set_visible(False)

            self.s.apply_footer(fig)
            return fig

        # ---------- Small Multiples (scatter) ----------
        def small_multiples(self, dfs, titles, title: str = "", subtitle: str = "",
                            figsize: Tuple[float, float] | None = (12, 8)):
            n = len(dfs)
            if n <= 0:
                raise ValueError("small_multiples: need at least one panel.")
            fig, axes = self._new_fig_axes(1, n, title=title, subtitle=subtitle, sharey=True, figsize=figsize)
            axes = np.array(axes).ravel()
            for ax, df, t in zip(axes, dfs, titles):
                ax.scatter(df["x"], df["y"], s=30, alpha=0.7)
                ax.set_title(str(t), fontsize=self.s.subtitle_fontsize)
                self._set_grid(ax, axis=None)
                self._hide_spines(ax, ("top", "right"))
            self.s.apply_footer(fig)
            return fig

        # ---------- Lorenz ----------
        def lorenz(self, values, title: str = "", subtitle: str = "", figsize: Tuple[float, float] | None = (6, 6)):
            vals = np.asarray(values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0 or np.nansum(vals) == 0:
                # draw equality line only
                x = np.array([0.0, 1.0]); y = np.array([0.0, 1.0])
                fig, ax = self._new_fig_ax(title, subtitle, figsize)
                ax.plot(x, y, lw=2)
                ax.set_xlabel("Cumulative share of players", fontsize=self.s.label_fontsize)
                ax.set_ylabel("Cumulative share of contributions", fontsize=self.s.label_fontsize)
                self._set_grid(ax, axis=None)
                self.s.apply_footer(fig)
                return fig

            vals = np.sort(vals)
            cumvals = np.cumsum(vals) / np.sum(vals)
            cumplayers = np.linspace(0, 1, len(vals))
            fig, ax = self._new_fig_ax(title, subtitle, figsize)
            ax.plot(cumplayers, cumvals, lw=2)
            ax.plot([0, 1], [0, 1], color=self.s.colors.get("line", "black"), linestyle="--")
            self._set_labels(ax, xlabel="Cumulative share of players", ylabel="Cumulative share of contributions")
            self._set_grid(ax, axis=None)
            self.s.apply_footer(fig)
            return fig

    # ---------------------- Showcase (data-agnostic shell) ----------------------
    def __init__(self, *,
                 db_path: str = "analysis_results.db",
                 estimator: str = "ols",
                 ridge_alpha: float | None = None,
                 lasso_alpha: float | None = None,
                 save_plot: bool = True,
                 show_plot: bool = False,
                 figsize: Tuple[float, float] = (9, 6)):
        """
        Data-agnostic initialization:
          - sets estimator & plotting defaults
          - DOES NOT set country/league/season
          - DOES NOT load data

        Choose scope and load data later via .load(...), or put a DataFrame
        directly via .set_players(df, context_label=...).
        """
        self.db_path = db_path
        self.estimator = estimator.strip().lower()
        self.ridge_alpha = ridge_alpha
        self.lasso_alpha = lasso_alpha
        self.save_plot = save_plot
        self.show_plot = show_plot

        # plotting defaults (centralized)
        self.figsize = figsize
        self.default_colors = [
            "tab:blue","tab:orange","tab:green","tab:red","tab:purple",
            "tab:brown","tab:pink","tab:gray","tab:olive","tab:cyan"
        ]
        self._imp_col = "impact"  # resolved in _resolve_imp_col(df)
        self.style = OffBalanceShowcase.OBSVizStyle()
        # pass parent to graphs for central defaults/palettes
        self.graphs = OffBalanceShowcase.Graphs(self.style, parent=self)

        # data holders & context (managed externally)
        self.players_df = pd.DataFrame()
        self.players_df_comp = pd.DataFrame()
        self.country: str | None = None
        self.league: str | None = None
        self.season: int | None = None
        self._context_label: str = ""   # optional human label for titles

    # ---------- Optional helpers you may already have elsewhere ----------
    def _fmt_scope_title(self, fallback: str = "Scope") -> str:
        """Build a compact scope label for titles (Country / League / Season)."""
        parts = []
        if self.country: parts.append(self.country)
        if self.league:  parts.append(self.league)
        if self.season is not None: parts.append(str(self.season))
        return " / ".join(parts) if parts else fallback

    def set_context(self, *, country: str | None = None,
                    league: str | None = None,
                    season: int | None = None,
                    label: str | None = None) -> None:
        """Set context for titles and for methods that require season anchoring."""
        if country is not None: self.country = country
        if league  is not None: self.league  = league
        if season  is not None: self.season  = int(season)
        if label   is not None: self._context_label = str(label)
    
    def fetch_scope(self, *,
                    country: str | None = None,
                    league: str | None = None,
                    season: int | None = None,
                    extra_where: str | None = None,
                    params: list | tuple | None = None) -> pd.DataFrame:
        """
        Return a DataFrame from analysis_results by optional filters.
        Does NOT mutate object state.
        """
        where = []
        qparams: list = []
        if country is not None:
            where.append("country = ?"); qparams.append(country)
        if league is not None:
            where.append("league = ?");  qparams.append(league)
        if season is not None:
            where.append("season = ?");  qparams.append(int(season))
        if extra_where:
            where.append(f"({extra_where})")
            if params:
                qparams.extend(list(params))
    
        sql = "SELECT * FROM analysis_results"
        if where:
            sql += " WHERE " + " AND ".join(where)
    
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(sql, conn, params=qparams)
        return df
    
    def _resolve_imp_col(self, df: pd.DataFrame) -> str:
        """
        Support both older 'impact' naming and newer 'contribution_*' naming.
        """
        cols = set(df.columns)
    
        def _alpha_key(prefix: str, a: float) -> str:
            # build both legacy and new key to test
            a_str = str(float(a)).replace('.', '_')
            legacy = f"{prefix}_impact_alpha_{a_str}"
            modern = f"{prefix}_contribution_alpha_{a_str}"
            return modern if modern in cols else legacy
    
        if self.estimator == "ols":
            if "contribution_ols" in cols:
                return "contribution_ols"
            if "impact" in cols:
                return "impact"
            raise ValueError("Neither 'contribution_ols' nor 'impact' found in DB.")
    
        if self.estimator == "ridge":
            if self.ridge_alpha is None:
                # pick first ridge column (new or old)
                cands = [c for c in df.columns if c.startswith("ridge_contribution_alpha_")] \
                        or [c for c in df.columns if c.startswith("ridge_impact_alpha_")]
                if not cands:
                    raise ValueError("No ridge contribution/impact columns found in DB.")
                return cands[0]
            key = _alpha_key("ridge", float(self.ridge_alpha))
            if key not in cols:
                raise ValueError(f"Ridge column not found for alpha={self.ridge_alpha}: tried '{key}'.")
            return key
    
        if self.estimator == "lasso":
            if self.lasso_alpha is None:
                cands = [c for c in df.columns if c.startswith("lasso_contribution_alpha_")] \
                        or [c for c in df.columns if c.startswith("lasso_impact_alpha_")]
                if not cands:
                    raise ValueError("No lasso contribution/impact columns found in DB.")
                return cands[0]
            key = _alpha_key("lasso", float(self.lasso_alpha))
            if key not in cols:
                raise ValueError(f"Lasso column not found for alpha={self.lasso_alpha}: tried '{key}'.")
            return key
    
        raise ValueError(f"Unknown estimator '{self.estimator}'.")
    

    def load(self, *,
             country: str | None = None,
             league: str | None = None,
             season: int | None = None,
             extra_where: str | None = None,
             params: list | tuple | None = None,
             context_label: str | None = None) -> None:
        """
        Load players_df from analysis_results using optional filters and
        set context for titles and season-anchored helpers.
    
        If you pass country/league/season here, they are stored on the object.
        """
        # Update context first (needed by transfer-window helpers)
        self.set_context(country=country, league=league, season=season, label=context_label)
    
        df = self.fetch_scope(country=country, league=league, season=season,
                              extra_where=extra_where, params=params)
        if df.empty:
            raise ValueError("No data loaded for the given filters.")
    
        # resolve estimator / column name
        self._imp_col = self._resolve_imp_col(df)
    
        # normalize player_name and drop unknown/blank
        df["player_name"] = df["player_name"].fillna("").astype(str)
        df = df[~df["player_name"].str.strip().str.lower().isin(["unknown", ""])]
    
        self.players_df = df

    def merge_profiles_and_transfers(
        self,
        *,
        profiles_db: str = "player_profiles.db",
        transfers_db: str | None = "player_transfer.db",
        season_start_month: int = 8, season_start_day: int = 1,
        endseason_cutoff_month: int = 10, endseason_cutoff_day: int = 1,  # Oct 1 (next year)
        mid_start_month: int = 12, mid_start_day: int = 1,                # Dec 1 (season year)
        mid_end_month: int = 3, mid_end_day: int = 1                      # Mar 1 (next year)
    ) -> None:
        """
        Enrich self.players_df with:
          - nationality (from profiles)
          - age_at_season_start (Aug 1 by default)
          - fill unknown player_name/position from profiles when available
          - transfer_window bucket using DB (dates) and heuristics
        """
    
        if self.players_df.empty:
            raise ValueError("Load data first (call .load()).")
    
        # ---- pull minimal profiles for this season ----
        prof = pd.DataFrame()
        if profiles_db and os.path.exists(profiles_db):
            try:
                with sqlite3.connect(profiles_db) as conP:
                    q = """SELECT player_id, season, name, position, nationality, birth_date
                           FROM player_profiles"""
                    prof = pd.read_sql_query(q, conP)
            except Exception:
                prof = pd.DataFrame()
    
        # best-effort normalize
        if not prof.empty:
            prof["player_id"] = pd.to_numeric(prof["player_id"], errors="coerce").astype("Int64")
            prof["season"]    = pd.to_numeric(prof["season"], errors="coerce").astype("Int64")
            prof = prof[prof["season"] == int(self.season)]
            prof = prof.drop_duplicates("player_id")
    
            # left merge on player_id
            df = self.players_df.merge(
                prof[["player_id","name","position","nationality","birth_date"]],
                on="player_id", how="left", suffixes=("","_prof")
            )
            # Fill unknown/blank player_name and position from profiles
            unk_name = df["player_name"].fillna("").str.strip().str.lower().isin(["", "unknown"])
            df.loc[unk_name, "player_name"] = df.loc[unk_name, "name"].where(df.loc[unk_name, "name"].notna(), df.loc[unk_name, "player_name"])
            unk_pos = df["position"].fillna("").str.strip().str.lower().isin(["", "unknown"])
            df.loc[unk_pos, "position"] = df.loc[unk_pos, "position_prof"].where(df.loc[unk_pos, "position_prof"].notna(), df.loc[unk_pos, "position"])
            # Nationality (prefer profiles if present)
            if "nationality" in df.columns and "nationality_prof" in df.columns:
                df["nationality"] = df["nationality"].where(df["nationality"].notna(), df["nationality_prof"])
            elif "nationality_prof" in df.columns and "nationality" not in df.columns:
                df["nationality"] = df["nationality_prof"]
            # Age at season start
            start_dt = pd.Timestamp(date(int(self.season), season_start_month, season_start_day), tz="UTC")
            bdates = pd.to_datetime(df["birth_date"], errors="coerce", utc=True)
            df["age_at_season_start"] = ((start_dt - bdates).dt.days / 365.25).astype(float)
            self.players_df = df.drop(columns=[c for c in ["name","position_prof","nationality_prof"] if c in df.columns])
        else:
            # No profiles DB; still ensure the column exists for downstream modules
            if "age_at_season_start" not in self.players_df.columns:
                self.players_df["age_at_season_start"] = np.nan
    
        # ---- transfer window annotation (dates + heuristics fallback) ----
        self._ensure_transfer_window(
            transfers_db=transfers_db,
            season_start_month=season_start_month, season_start_day=season_start_day,
            endseason_cutoff_month=endseason_cutoff_month, endseason_cutoff_day=endseason_cutoff_day,
            mid_start_month=mid_start_month, mid_start_day=mid_start_day,
            mid_end_month=mid_end_month, mid_end_day=mid_end_day
        )

    def _ensure_transfer_window(
        self,
        *,
        transfers_db: str | None,
        season_start_month: int = 8, season_start_day: int = 1,
        endseason_cutoff_month: int = 10, endseason_cutoff_day: int = 1,  # Oct 1 (next year)
        mid_start_month: int = 12, mid_start_day: int = 1,                # Dec 1 (season year)
        mid_end_month: int = 3, mid_end_day: int = 1                      # Mar 1 (next year)
    ) -> None:
        """
        Populate/overwrite self.players_df['transfer_window'] with:
          'mid-season' | 'end-of-season' | 'no-transfer'
    
        Priority:
          1) Use transfer dates from player_transfer.db if present.
          2) Else if current 'team(s)' looks multi-team -> 'mid-season'.
          3) Else if previous-season team != current team -> 'end-of-season'.
          4) Else 'no-transfer'.
        """
    
        df = self.players_df.copy()
        if df.empty:
            return
    
        # Season anchors (with year rollovers)
        Y = int(self.season)
        season_start = pd.Timestamp(date(Y, season_start_month, season_start_day), tz="UTC")
        mid_start    = pd.Timestamp(date(Y, mid_start_month, mid_start_day), tz="UTC")
        mid_end      = pd.Timestamp(date(Y + 1, mid_end_month, mid_end_day), tz="UTC")
        eos_cutoff   = pd.Timestamp(date(Y + 1, endseason_cutoff_month, endseason_cutoff_day), tz="UTC")
    
        # --- 1) Read transfers if available -------------------------------------
        transfers = pd.DataFrame()
        if transfers_db and os.path.exists(transfers_db):
            try:
                with sqlite3.connect(transfers_db) as conT:
                    # Be permissive about schema: look for plausible columns
                    cols = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table'", conT
                    )["name"].tolist()
                    # pick first table that has player_id and date
                    chosen = None
                    for t in cols:
                        info = pd.read_sql_query(f'PRAGMA table_info("{t}")', conT)
                        low = {c.lower() for c in info["name"].tolist()}
                        if "player_id" in low and ("date" in low or "transfer_date" in low):
                            chosen = t
                            break
                    if chosen:
                        transfers = pd.read_sql_query(f'SELECT * FROM "{chosen}"', conT)
            except Exception:
                transfers = pd.DataFrame()
    
        # Normalize transfer date and reduce to one relevant date per player
        if not transfers.empty:
            # unify column names
            cols = {c.lower(): c for c in transfers.columns}
            pid_col = cols.get("player_id")
            date_col = cols.get("date") or cols.get("transfer_date")
            if pid_col and date_col:
                tt = transfers[[pid_col, date_col]].copy()
                tt.columns = ["player_id", "tdate_raw"]
                tt["player_id"] = pd.to_numeric(tt["player_id"], errors="coerce").astype("Int64")
                tt["tdate"] = tt["tdate_raw"].apply(self._parse_date_safe)
                # Keep the earliest transfer that falls between season start and eos_cutoff(+ buffer),
                # or the closest transfer to season_start if none inside the window.
                mask_window = (tt["tdate"] >= season_start) & (tt["tdate"] < eos_cutoff)
                tt_in = tt[mask_window].sort_values(["player_id", "tdate"]).dropna(subset=["player_id","tdate"])
                if tt_in.empty:
                    tt_use = (tt.sort_values(["player_id","tdate"])
                                .dropna(subset=["player_id","tdate"])
                                .groupby("player_id", as_index=False).first())
                else:
                    tt_use = tt_in.groupby("player_id", as_index=False).first()
            else:
                tt_use = pd.DataFrame()
        else:
            tt_use = pd.DataFrame()
    
        # Map player_id -> transfer_date (Timestamp)
        tmap = {}
        if not tt_use.empty:
            for _, r in tt_use.iterrows():
                tmap[int(r["player_id"])] = r["tdate"]
    
        # --- 2) Pull previous-season team(s) for heuristic -----------------------
        prev_map = {}
        try:
            with sqlite3.connect(self.db_path) as conA:
                qprev = """SELECT player_id, [team(s)] AS team_s
                           FROM analysis_results
                           WHERE country=? AND league=? AND season=?"""
                prev = pd.read_sql_query(qprev, conA, params=[self.country, self.league, Y-1])
                prev["player_id"] = pd.to_numeric(prev["player_id"], errors="coerce").astype("Int64")
                prev = prev.dropna(subset=["player_id"])
                prev_map = prev.drop_duplicates("player_id").set_index("player_id")["team_s"].to_dict()
        except Exception:
            prev_map = {}
    
        # --- 3) Classify each row ------------------------------------------------
        windows = []
        for _, row in df.iterrows():
            pid = row.get("player_id")
            cur_team = str(row.get("team(s)", "") or "")
            label = "no-transfer"
    
            # 3.1) date-based if available
            d = tmap.get(int(pid)) if pd.notnull(pid) else None
            if d is not None and pd.notnull(d):
                if (d >= mid_start) and (d < mid_end):
                    label = "mid-season"
                elif (d < mid_start) or (d >= mid_end and d < eos_cutoff):
                    label = "end-of-season"
                else:
                    label = "no-transfer"
            else:
                # 3.2) textual multi-team signal for this season
                if self._looks_multiteam(cur_team):
                    label = "mid-season"
                else:
                    # 3.3) team changed from previous season
                    prev_team = prev_map.get(int(pid))
                    if prev_team is not None:
                        if str(prev_team) != cur_team and cur_team != "":
                            label = "end-of-season"
                    # otherwise remains 'no-transfer'
    
            windows.append(label)
    
        self.players_df["transfer_window"] = pd.Series(windows, index=self.players_df.index)

    def _looks_multiteam(self, s: str) -> bool:
        # Heuristics for "played for >1 team this season"
        if not s:
            return False
        s = str(s)
        if s.startswith("Multiple Teams"):
            return True
        # any obvious separators between club names
        for tok in [" / ", " & ", " + ", ";", "|", "→", "->"]:
            if tok in s:
                return True
        return False

    
    def _finish(self, fig: plt.Figure, name: str) -> Optional[str]:
        """Standardized save/show/close for all plots."""
        out = os.path.join(f"{self._fmt_scope_title('scope').replace(' / ','_')}_{name}.png")
        if self.save_plot:
            fig.savefig(out, dpi=140, bbox_inches="tight")
        if self.show_plot:
            plt.show()
        plt.close(fig)
        return out if self.save_plot else None
 





    # ---------------------- Modules ----------------------
    def mod_top_vs_bottom(self, k=5, show_values=False, show_club=False):
        # Select top k and bottom k players by contribution
        df = self.players_df.sort_values(self._imp_col, ascending=False)
        sel = pd.concat([df.head(k), df.tail(k)])
        
        # Prepare items for tornado chart
        items = []
        for _, row in sel.iterrows():
            name = str(row["player_name"])
            if show_club:
                # Include club name below player name (if available)
                club = str(row.get("team(s)", ""))  # team(s) column holds the club name(s)
                if club:
                    name = f"{name}\n{club}"
            items.append({
                "name": name,
                "value": row[self._imp_col],
                "color": (self.style.colors["top"] if row[self._imp_col] >= 0 
                          else self.style.colors["bottom"])
            })
        
        # Create tornado chart, with optional value labels
        fig = self.graphs.tornado(
            items,
            title=f"{self._fmt_scope_title()}: Top {k} and Bottom {k}",
            subtitle=("Bars right = high contributors; Bars left = low contributors.\n"
                      "Labels centered at zero for direct comparison."),
            xlabel="Off-Balance-Sheet Contribution",
            symmetric=True,
            show_values=show_values   # show numeric values at bar ends if True
        )
        return self._finish(fig, "tornado")    
    
    def mod_contrib_vs_fte_by_nationality(
        self,
        *,
        top_n: int = 8,                 # how many nationalities to break out explicitly
        include_other: bool = True,     # combine all remaining into "Other"
        min_minutes: float = 0.0,       # optional minutes filter
        fte_anchor_min: float = 20.0,   # anchors: FTE ≥ this
        fte_breakout_max: float = 5.0,  # breakouts: FTE ≤ this
        top_k_anchor: int = 4,
        top_k_breakout: int = 4,
    ):
        """
        Scatter of Off-Balance-Sheet Contribution vs FTE, grouped by nationality.
        Requires 'nationality' to be present (run merge_profiles_and_transfers first).
        """
        if "nationality" not in self.players_df.columns:
            raise ValueError("Column 'nationality' not found. Run merge_profiles_and_transfers() first.")
    
        df = self.players_df.copy()
    
        # FTE (safe)
        if "FTE_games_played" in df.columns:
            fte = pd.to_numeric(df["FTE_games_played"], errors="coerce")
        else:
            fte = pd.to_numeric(df.get("minutes_played", 0), errors="coerce") / 90.0
        df["FTE"] = fte.clip(lower=0).fillna(0.0)
    
        # optional minutes filter
        if "minutes_played" in df.columns and min_minutes > 0:
            df = df[pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0.0) >= float(min_minutes)]
    
        # top-N nationalities by player count
        counts = df["nationality"].fillna("Unknown").value_counts()
        keep = set(counts.head(int(top_n)).index.tolist())
        df["_nat_grp"] = np.where(df["nationality"].isin(keep), df["nationality"], "Other")
        if not include_other:
            df = df[df["_nat_grp"] != "Other"]
    
        # build sources (one per nationality bucket)
        sources = []
        for nat in sorted(df["_nat_grp"].unique(), key=lambda x: (x == "Other", x)):
            sub = df[df["_nat_grp"] == nat]
            if sub.empty: 
                continue
            sources.append({
                "x": sub["FTE"].to_numpy(dtype=float),
                "y": sub[self._imp_col].to_numpy(dtype=float),
                "labels": sub["player_name"].astype(str).to_numpy(),
                "name": str(nat),      # color: auto; legend: auto because >1 source
                # marker/size/alpha -> defaults from scatter_fte
            })
    
        title = f"{self.league} {self.season}: Off-Balance-Sheet Contributions vs FTE by Nationality"
        subtitle = ("Each dot is a player. Colors = nationality groups.\n"
                    f"Red labels: standout anchors (FTE ≥ {fte_anchor_min:g}) and breakouts (FTE ≤ {fte_breakout_max:g}).")
    
        fig = self.graphs.scatter_fte(
            sources=sources,
            title=title,
            subtitle=subtitle,
            xlabel="FTE games (minutes/90)",
            ylabel="Off-Balance-Sheet Contribution",
            hline="zero",
            regline=False,
            reg_ci=None
        )
        return self._finish(fig, "scatter_fte_by_nationality")
    
    def mod_contrib_vs_age(
        self,
        *,
        by_position: bool = False,       # if True: split sources by position bucket
        min_minutes: float = 0.0,        # optional minutes filter
        anchor_age_min: float | None = None,   # highlight older anchors (>=)
        breakout_age_max: float | None = None, # highlight younger breakouts (<=)
        top_k_anchor: int = 4,
        top_k_breakout: int = 4,
        add_regression: bool = True,
        add_ci: str | None = "prediction",     # None | "prediction" | "mean"
    ):
        """
        Scatter of Off-Balance-Sheet Contribution vs Age at season start.
        Requires 'age_at_season_start' (run merge_profiles_and_transfers first).
        """
        if "age_at_season_start" not in self.players_df.columns:
            raise ValueError("Column 'age_at_season_start' not found. Run merge_profiles_and_transfers() first.")
    
        df = self.players_df.copy()
    
        # optional minutes filter
        if "minutes_played" in df.columns and min_minutes > 0:
            df = df[pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0.0) >= float(min_minutes)]
    
        # fallback position bucket
        def _bucket(pos: str) -> str:
            p = (pos or "").lower()
            if "gk" in p or "keeper" in p: return "Goalkeepers"
            if "def" in p or "back" in p:  return "Defenders"
            if "mid" in p:                 return "Midfielders"
            if any(k in p for k in ["fw","st","cf","wing","forward"]): return "Attackers"
            return "Other"
        df["_pos_grp"] = df.get("position", "").astype(str).map(_bucket)
    
        x = pd.to_numeric(df["age_at_season_start"], errors="coerce").to_numpy()
    
        # sources
        sources = []
        if by_position:
            for grp in ["Goalkeepers","Defenders","Midfielders","Attackers","Other"]:
                sub = df[df["_pos_grp"] == grp]
                if sub.empty: continue
                sources.append({
                    "x": pd.to_numeric(sub["age_at_season_start"], errors="coerce").to_numpy(),
                    "y": pd.to_numeric(sub[self._imp_col], errors="coerce").to_numpy(),
                    "labels": sub["player_name"].astype(str).to_numpy(),
                    "name": grp
                })
        else:
            sources.append({
                "x": pd.to_numeric(df["age_at_season_start"], errors="coerce").to_numpy(),
                "y": pd.to_numeric(df[self._imp_col], errors="coerce").to_numpy(),
                "labels": df["player_name"].astype(str).to_numpy(),
                "name": None
            })
    
        # highlight older anchors / younger breakouts if thresholds provided
        highlights = []
        if anchor_age_min is not None:
            anc = (df[pd.to_numeric(df["age_at_season_start"], errors="coerce") >= float(anchor_age_min)]
                   .sort_values(self._imp_col, ascending=False).head(int(top_k_anchor)))
            for _, r in anc.iterrows():
                highlights.append({"x": float(r["age_at_season_start"]),
                                   "y": float(r[self._imp_col]),
                                   "text": str(r["player_name"]), "side": "left"})
        if breakout_age_max is not None:
            bro = (df[pd.to_numeric(df["age_at_season_start"], errors="coerce") <= float(breakout_age_max)]
                   .sort_values(self._imp_col, ascending=False).head(int(top_k_breakout)))
            for _, r in bro.iterrows():
                highlights.append({"x": float(r["age_at_season_start"]),
                                   "y": float(r[self._imp_col]),
                                   "text": str(r["player_name"]), "side": "right"})
    
        title = f"{self.league} {self.season}: Off-Balance-Sheet Contributions vs Age"
        subtitle = ("Each dot is a player."
                    + (" Colors = positions." if by_position else "")
                    + (f"\nRed labels: anchors age ≥ {anchor_age_min:g}." if anchor_age_min is not None else "")
                    + (f" Breakouts age ≤ {breakout_age_max:g}." if breakout_age_max is not None else ""))
    
        fig = self.graphs.scatter_fte(
            sources=sources,
            title=title,
            subtitle=subtitle,
            xlabel="Age at season start (years)",
            ylabel="Off-Balance-Sheet Contribution",
            hline="zero",
            regline=bool(add_regression),
            reg_ci=add_ci,
            highlights=highlights
        )
        return self._finish(fig, "scatter_age")

    def mod_contrib_vs_fte_by_transfer_status(
        self,
        *,
        min_minutes: float = 0.0,
        highlight_top_k_each: int = 3,
    ):
        """
        Scatter of Off-Balance-Sheet Contribution vs FTE by transfer-window status.
        Requires 'transfer_window' (set by merge_profiles_and_transfers).
        """
        # at the top of mod_contrib_vs_fte_by_transfer_status
        if "transfer_window" not in self.players_df.columns:
            # populate using heuristics only (no profiles needed here)
            self._ensure_transfer_window(
                transfers_db=None,  # don’t rely on dates if you prefer; or pass the path you use
                season_start_month=8, season_start_day=1,
                endseason_cutoff_month=10, endseason_cutoff_day=1,
                mid_start_month=12, mid_start_day=1,
                mid_end_month=3, mid_end_day=1
            )

        if "transfer_window" not in self.players_df.columns:
            raise ValueError("Column 'transfer_window' not found. Run merge_profiles_and_transfers() first.")
    
        df = self.players_df.copy()
    
        # FTE (safe)
        if "FTE_games_played" in df.columns:
            fte = pd.to_numeric(df["FTE_games_played"], errors="coerce")
        else:
            fte = pd.to_numeric(df.get("minutes_played", 0), errors="coerce") / 90.0
        df["FTE"] = fte.clip(lower=0).fillna(0.0)
    
        # optional minutes filter
        if "minutes_played" in df.columns and min_minutes > 0:
            df = df[pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0.0) >= float(min_minutes)]
    
        # normalized window values
        order = ["no-transfer","end-of-season","mid-season"]
        df["_win"] = pd.Categorical(df["transfer_window"].astype(str).str.lower(), categories=order, ordered=True)
    
        sources = []
        for cat in order:
            sub = df[df["_win"] == cat]
            if sub.empty: 
                continue
            sources.append({
                "x": sub["FTE"].to_numpy(dtype=float),
                "y": sub[self._imp_col].to_numpy(dtype=float),
                "labels": sub["player_name"].astype(str).to_numpy(),
                "name": cat.replace("-", " ").title()
            })
    
        title = f"{self.league} {self.season}: Contributions vs FTE by Transfer Window"
        subtitle = "Colors represent transfer timing (none, end-of-season, mid-season). Labels show top contributors in each bucket."
    
        fig = self.graphs.scatter_fte(
            sources=sources,
            title=title,
            subtitle=subtitle,
            xlabel="FTE games (minutes/90)",
            ylabel="Off-Balance-Sheet Contribution",
            hline="zero",
            regline=False,
        )
        return self._finish(fig, "scatter_fte_transfer_status")

    def mod_contrib_vs_fte_multi(
        self,
        groups: dict[str, list[str]] = None,   # if None → one source per pos_bucket found (all players)
        color_map: dict[str, str] = None,      # optional {display_name: color}
        fte_anchor_min: float = 20.0,
        fte_breakout_max: float = 5.0,
        top_k_anchor: int = 4,
        top_k_breakout: int = 4,
        title_suffix: str = "Off-Balance-Sheet Contributions vs FTE",
        xlabel: str = "FTE games (minutes/90)",
        ylabel: str = "Off-Balance-Sheet Contribution",
    ):
        """
        Multi-category scatter: by default includes *all* players, grouped by pos_bucket.
        If `groups` is None, we auto-create one source per bucket present in the data.
    
        groups (optional): mapping of display-name -> list of lower-case tokens to match in df['pos_bucket'].
                           If None, we infer from the data (one source per bucket present).
        """
        df = self.players_df.copy()
    
        # --- FTE ---
        if "FTE_games_played" in df.columns:
            fte = pd.to_numeric(df["FTE_games_played"], errors="coerce")
        else:
            fte = pd.to_numeric(df.get("minutes_played", 0), errors="coerce") / 90.0
        df["FTE"] = fte.clip(lower=0).fillna(0.0)
    
        # --- Position buckets (lower-case tokens for matching) ---
        if "pos_bucket" not in df.columns:
            def _pos_bucket(pos: str) -> str:
                p = (pos or "").strip().lower()
                if any(k in p for k in ["G","gk","keeper","goalkeeper"]): return "goalkeeper"
                if any(k in p for k in ["D","def","cb","rb","lb","back","defender"]): return "defender"
                if any(k in p for k in ["M","mid","cm","dm","am","wing","midfielder"]): return "midfielder"
                if any(k in p for k in ["F","fw","st","lw","rw","cf","striker","forward"]): return "attacker"
                return "other"
            df["pos_bucket"] = df.get("position","").astype(str).map(_pos_bucket)
        else:
            df["pos_bucket"] = df["pos_bucket"].astype(str).str.lower()
    
        # --- If no groups provided, auto-build one per bucket present (ALL players) ---
        if groups is None:
            # Keep a stable, readable order
            order = ["goalkeeper", "defender", "midfielder", "attacker", "other"]
            present = [b for b in order if b in df["pos_bucket"].unique()]
            # Display names: title-case, except GK → "Goalkeepers"
            def _disp(b):
                return "Goalkeepers" if b == "goalkeeper" else b.capitalize() + "s"
            groups = { _disp(b): [b] for b in present }  # e.g., {"Defenders":["defender"], ...}
    
        # --- Colors (cycle if not provided) ---
        if color_map is None:
            palette = ["tab:green","tab:blue","tab:orange","tab:red","tab:purple","tab:brown","tab:pink","tab:cyan","tab:olive","tab:gray"]
            color_map = {name: palette[i % len(palette)] for i, name in enumerate(groups.keys())}
    
        # --- Build sources (skip empties) ---
        sources = []
        for display_name, wanted_tokens in groups.items():
            wanted = [w.strip().lower() for w in wanted_tokens]
            mask = df["pos_bucket"].isin(wanted)
            src_df = df.loc[mask]
            if src_df.empty:
                continue
            sources.append({
                "x": src_df["FTE"].to_numpy(dtype=float),
                "y": src_df[self._imp_col].to_numpy(dtype=float),
                "labels": src_df["player_name"].astype(str).to_numpy(),
                "color": color_map.get(display_name, "tab:blue"),
                "name": display_name,   # legend label (shown if show_legend=True)
                "marker": "o",
                "size": 38,
                "alpha": 0.75,
            })
    
        # --- Highlights: anchors & breakouts across ALL players (same rules as before) ---
        anchors_pool = df[df["FTE"] >= float(fte_anchor_min)]
        anchors = anchors_pool.sort_values(self._imp_col, ascending=False).head(int(top_k_anchor))
    
        breakouts_pool = df[df["FTE"] <= float(fte_breakout_max)]
        breakouts = breakouts_pool.sort_values(self._imp_col, ascending=False).head(int(top_k_breakout))
    
        highlights = []
        for _, r in anchors.iterrows():
            highlights.append({
                "x": float(r["FTE"]),
                "y": float(r[self._imp_col]),
                "text": str(r["player_name"]),
                "side": "left",   # label to the left of the dot
                "color": "red",
            })
        for _, r in breakouts.iterrows():
            highlights.append({
                "x": float(r["FTE"]),
                "y": float(r[self._imp_col]),
                "text": str(r["player_name"]),
                "side": "right",  # label to the right of the dot
                "color": "red",
            })
    
        # --- Titles ---
        league = (self.league or "League")
        season = (self.season or "")
        title = f"{league} {season}: {title_suffix}"
        subtitle = (
            "Each dot is a player. Multiple categories shown in different colors.\n"
            f"Red labels: standout anchors (FTE ≥ {fte_anchor_min:g}) and breakouts (FTE ≤ {fte_breakout_max:g})."
        )
    
        # --- Draw ---
        fig = self.graphs.scatter_fte(
            sources=sources,
            title=title,
            subtitle=subtitle,
            xlabel=xlabel,
            ylabel=ylabel,
            show_legend=False,
            highlights=highlights,
        )
        return self._finish(fig, "scatter_fte_multi")

    def mod_team_ranking(self):
        # Exclude "Multiple Teams" entries (transfer combinations) from the team list
        df = self.players_df.copy()
        mask = df["team(s)"].fillna("").str.startswith("Multiple Teams")
        df = df[~mask]
        if df.empty:
            raise ValueError("No team data available for ranking.")
        
        # Compute average contribution per team and sort values
        avg_contrib = df.groupby("team(s)")[self._imp_col].mean().sort_values(ascending=True)
        fig = self.graphs.ranking(
            avg_contrib.index.tolist(), avg_contrib.values,
            title=f"{self._fmt_scope_title()}: Average by Team",#title=f"{self.league} {self.season}: Average by Team",
            subtitle="Teams ranked by average off-balance contributions"
        )
        return self._finish(fig, "ranking")

    
    # def mod_transfer_combinations(self):
    #     """
    #     Ranking chart for players who are recorded under 'Multiple Teams (...)' in this season.
    #     - Keeps only those rows.
    #     - Uses the content inside parentheses as label.
    #     - Replaces commas with ' → ' in labels for readability.
    #     """
    #     import re
    
    #     df = self.players_df.copy()
    #     multi_mask = df["team(s)"].fillna("").str.startswith("Multiple Teams")
    #     df_multi = df[multi_mask]
    #     if df_multi.empty:
    #         print("No multiple-team transfer entries to display.")
    #         return None
    
    #     # Average contribution by the combined team entry
    #     avg_contrib = (
    #         df_multi.groupby("team(s)")[self._imp_col]
    #         .mean()
    #         .sort_values(ascending=True)
    #     )
    
    #     # Build display labels: extract (...) content and replace commas with ' → '
    #     labels = []
    #     for team_label in avg_contrib.index:
    #         label_str = str(team_label)
    #         # extract content inside parentheses, if present
    #         start = label_str.find("(")
    #         end   = label_str.rfind(")")
    #         if start != -1 and end != -1 and end > start:
    #             label_str = label_str[start+1:end]
    #         else:
    #             # fallback: strip the leading phrase
    #             label_str = label_str.replace("Multiple Teams", "").strip()
    
    #         # replace commas with arrow
    #         label_str = re.sub(r"\s*,\s*", " → ", label_str).strip()
    #         labels.append(label_str)
    
    #     fig = self.graphs.ranking(
    #         labels, avg_contrib.values,
    #         title=f"{self._fmt_scope_title()}: Multiple Teams (Transfer) Average",#title=f"{self.league} {self.season}: Multiple Teams (Transfer) Average",
    #         subtitle="Average contribution for players who featured for multiple teams (sequence shown with arrows)"
    #     )
    #     return self._finish(fig, "ranking_multiteam")



    # def mod_position_buckets(self):
    #     df=self.players_df.copy(); df["pos_bucket"]=df["position"]
    #     fig=self.graphs.bucket(df["pos_bucket"],df[self._imp_col],group_labels=df["pos_bucket"].unique(),
    #         title=f"{self.league} {self.season}: Contributions by Position",
    #         subtitle="Boxplots show spread of contributions per position")
    #     return self._finish(fig,"bucket")

    
    
    def mod_season_improve(
        self,
        comp_df,
        *,
        max_items: int = 30,
        sort_start_ascending: bool = True,
        show_delta: bool = True,
        color: str | None = None,   # None → use slope's default (first cycle color)
    ):
        """
        Improvements only (current > previous).
        Builds slope items for the simplified Graphs.slope API.
    
        Behavior (handled by Graphs.slope):
          - Sorts by START value (previous season) ascending/descending.
          - For increases: label appears LEFT of the start (right-aligned).
          - Δ printed at the final value (right for positive delta).
          - A dot is drawn at the start value (previous season).
        """
        # Resolve estimator column in the comparison frame
        comp_imp_col = self._resolve_imp_col(comp_df)
    
        # Current vs previous
        now  = self.players_df[["player_name", self._imp_col]].rename(columns={self._imp_col: "imp_now"})
        prev = comp_df[["player_name", comp_imp_col]].rename(columns={comp_imp_col: "imp_prev"})
    
        merged = (
            pd.merge(now, prev, on="player_name", how="inner")
              .dropna(subset=["imp_prev", "imp_now"])
        )
        if merged.empty:
            raise ValueError("No overlap between seasons to plot improvements.")
    
        # Keep only improvements and cap
        merged = merged[merged["imp_now"] > merged["imp_prev"]]
        if merged.empty:
            raise ValueError("No players improved from previous to current season.")
        merged = merged.head(int(max_items)).reset_index(drop=True)
    
        # Build items (Δ shown if requested; color optional → default cycle if None)
        items = []
        for _, r in merged.iterrows():
            d = {
                "start_value": float(r["imp_prev"]),
                "final_value": float(r["imp_now"]),
                "label": str(r["player_name"]),
                "flag_to_write_value": bool(show_delta),
                "linewidth": 1.8,
            }
            if color is not None:
                d["color"] = color
            items.append(d)
    
        title    = f"{self._fmt_scope_title('Scope')}: Improvements from Previous to Current"
        subtitle = "Arrow: previous → current. Labels auto-placed; Δ shown at the final value."
    
        fig = self.graphs.slope(
            items,
            title=title,
            subtitle=subtitle,
            sort_start_ascending=bool(sort_start_ascending),
        )
        return self._finish(fig, "slope_improved_only")


    def mod_season_decline(
        self,
        comp_df,
        *,
        max_items: int = 30,
        sort_start_ascending: bool = True
    ):
        """
        Declines only (current < previous).
        Builds slope items and uses the simplified Graphs.slope API.
        - Names auto-placed relative to START as per the slope spec.
        - Δ shown per row (flag=True).
        - Color set to red for declines (module-level choice; slope has a neutral default).
        """
        comp_imp_col = self._resolve_imp_col(comp_df)
    
        now  = self.players_df[["player_name", self._imp_col]].rename(columns={self._imp_col: "imp_now"})
        prev = comp_df[["player_name", comp_imp_col]].rename(columns={comp_imp_col: "imp_prev"})
    
        merged = pd.merge(now, prev, on="player_name", how="inner").dropna(subset=["imp_prev", "imp_now"])
        if merged.empty:
            raise ValueError("No overlap between seasons to plot season decline.")
    
        # Declines only & cap
        merged = merged[merged["imp_now"] < merged["imp_prev"]]
        if merged.empty:
            raise ValueError("No players declined from previous to current season.")
        merged = merged.head(int(max_items)).reset_index(drop=True)
    
        # Build items (make deltas visible; color red here)
        items = []
        for _, r in merged.iterrows():
            items.append({
                "start_value": float(r["imp_prev"]),
                "final_value": float(r["imp_now"]),
                "label": str(r["player_name"]),
                "flag_to_write_value": True,
                "color": "red",
                "linewidth": 1.8,
            })
    
        title    = f"{self._fmt_scope_title('Scope')}: Declines from Previous to Current"
        subtitle = "Arrow: previous → current. Labels auto-placed; Δ shown at the final value."
    
        fig = self.graphs.slope(
            items,
            title=title,
            subtitle=subtitle,
            sort_start_ascending=bool(sort_start_ascending),
        )
        return self._finish(fig, "slope_declined_only")
    
    def mod_team_buckets_horizontal(
        self,
        *,
        order: list[str] | None = None,      # explicit team order; if None we'll sort by `order_by`
        min_players_per_team: int = 3,
        symmetric: bool = True,
        showfliers: bool = True,
    
        # NEW (optional) ordering controls:
        order_by: str = "median",            # "median" | "mean" | "iqr" | "count"
        ascending: bool = True,              # sort direction when order is not provided
        top_n: int | None = None             # keep only first N teams after ordering
    ):
        """
        Tornado-style horizontal buckets: one box per team showing the distribution of player
        contributions. Centered team names at x=0. The middle (IQR) part is colored (handled
        by Graphs.bucket_horizontal).
    
        - Filters out 'Multiple Teams...' pseudo-teams.
        - Keeps only teams with >= `min_players_per_team` observations.
        - If `order` is None, teams are ordered by `order_by` (median|mean|iqr|count).
        - Set `symmetric=True` to force symmetric x-limits around 0 (recommended for comparison).
        """
        import numpy as np
        import pandas as pd
    
        df = self.players_df.copy()
    
        # Exclude pseudo-teams and clean
        df = df[~df["team(s)"].fillna("").str.startswith("Multiple Teams")]
        df = df.dropna(subset=[self._imp_col, "team(s)"])
        if df.empty:
            raise ValueError("No team/player rows available after filtering.")
    
        # Build: team -> list of values (float), require minimum sample size
        groups: dict[str, list[float]] = {}
        for tm, sub in df.groupby("team(s)"):
            vals = pd.to_numeric(sub[self._imp_col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size >= int(min_players_per_team):
                groups[tm] = vals.tolist()
    
        if not groups:
            raise ValueError(f"No teams with at least {min_players_per_team} observations.")
    
        # Determine order
        if order is None:
            def _score(v: list[float]) -> float:
                arr = np.asarray(v, dtype=float)
                if order_by == "mean":
                    return float(np.nanmean(arr))
                if order_by == "iqr":
                    q1, q3 = np.nanpercentile(arr, [25, 75])
                    return float(q3 - q1)
                if order_by == "count":
                    return float(arr.size)
                # default: median
                return float(np.nanmedian(arr))
    
            order = sorted(groups.keys(), key=lambda t: _score(groups[t]), reverse=not bool(ascending))
        else:
            # keep only teams present, preserve user order
            order = [t for t in order if t in groups]
    
        if top_n is not None:
            order = order[:int(top_n)]
    
        if not order:
            raise ValueError("No teams left after applying ordering/top_n filters.")
    
        # Assemble items for the horizontal bucket graph
        items = [{"name": tm, "values": groups[tm]} for tm in order]
    
        title = f"{self._fmt_scope_title('Scope')}: Team Distributions (Player Contributions)"
        subtitle = ("Boxes show 25–75% (IQR) with whiskers/outliers; "
                    "colored center band stays visible under centered labels.")
    
        fig = self.graphs.bucket_horizontal(
            items,
            title=title,
            subtitle=subtitle,
            xlabel="Off-Balance-Sheet Contribution",
            symmetric=bool(symmetric),
            showfliers=bool(showfliers),
            # You can tweak these if you want a thicker band:
            # box_width=0.55, band_scale=1.25, band_alpha=0.90
        )
        return self._finish(fig, "bucket_horizontal_teams") 
    
    def mod_position_buckets_by_season(
        self,
        seasons: list[int],                         # e.g., [2023, 2024]
        *,
        positions: tuple[str, ...] = ("Defenders","Midfielders","Strikers"),
        country: str | None = None,
        league: str | None = None,
    ):
        """
        Grouped vertical buckets by position buckets across multiple seasons (same league).
        Requires identical sub-observations across top groups (we enforce with 'seasons').
    
        positions: tuple of top-level buckets to include (order preserved).
                   We auto-map raw 'position' strings to these coarse buckets.
        """
        if country is None: country = self.country
        if league  is None: league  = self.league
        if not country or not league:
            raise ValueError("Set context or pass country= and league=.")
    
        if not seasons or len(seasons) < 2:
            raise ValueError("Provide at least two seasons (e.g., [2023, 2024]).")
    
        # helper: map position text to coarse bucket
        def _bucket(pos: str) -> str:
            p = (pos or "").lower()
            if any(k in p for k in ["d","def","back","cb","rb","lb","df"]): return "Defenders"
            if any(k in p for k in ["m","mid","cm","dm","am","wing","mf"]): return "Midfielders"
            if any(k in p for k in ["f","fw","st","cf","striker","forward","attacker"]): return "Strikers"
            return "Other"
    
        # build nested {top -> {season_label -> values}}
        data_map: dict[str, dict[str, list[float]]] = {top: {} for top in positions}
        for s in seasons:
            df_s = self.fetch_scope(country=country, league=league, season=int(s))
            if df_s.empty:
                # keep season label, but empty
                for top in positions:
                    data_map[top][f"{league} {s}"] = []
                continue
    
            # normalize
            df_s = df_s.dropna(subset=[self._imp_col])
            df_s["_bucket"] = df_s.get("position","").astype(str).map(_bucket)
    
            for top in positions:
                vals = df_s.loc[df_s["_bucket"] == top, self._imp_col].astype(float).to_numpy()
                data_map[top][f"{league} {s}"] = vals.tolist()
    
        # ensure identical sub-order across groups
        sub_order = [f"{league} {s}" for s in seasons]
        group_order = list(positions)
    
        title = f"{league}: Position Buckets across Seasons"
        subtitle = f"Distributions for {', '.join(positions)} in " + ", ".join(str(s) for s in seasons)
    
        fig = self.graphs.bucket_grouped_vertical(
            data_map,
            group_order=group_order,
            sub_order=sub_order,
            title=title,
            subtitle=subtitle,
            ylabel="Off-Balance-Sheet Contribution",
            show_legend=True,
            figsize=(12, 6),
        )
        return self._finish(fig, "bucket_grouped_positions")


    def mod_multiline_top_players_over_seasons(
        self,
        *,
        top_k: int = 8,
        seasons: list[int] | None = None,   # None → [season-4 .. season]
        agg: str = "sum",                   # aggregate duplicates per player per season
        x_axis: str = "season"              # "season" | "age"
    ):
        import numpy as np
        import pandas as pd
    
        if self.players_df.empty:
            raise ValueError("Load data first (show.load(...)).")
    
        if seasons is None:
            if self.season is None:
                raise ValueError("Provide seasons=... or set context with set_context(..., season=...).")
            seasons = list(range(int(self.season) - 4, int(self.season) + 1))
        seasons = sorted({int(s) for s in seasons})
    
        cur = self.players_df.dropna(subset=[self._imp_col]).copy()
    
        # pick players by current season contribution (prefer player_id)
        use_id = "player_id" in cur.columns and cur["player_id"].notna().any()
        if use_id:
            cur_agg = (cur.groupby("player_id", as_index=False)[self._imp_col]
                         .sum().sort_values(self._imp_col, ascending=False))
            top_keys = cur_agg["player_id"].head(int(top_k)).tolist()
            id2name = (cur.dropna(subset=["player_id","player_name"])
                         .drop_duplicates("player_id")
                         .set_index("player_id")["player_name"].to_dict())
            labels_all = [id2name.get(k, f"ID {k}") for k in top_keys]
            key_field = "player_id"
        else:
            cur_agg = (cur.groupby("player_name", as_index=False)[self._imp_col]
                         .sum().sort_values(self._imp_col, ascending=False))
            top_keys = cur_agg["player_name"].head(int(top_k)).tolist()
            labels_all = list(top_keys)
            key_field = "player_name"
    
        # helper: one value per player per season
        def _value_for(df_s: pd.DataFrame, key):
            if df_s.empty:
                return np.nan
            if key_field not in df_s.columns:
                if key_field == "player_id":
                    return np.nan
                mask = (df_s["player_name"] == key)
            else:
                mask = (df_s[key_field] == key)
            vals = pd.to_numeric(df_s.loc[mask, self._imp_col], errors="coerce").dropna()
            if vals.empty: return np.nan
            return float(vals.mean() if agg == "mean" else vals.sum())
    
        # build x (season or age) and Y per player
        X = np.array(seasons, dtype=float)
        Ys, labels = [], []
    
        for k, lab in zip(top_keys, labels_all):
            series, x_vals = [], []
    
            for s in seasons:
                df_s = self.fetch_scope(country=self.country, league=self.league, season=s)
                if df_s.empty:
                    val = np.nan
                else:
                    # add age if requested and present
                    if x_axis == "age":
                        if "age_at_season_start" not in df_s.columns and "birth_date" in df_s.columns:
                            # try to compute age from birth_date
                            start_dt = pd.Timestamp(datetime.date(int(s), 8, 1), tz="UTC")
                            b = pd.to_datetime(df_s["birth_date"], errors="coerce", utc=True)
                            df_s = df_s.copy()
                            df_s["age_at_season_start"] = ((start_dt - b).dt.days / 365.25).astype(float)
                        # we still use contribution as y; x will be season (age axis is only for ticks)
                    val = _value_for(df_s, k)
    
                series.append(val)
                x_vals.append(s)
    
            series = np.array(series, dtype=float)
            # keep only players with >1 valid points
            if np.isfinite(series).sum() >= 2:
                Ys.append(series)
                labels.append(lab)
    
        if not Ys:
            raise ValueError("No players with more than one observation across the chosen seasons.")
    
        # axis label choice
        xlab = "Season" if x_axis == "season" else "Age at season start (years)"
    
        title = f"{self._fmt_scope_title('Scope')}: Top {len(labels)} Players Across Seasons"
        subtitle = "Each line = one player; only players with >1 observation shown."
    
        fig = self.graphs.multiline(
            x=X, ys=Ys, labels=labels,
            title=title, subtitle=subtitle,
            figsize=(10, 6), integer_xticks=True
        )
        # override xlabel if using age axis
        fig.axes[0].set_xlabel(xlab, fontsize=self.style.label_fontsize)
        return self._finish(fig, "multiline_top_players")


    # def mod_multiline_top_players_over_seasons(
    #     self,
    #     *,
    #     top_k: int = 8,
    #     seasons: list[int] | None = None,   # None → [season-4 .. season]
    #     agg: str = "sum",                   # aggregate if a player has multiple rows/teams in a season
    # ):
    #     """
    #     Multiline: top-K current-season players, values across seasons (previous → current).
    #       - Uses self.players_df as the current season.
    #       - Resolves players by player_id when available; falls back to names.
    #       - agg: "sum" or "mean" across duplicate rows per player per season.
    #     """
    #     import numpy as np
    #     import pandas as pd
    
    #     if self.players_df.empty:
    #         raise ValueError("Load data first (show.load(...)).")
    #     if seasons is None:
    #         if self.season is None:
    #             raise ValueError("Provide seasons=... or set context with set_context(..., season=...).")
    #         seasons = list(range(int(self.season) - 4, int(self.season) + 1))
    #     seasons = sorted(set(int(s) for s in seasons))
    
    #     cur = self.players_df.dropna(subset=[self._imp_col]).copy()
    
    #     # pick players by current season contribution (use player_id if possible)
    #     use_id = "player_id" in cur.columns and cur["player_id"].notna().any()
    #     if use_id:
    #         cur_agg = (cur.groupby("player_id", as_index=False)[self._imp_col]
    #                       .sum()
    #                       .sort_values(self._imp_col, ascending=False))
    #         top_keys = cur_agg["player_id"].head(int(top_k)).tolist()
    #         id2name = (cur.dropna(subset=["player_id","player_name"])
    #                       .drop_duplicates("player_id")
    #                       .set_index("player_id")["player_name"].to_dict())
    #         labels = [id2name.get(k, f"ID {k}") for k in top_keys]
    #         key_field = "player_id"
    #     else:
    #         cur_agg = (cur.groupby("player_name", as_index=False)[self._imp_col]
    #                       .sum()
    #                       .sort_values(self._imp_col, ascending=False))
    #         top_keys = cur_agg["player_name"].head(int(top_k)).tolist()
    #         labels = list(top_keys)
    #         key_field = "player_name"
    
    #     # helper to aggregate a season df to one value per player key
    #     def _value_for(df_s: pd.DataFrame, key):
    #         if df_s.empty: 
    #             return np.nan
    #         col = self._imp_col
    #         if key_field not in df_s.columns:
    #             # fallback to names if ids missing
    #             if key_field == "player_id":
    #                 return np.nan
    #             vals = pd.to_numeric(df_s.loc[df_s["player_name"] == key, col], errors="coerce")
    #         else:
    #             vals = pd.to_numeric(df_s.loc[df_s[key_field] == key, col], errors="coerce")
    #         if vals.empty:
    #             return np.nan
    #         if agg == "mean":
    #             return float(vals.mean())
    #         return float(vals.sum())
    
    #     # build Y series per player across requested seasons
    #     Ys = []
    #     for k in top_keys:
    #         series = []
    #         for s in seasons:
    #             df_s = self.fetch_scope(country=self.country, league=self.league, season=s)
    #             series.append(_value_for(df_s, k))
    #         Ys.append(np.array(series, dtype=float))
    
    #     # draw
    #     title = f"{self._fmt_scope_title('Scope')}: Top {top_k} Players Across Seasons"
    #     subtitle = "Each line = one player; value per season."
    #     fig = self.graphs.multiline(seasons, Ys, labels, title=title, subtitle=subtitle, figsize=(10, 6))
    #     return self._finish(fig, "multiline_top_players")

    # def mod_player_trajectories(self,long_df,players):
    #     x=long_df["season"].unique()
    #     ys=[long_df[long_df["player_name"]==p][self._imp_col].values for p in players]
    #     fig=self.graphs.multiline(x,ys,labels=players,
    #         title="Player Trajectories Across Seasons",
    #         subtitle="Each line: one player’s contributions over time")
    #     return self._finish(fig,"multiline")

    # def mod_position_small_multiples(self):
    #     dfs=[self.players_df[self.players_df["position"]==p][["minutes_played","impact"]].rename(columns={"minutes_played":"x","impact":"y"}) for p in self.players_df["position"].unique()]
    #     fig=self.graphs.small_multiples(dfs,titles=self.players_df["position"].unique(),
    #         title=f"{self.league} {self.season}: Contributions vs Minutes by Position",
    #         subtitle="Each panel shows one position")
    #     return self._finish(fig,"small_multiples")

    def mod_small_multiples_top_teams_scatter(
        self,
        *,
        seasons: list[int] | None = None,    # default → [season-1, season]
        n_panels: int = 4,                    # must be one of {2,3,4,6,8}
        position_filter: str = "Midfielders",
        seed: int = 13,
        agg: str = "sum"                      # "sum" | "mean" per season
    ):
        """
        Small-multiples timeseries panels (connected lines) for players.
        Example: 4 random midfielders across the two seasons.
    
        - Picks players present in ALL requested seasons and matching `position_filter` in the current season.
        - One panel per player; faint lines of others shown for context.
        - Allowed n_panels: {2,3,4,6,8}.
        """
        import numpy as np
        import pandas as pd
        rng = np.random.default_rng(int(seed))
    
        if seasons is None:
            if self.season is None:
                raise ValueError("Provide seasons=... or set context with set_context(..., season=...).")
            seasons = [int(self.season) - 1, int(self.season)]
        seasons = [int(s) for s in seasons]
        if len(seasons) < 2:
            raise ValueError("Need at least two seasons.")
    
        if n_panels not in {2,3,4,6,8}:
            raise ValueError("n_panels must be one of {2,3,4,6,8}.")
    
        # Position bucketing
        def _bucket(pos: str) -> str:
            p = (pos or "").lower()
            if any(k in p for k in ["d","def","back","cb","rb","lb","df"]): return "Defenders"
            if any(k in p for k in ["m","mid","cm","dm","am","wing","mf"]): return "Midfielders"
            if any(k in p for k in ["f","fw","st","cf","striker","forward","attacker"]): return "Strikers"
            return "Other"
    
        # Load each season and build {season: df}
        dfs = {}
        for s in seasons:
            df_s = self.fetch_scope(country=self.country, league=self.league, season=s)
            if df_s.empty:
                raise ValueError(f"No data for season {s}.")
            df_s = df_s.copy()
            df_s["_bucket"] = df_s.get("position","").astype(str).map(_bucket)
            dfs[s] = df_s
    
        # Use player_id if possible across seasons
        use_id = all("player_id" in dfs[s].columns for s in seasons)
        key_field = "player_id" if use_id else "player_name"
    
        # Candidates: present in all seasons & match pos bucket in the latest season
        latest = max(seasons)
        cur_df = dfs[latest]
        wanted = set(cur_df.loc[cur_df["_bucket"] == position_filter, key_field].dropna().astype(object))
    
        for s in seasons:
            have = set(dfs[s][key_field].dropna().astype(object))
            wanted &= have
    
        if not wanted:
            raise ValueError(f"No {position_filter} found in all requested seasons.")
    
        wanted = list(wanted)
        if len(wanted) < n_panels:
            n_panels = len(wanted)
    
        sample_keys = rng.choice(wanted, size=n_panels, replace=False).tolist()
    
        # Map id → name for titles
        if use_id:
            id2name = (cur_df.dropna(subset=["player_id","player_name"])
                            .drop_duplicates("player_id")
                            .set_index("player_id")["player_name"].to_dict())
        else:
            id2name = {}
    
        # Build panels: y values per season for each selected player
        panels = []
        for k in sample_keys:
            ys = []
            for s in seasons:
                df_s = dfs[s]
                mask = (df_s[key_field] == k)
                vals = pd.to_numeric(df_s.loc[mask, self._imp_col], errors="coerce").dropna()
                y = float(vals.mean() if agg == "mean" else vals.sum()) if not vals.empty else np.nan
                ys.append(y)
            title = id2name.get(k, str(k))
            panels.append({"title": title, "y": ys})  # color optional: default cycle
    
        title = f"{self._fmt_scope_title('Scope')}: {n_panels} {position_filter} (lines across seasons)"
        subtitle = "Each panel highlights one player; faint lines show the other selected players."
    
        fig = self.graphs.small_multiples_lines(
            x=seasons, panels=panels,
            title=title, subtitle=subtitle, figsize=None  # auto by layout
        )
        return self._finish(fig, "small_multiples_players_lines")
 

    def mod_lorenz_contributions(
        self,
        *,
        use_abs: bool = True,           # Lorenz usually on magnitudes
        min_minutes: float = 0.0
    ):
        """
        Lorenz curve of player contributions for the current scope.
        Adds Gini coefficient to the subtitle.
        """
        import numpy as np
        import pandas as pd
    
        if self.players_df.empty:
            raise ValueError("Load data first.")
    
        df = self.players_df.copy()
        if min_minutes > 0 and "minutes_played" in df.columns:
            df = df[pd.to_numeric(df["minutes_played"], errors="coerce").fillna(0.0) >= float(min_minutes)]
    
        vals = pd.to_numeric(df[self._imp_col], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            raise ValueError("No contribution values to plot.")
    
        if use_abs:
            vals = np.abs(vals)
    
        # Gini from Lorenz polygon
        v = np.sort(vals)
        if v.sum() == 0:
            gini = 0.0
        else:
            cum = np.cumsum(v)
            cum = np.insert(cum, 0, 0.0)
            p = np.linspace(0.0, 1.0, len(cum))
            L = cum / cum[-1]
            auc = np.trapz(L, p)   # area under Lorenz
            gini = 1.0 - 2.0 * auc
    
        title = f"{self._fmt_scope_title('Scope')}: Lorenz Curve of Contributions"
        subtitle = f"Players sorted by contribution share. Gini = {gini:.3f}"
    
        fig = self.graphs.lorenz(vals, title=title, subtitle=subtitle, figsize=(6,6))
        return self._finish(fig, "lorenz_contributions")

    # def mod_concentration_curve(self):
    #     vals=self.players_df[self._imp_col].abs().to_numpy()
    #     fig=self.graphs.lorenz(vals,
    #         title=f"{self.league} {self.season}: Concentration of Contributions",
    #         subtitle="Cumulative share of players vs cumulative share of contributions")
    #     return self._finish(fig,"lorenz")

# ----------------------------- Run module -----------------------------
if __name__ == "__main__":
    country_name = "Germany"
    league_name  = "Bundesliga"
    season       = 2023

    show = OffBalanceShowcase(
        db_path="analysis_results.db",
        estimator="ridge", ridge_alpha=1.0,
        save_plot=False, show_plot=True
    )

    # Choose scope here (NOT in __init__)
    show.load(country=country_name, league=league_name, season=season)

    # (Optional) enrich with profiles/transfers (needs season context set above)
    show.merge_profiles_and_transfers(
        profiles_db="player_profiles.db",
        transfers_db="player_transfer.db",
        season_start_month=8, season_start_day=1,
        endseason_cutoff_month=10, endseason_cutoff_day=1,
        mid_start_month=12, mid_start_day=1,
        mid_end_month=3, mid_end_day=1
    )

    
    #"""
    # now run any module
    out = show.mod_contrib_vs_fte_multi()
    show.mod_contrib_vs_fte_by_nationality(top_n=10, include_other=True, min_minutes=180)
    show.mod_contrib_vs_age(by_position=True, min_minutes=180, anchor_age_min=30, breakout_age_max=22,
                            add_regression=True, add_ci="prediction")
    show.mod_contrib_vs_fte_by_transfer_status(min_minutes=180, highlight_top_k_each=4)
    # Now run modules against current scope
    out2 = show.mod_top_vs_bottom(k=5, show_values=True, show_club=True)
    print("Tornado saved to:", out2)

    out3 = show.mod_team_ranking()
    print("Ranking chart saved to:", out3)

    # Flexible comparison: fetch any previous/alternate scope
    comp_df = show.fetch_scope(country=country_name, league=league_name, season=season-1)
    out4 = show.mod_season_improve(comp_df)
    # print("Slope chart saved to:", out4) 
    
    out_decline = show.mod_season_decline(comp_df )
    show.set_context(country="Germany", league="Bundesliga")
    
    # Compare two seasons for the same league
    out = show.mod_position_buckets_by_season([2022, 2023],
                                positions=("Defenders","Midfielders","Strikers"))

    out = show.mod_team_buckets_horizontal(min_players_per_team=5, symmetric=True)
    # # Small multiples: top 6 teams by median contribution
    
    # Multiline: top players over seasons, integers on x-axis, only players with >1 points
    out1 = show.mod_multiline_top_players_over_seasons(top_k=8, seasons=None, x_axis="season")
    
    # Small-multiples: 4 random midfielders across last two seasons, lines with faint context
    out2 = show.mod_small_multiples_top_teams_scatter(seasons=None, n_panels=4, position_filter="Midfielders")
    
    # Lorenz: Gini included
    out3 = show.mod_lorenz_contributions(use_abs=True, min_minutes=180)

