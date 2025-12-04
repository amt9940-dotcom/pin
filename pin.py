# file: pin_group_optimizer_best_with_mid.py
"""
Best-pin optimizer (exhaustive) with fixed 3×5 grid, fixed site diameters,
and eccentric clevis load at (0, 3.25). For each tested pin count K, reports:
- WORST combination
- MID combination (closest to midpoint between worst and best P_fail)
- BEST combination

Coordinate system & fixed sites (index → (x,y) in inches)
---------------------------------------------------------
1:(1.5,0),  2:(2.0,0),  3:(2.5,0),
4:(1.5,0.5),5:(2.0,0.5),6:(2.5,0.5),
7:(1.5,1),  8:(2.0,1),  9:(2.5,1),
10:(1.5,1.5),11:(2.0,1.5),12:(2.5,1.5),
13:(1.5,2), 14:(2.0,2), 15:(2.5,2)

Fixed pin diameters at sites (in)
---------------------------------
0.188 → {1, 3, 6, 7, 15}
0.125 → {2, 4, 8, 9, 10, 12, 13, 14}
0.250 → {5, 11}

How to choose how many pins to test
-----------------------------------
- Set NUMBER_OF_PINS below (int) to force exactly that many pins.
- Or leave NUMBER_OF_PINS=None and pass:
    --n K          → exactly K
    --ns K1 K2 ... → set of K
    --n-range A-B  → all K in [A..B]

Default mode = 'eccentric' (clevis at 0,3.25; vertical DOWN).
Also supports: 'inline' (equal-share only), 'pm' (fixed M).
"""

from __future__ import annotations

import argparse
import csv
import itertools as it
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# EDIT THIS to force exactly how many pins are tested; set to None to use CLI.
NUMBER_OF_PINS: Optional[int] = None
# ──────────────────────────────────────────────────────────────────────────────

Point = Tuple[float, float]

@dataclass(frozen=True)
class Site:
    idx: int
    x: float
    y: float
    d: float  # fixed diameter (in)

# ----------------------------- CLI & setup ------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best/worst/mid pin-group optimizer with fixed sites and diameters.")
    p.add_argument("--mode", choices=["eccentric", "inline", "pm"], default="eccentric",
                   help="eccentric (default): clevis load at (load-x,load-y); inline: equal share; pm: fixed M.")
    p.add_argument("--tau", type=float, default=6000.0, help="Allowable pin shear stress (psi).")
    shear = p.add_mutually_exclusive_group()
    shear.add_argument("--double-shear", dest="double_shear", action="store_true", help="A=πd²/2 (default).")
    shear.add_argument("--single-shear", dest="double_shear", action="store_false", help="A=πd²/4.")
    p.set_defaults(double_shear=True)

    # Pin count control when NUMBER_OF_PINS is None
    p.add_argument("--n", type=int, help="Exactly K pins.")
    p.add_argument("--ns", type=int, nargs="+", help="Specific K values.")
    p.add_argument("--n-range", type=str, default="3-8", help="Range A-B.")

    # Physics
    p.add_argument("--M", type=float, default=0.0, help="Fixed torsional moment (lb·in) for mode=pm.")
    p.add_argument("--cw-tangent", action="store_true", help="mode=pm only; positive M induces clockwise tangential.")
    p.add_argument("--load-x", type=float, default=0.0, help="Clevis x (in) [eccentric].")
    p.add_argument("--load-y", type=float, default=3.25, help="Clevis y (in) [eccentric].")

    # Constraints/runtime
    p.add_argument("--min-spacing", type=float, default=0.0, help="Min center-to-center spacing (in).")
    p.add_argument("--limit-per-size", type=int, default=0, help="Cap groups per K (0=exhaustive; leave 0 for truth).")

    # Output
    p.add_argument("--outdir", type=str, default="out", help="CSV directory.")
    p.add_argument("--quiet", action="store_true", help="Hide console tables.")
    return p.parse_args()

def build_sites() -> List[Site]:
    idx_to_xy = {
        1:(1.5,0.0),  2:(2.0,0.0),  3:(2.5,0.0),
        4:(1.5,0.5),  5:(2.0,0.5),  6:(2.5,0.5),
        7:(1.5,1.0),  8:(2.0,1.0),  9:(2.5,1.0),
        10:(1.5,1.5), 11:(2.0,1.5), 12:(2.5,1.5),
        13:(1.5,2.0), 14:(2.0,2.0), 15:(2.5,2.0),
    }
    d_map: Dict[int, float] = {}
    for i in [1, 3, 6, 7, 15]: d_map[i] = 0.188
    for i in [2, 4, 8, 9, 10, 12, 13, 14]: d_map[i] = 0.125
    for i in [5, 11]: d_map[i] = 0.25

    sites: List[Site] = []
    for i in range(1, 16):
        x, y = idx_to_xy[i]; d = d_map[i]
        sites.append(Site(idx=i, x=x, y=y, d=d))
    # Sanity
    assert abs(sites[0].x-1.5)<1e-9 and abs(sites[0].y-0.0)<1e-9
    assert sites[-1].idx==15 and abs(sites[-1].x-2.5)<1e-9 and abs(sites[-1].y-2.0)<1e-9
    return sites

# ----------------------------- Math helpers -----------------------------------

def centroid(points: Sequence[Point]) -> Point:
    n = len(points)
    sx = sum(x for x,_ in points); sy = sum(y for _,y in points)
    return (sx/n, sy/n) if n else (0.0, 0.0)

def shifted(points: Sequence[Point]) -> Tuple[List[Point], Point]:
    c = centroid(points)
    return ([(x-c[0], y-c[1]) for (x,y) in points], c)

def shear_area(d: float, double_shear: bool) -> float:
    return math.pi * d*d / (2.0 if double_shear else 4.0)

def capacity_lbf(d: float, tau: float, double_shear: bool) -> float:
    return tau * shear_area(d, double_shear)

def spacing_ok(points: Sequence[Point], min_spacing: float) -> bool:
    if min_spacing <= 0 or len(points) < 2: return True
    s2 = min_spacing*min_spacing
    for i in range(len(points)):
        xi, yi = points[i]
        for j in range(i+1, len(points)):
            dx = xi - points[j][0]; dy = yi - points[j][1]
            if dx*dx + dy*dy < s2 - 1e-12: return False
    return True

def tangential_unit(x: float, y: float) -> Tuple[float, float]:
    rho = math.hypot(x, y)
    if rho == 0.0: return (0.0, 0.0)
    ang = math.atan2(y, x)
    return (-math.sin(ang), math.cos(ang))  # CCW unit tangent

# --- capacities ---

def pfail_inline(diams: Sequence[float], tau: float, double_shear: bool) -> float:
    n = len(diams)
    if n == 0: return 0.0
    cmin = min(capacity_lbf(d, tau, double_shear) for d in diams)
    return n * cmin

def pfail_pm(points0: Sequence[Point], diams: Sequence[float], tau: float, double_shear: bool,
             M: float, cw_tangent: bool) -> float:
    n = len(diams)
    if n == 0: return 0.0
    rhos = [math.hypot(x, y) for (x,y) in points0]
    sum_r2 = sum(r*r for r in rhos)
    k = 0.0 if sum_r2==0.0 else M/sum_r2
    caps = [capacity_lbf(d, tau, double_shear) for d in diams]
    limits: List[float] = []
    for (x,y), rho, Ci in zip(points0, rhos, caps):
        if rho==0.0 or k==0.0:
            Fmx, Fmy = 0.0, 0.0
        else:
            ang = math.atan2(y, x)
            if cw_tangent:
                tx, ty = math.sin(ang), -math.cos(ang)
            else:
                tx, ty = tangential_unit(x, y)
            Ft = k * rho
            Fmx, Fmy = Ft*tx, Ft*ty
        a = 1.0; b = -2.0*Fmy; c = Fmx*Fmx + Fmy*Fmy - Ci*Ci
        disc = b*b - 4*a*c
        if disc < 0: limits.append(0.0); continue
        y1 = (-b + math.sqrt(disc)) / (2*a)
        y2 = (-b - math.sqrt(disc)) / (2*a)
        ystar = max([y for y in (y1,y2) if y >= 0.0], default=0.0)
        limits.append(n * ystar)
    return min(limits) if limits else 0.0

def pfail_eccentric(points: Sequence[Point], diams: Sequence[float], tau: float, double_shear: bool,
                    load_x: float, load_y: float) -> Tuple[float, Dict[str, Any]]:
    """
    Vertical DOWN force at (load_x, load_y).  M/P = -(xL - xC).
    Returns (P_fail, debug_dict).
    """
    n = len(points)
    pts0, (cx, cy) = shifted(points)
    rhos = [math.hypot(x, y) for (x,y) in pts0]
    sum_r2 = sum(r*r for r in rhos)
    if sum_r2 == 0.0:
        return pfail_inline(diams, tau, double_shear), {"cx":cx,"cy":cy,"sum_r2":sum_r2,"M_over_P":0.0}

    M_over_P = -(load_x - cx)  # only x lever arm matters for vertical load
    k_per_P = M_over_P / sum_r2

    ai: List[float] = []; bi: List[float] = []
    for (x,y), rho in zip(pts0, rhos):
        if rho == 0.0:
            ai.append(0.0); bi.append(0.0)
        else:
            tx, ty = tangential_unit(x, y)
            Ft_per_P = k_per_P * rho
            ai.append(Ft_per_P * tx)
            bi.append(Ft_per_P * ty)

    caps = [capacity_lbf(d, tau, double_shear) for d in diams]
    n_pins = len(points)
    limits: List[float] = []
    per_pin_S: List[float] = []
    for Ci, a_i, b_i in zip(caps, ai, bi):
        S = math.hypot(a_i, b_i - 1.0/n_pins)
        per_pin_S.append(S)
        if S <= 0.0: limits.append(0.0)
        else:        limits.append(Ci / S)

    debug = {"cx":cx, "cy":cy, "sum_r2":sum_r2, "M_over_P":M_over_P,
             "k_per_P":k_per_P, "a":ai, "b":bi, "S":per_pin_S}
    return (min(limits) if limits else 0.0, debug)

# ------------------------------- I/O ------------------------------------------

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def write_ranking_csv(path: str, size_k: int, rows: List[Dict[str, Any]]) -> str:
    ensure_outdir(path)
    fp = os.path.join(path, f"ranking_size_{size_k}.csv")
    cols = ["size","group_indices","group_points","diameters","P_fail_lbf","centroid_x","centroid_y","sum_r2"]
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)
    return fp

def write_winners_csv(path: str, winners: List[Dict[str, Any]]) -> str:
    ensure_outdir(path)
    fp = os.path.join(path, "winners_only.csv")
    cols = ["size","type","group_indices","group_points","diameters","P_fail_lbf"]
    with open(fp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in winners: w.writerow(r)
    return fp

def print_table(title: str, rows: List[Dict[str, Any]], limit: Optional[int] = None) -> None:
    print(title); print("-"*len(title))
    if not rows: print("(no rows)\n"); return
    if limit is not None: rows = rows[:limit]
    print("{:>4}  {:<20}  {:>12}  {:<24}".format("size","group_indices","P_fail_lbf","diameters"))
    for r in rows:
        print("{:>4}  {:<20}  {:>12.1f}  {:<24}".format(
            r["size"], r["group_indices"][:20], r["P_fail_lbf"], r["diameters"][:24]))
    print()

def print_detail(label: str, row: Dict[str, Any], mode: str, load_x: float, load_y: float) -> None:
    print(f"{label} combination (index: (x,y) d):")
    idxs = [int(s) for s in row["group_indices"].split(",")]
    coords = [tuple(map(float, p.strip("()").split(","))) for p in row["group_points"].split(";")]
    diams = [float(s) for s in row["diameters"].split(",")]
    for j,(i,(x,y),d) in enumerate(zip(idxs, coords, diams), start=1):
        print(f"  {j:>2}. {i}: ({x:.3f},{y:.3f})  d={d:.5f} in")
    print(f"  centroid C=({row['centroid_x']:.3f},{row['centroid_y']:.3f}),  Σρ²={row['sum_r2']:.6f} in²")
    if mode == "eccentric":
        Lx = load_x - row['centroid_x']
        print(f"  M/P = -(xL-xC) = -({load_x:.3f}-{row['centroid_x']:.3f}) = {-Lx:.6f} lb·in/lb")
    print(f"  P_fail ≈ {row['P_fail_lbf']:.1f} lbf\n")

# ------------------------------ main logic ------------------------------------

def parse_n_range(text: str, nmax: int) -> List[int]:
    if "-" in text:
        a,b = text.split("-"); lo,hi = int(a), int(b)
    else:
        lo = hi = int(text)
    if lo < 1 or hi < lo: raise SystemExit("Bad --n-range")
    return [n for n in range(lo, min(hi, nmax)+1)]

def parse_sizes(args: argparse.Namespace, nmax: int) -> List[int]:
    if getattr(args,"n",None) is not None:
        if not (1 <= args.n <= nmax): raise SystemExit(f"--n must be in [1,{nmax}]")
        return [args.n]
    if getattr(args,"ns",None):
        uniq = sorted({n for n in args.ns if 1 <= n <= nmax})
        if not uniq: raise SystemExit(f"--ns must contain values in [1,{nmax}]")
        return uniq
    return parse_n_range(args.n_range, nmax)

def main() -> None:
    args = parse_args()
    sites = build_sites()
    n_sites = len(sites)

    # Decide sizes
    if NUMBER_OF_PINS is not None:
        if not (1 <= NUMBER_OF_PINS <= n_sites): raise SystemExit(f"NUMBER_OF_PINS must be in [1,{n_sites}]")
        sizes = [NUMBER_OF_PINS]
        print(f"[info] Using NUMBER_OF_PINS = {NUMBER_OF_PINS}")
    else:
        sizes = [k for k in parse_sizes(args, n_sites) if 1 <= k <= n_sites]
        print(f"[info] Evaluating pin counts: {sizes}")

    winners_rollup: List[Dict[str, Any]] = []
    global_best: Optional[Dict[str, Any]] = None

    for k in sizes:
        ranking: List[Dict[str, Any]] = []
        best_row: Optional[Dict[str, Any]] = None

        combos = it.combinations(range(n_sites), k)
        if args.limit_per_size and args.limit_per_size > 0:
            combos = it.islice(combos, args.limit_per_size)  # only if you accept approximate

        for idxs in combos:
            chosen = [sites[i] for i in idxs]
            pts = [(s.x, s.y) for s in chosen]
            if not spacing_ok(pts, args.min_spacing): continue

            pts0, (cx, cy) = shifted(pts)
            sum_r2 = sum((x*x + y*y) for (x,y) in pts0)
            diams = [s.d for s in chosen]

            if args.mode == "inline":
                Pfail = pfail_inline(diams, args.tau, args.double_shear)
            elif args.mode == "pm":
                Pfail = pfail_pm(pts0, diams, args.tau, args.double_shear, args.M, args.cw_tangent)
            else:  # eccentric (default)
                Pfail, _ = pfail_eccentric(pts, diams, args.tau, args.double_shear, args.load_x, args.load_y)

            row = {
                "size": k,
                "group_indices": ",".join(str(s.idx) for s in chosen),
                "group_points": ";".join(f"({s.x:.3f},{s.y:.3f})" for s in chosen),
                "diameters": ",".join(f"{s.d:.5f}" for s in chosen),
                "P_fail_lbf": Pfail,
                "centroid_x": cx, "centroid_y": cy, "sum_r2": sum_r2,
            }
            ranking.append(row)

            if (best_row is None
                or Pfail > best_row["P_fail_lbf"]
                or (Pfail == best_row["P_fail_lbf"]
                    and row["group_indices"] < best_row["group_indices"])):
                best_row = row

        ranking.sort(key=lambda r: (r["P_fail_lbf"], r["group_indices"]))
        fpath = write_ranking_csv(args.outdir, k, ranking)

        if ranking:
            worst = ranking[0]
            best  = ranking[-1]
            mid_target = 0.5*(worst["P_fail_lbf"] + best["P_fail_lbf"])
            mid = min(ranking, key=lambda r: abs(r["P_fail_lbf"] - mid_target))

            winners_rollup.append({"size": k, "type":"worst", **{kk: worst[kk] for kk in ["group_indices","group_points","diameters","P_fail_lbf"]}})
            winners_rollup.append({"size": k, "type":"mid",   **{kk: mid[kk]   for kk in ["group_indices","group_points","diameters","P_fail_lbf"]}})
            winners_rollup.append({"size": k, "type":"best",  **{kk: best[kk]  for kk in ["group_indices","group_points","diameters","P_fail_lbf"]}})

            if not args.quiet:
                print_table(f"Size {k} — worst", [worst])
                print_table(f"Size {k} — mid (≈ midpoint)", [mid])
                print_table(f"Size {k} — best", [best])
                print(f"[saved] full ranking → {fpath}\n")
                print_detail("WORST", worst, args.mode, args.load_x, args.load_y)
                print_detail("MID",   mid,   args.mode, args.load_x, args.load_y)
                print_detail("BEST",  best,  args.mode, args.load_x, args.load_y)

            # Track global best if multiple K given
            if (global_best is None
                or best["P_fail_lbf"] > global_best["P_fail_lbf"]
                or (best["P_fail_lbf"] == global_best["P_fail_lbf"]
                    and (best["size"], best["group_indices"]) < (global_best["size"], global_best["group_indices"]))):
                global_best = {"size": best["size"], **{kk: best[kk] for kk in ["group_indices","group_points","diameters","P_fail_lbf","centroid_x","centroid_y","sum_r2"]}}
        else:
            if not args.quiet:
                print_table(f"Size {k} — no valid combinations", [])

    winners_path = write_winners_csv(args.outdir, winners_rollup)
    if not args.quiet:
        print(f"[saved] winners summary → {winners_path}")

    if global_best and len(sizes) > 1 and not args.quiet:
        print("GLOBAL BEST across all tested pin counts:")
        print(f"  K = {global_best['size']}")
        print(f"  indices: {global_best['group_indices']}")
        print(f"  points : {global_best['group_points']}")
        print(f"  diams  : {global_best['diameters']}")
        print(f"  P_fail ≈ {global_best['P_fail_lbf']:.1f} lbf")

if __name__ == "__main__":
    main()
