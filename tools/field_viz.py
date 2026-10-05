"""Visualise the learned obstacle field, the DS and the trajectory.

Runs a controller, captures the frame of minimum obstacle clearance, and writes a
self-contained Plotly HTML (``out/field_frame.html`` + ``.json``) showing:

* the learned potential ``U_obs`` (heatmap) and vector field ``F_obs=−∇U``;
* the nominal DS ``f_DS`` and the modulated DS;
* the LiDAR points, the obstacle geometry and the flown trajectory.

    python tools/field_viz.py --controller field_snn --scene boxes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drone6dof.cli import make_controller
from drone6dof.config import CONTROL_LIMIT, INIT_STATE, preset_scene
from drone6dof.dynamics import NumpyPlantBackend
from drone6dof.field import FieldConfig, PotentialBasis, eval_grid, modulate_ds, nominal_ds
from drone6dof.sim import Simulation


def _obstacles_json(scene) -> list:
    out = []
    for o in getattr(scene, "obstacles", ()) or ():
        if getattr(o, "kind", "cylinder") == "cylinder":
            out.append({"kind": "cylinder", "center": list(o.center), "radius": float(o.radius)})
        else:
            out.append({"kind": "box", "center": list(o.center),
                        "half": list(o.half), "angle": float(o.angle)})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Field / DS / trajectory visualisation")
    ap.add_argument("--controller", default="field_snn")
    ap.add_argument("--scene", default="pillar")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args(argv)

    scene = preset_scene(args.scene, goal=(0.0, 0.0, 2.5))
    backend = NumpyPlantBackend(heading_target=scene.goal_np)
    ctrl = make_controller(args.controller, scene, CONTROL_LIMIT)
    sim = Simulation(backend, ctrl, scene, steps=args.steps, initial_state=INIT_STATE)

    best = None
    while not sim.done:
        sim.step()
        pos = sim.position
        clear = scene.clearance(pos)
        if best is None or clear < best["clearance"]:
            best = {
                "clearance": float(clear),
                "state": np.asarray(sim.state, dtype=float).copy(),
                "scan": None if ctrl.last_scan is None else np.asarray(ctrl.last_scan, float).copy(),
            }

    state = best["state"]
    cfg = getattr(ctrl, "cfg", FieldConfig())
    basis = PotentialBasis(cfg)
    coeffs = getattr(ctrl, "last_coeffs", None)
    centers = getattr(ctrl, "last_centers", None)
    if coeffs is None or centers is None:      # non-field controller: use teacher coeffs
        from drone6dof.field import teacher_coeffs
        coeffs = teacher_coeffs(scene, state, cfg)
        centers = basis.centers(state, scene.goal_np)

    extent = 2.6
    X, Y, U, Fx, Fy = eval_grid(basis, coeffs, centers, state[:2], extent=extent, n=41)
    goal = scene.goal_np
    stride = 4
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    v_nom = np.asarray([nominal_ds(p - goal, cfg.gains_dict()) for p in P])
    v_des = np.asarray([modulate_ds(vn, np.array([-Fx.ravel()[i], -Fy.ravel()[i], 0.0]), cfg)
                        for i, vn in enumerate(v_nom)])
    traj = sim.trajectory()

    payload = {
        "scene": args.scene,
        "controller": args.controller,
        "goal": [float(v) for v in goal],
        "state": [float(v) for v in state[:3]],
        "clearance_min": best["clearance"],
        "grid": {"x0": float(state[0] - extent), "x1": float(state[0] + extent),
                 "y0": float(state[1] - extent), "y1": float(state[1] + extent), "n": 41},
        "U": np.round(U, 4).tolist(),
        "F": {"x": np.round(X, 3).tolist(), "y": np.round(Y, 3).tolist(),
              "fx": np.round(Fx, 3).tolist(), "fy": np.round(Fy, 3).tolist()},
        "ds": {"x": X.ravel()[::stride].round(3).tolist(),
               "y": Y.ravel()[::stride].round(3).tolist(),
               "nom_u": v_nom[::stride, 0].round(3).tolist(),
               "nom_v": v_nom[::stride, 1].round(3).tolist(),
               "mod_u": v_des[::stride, 0].round(3).tolist(),
               "mod_v": v_des[::stride, 1].round(3).tolist()},
        "trajectory": np.round(traj[:, :3], 3).tolist(),
        "lidar": None if best["scan"] is None else {
            "angles": [float(a) for a in ctrl.scan_angles],
            "ranges": [float(v) for v in best["scan"]],
        },
        "obstacles": _obstacles_json(scene),
        "centers": np.round(centers, 3).tolist(),
        "coeffs": np.round(coeffs, 4).tolist(),
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "field_frame.json").write_text(json.dumps(payload))
    (outdir / "field_frame.html").write_text(_HTML.replace("__DATA__", json.dumps(payload)))
    print(f"wrote {outdir / 'field_frame.html'}  (min clearance {best['clearance']:+.3f} m, "
          f"controller {args.controller})")
    return 0


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>drone6dof field</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{background:#0b0f14;color:#c9d1d9;font:13px sans-serif;margin:0}
#c{width:100vw;height:100vh}</style></head><body><div id="c"></div>
<script>
const D = __DATA__;
const seg = (x0,y0,dx,dy)=>({x:[x0,x0+dx,null],y:[y0,y0+dy,null],type:'scatter',mode:'lines',showlegend:false,hoverinfo:'skip'});
const arrows = (xs,ys,us,vs,color,scale)=>{const tr=[];for(let i=0;i<xs.length;i++){const s=scale*0.25/Math.hypot(us[i],vs[i]+1e-6);tr.push(seg(xs[i],ys[i],us[i]*s,vs[i]*s));}
  return {type:'scatter',mode:'lines',x:[].concat(...tr.map(t=>t.x)),y:[].concat(...tr.map(t=>t.y)),line:{color,width:1},showlegend:true,name:color};};
const traces=[];
const W = D.F.x, V = D.F.y;
traces.push({type:'heatmap',z:D.U,x:D.F.x[0],y:D.F.y.map(r=>r[0]),colorscale:'Viridis',opacity:0.85,showscale:true,colorbar:{title:'U'}});
traces.push(arrows(D.ds.x,D.ds.y,D.ds.nom_u,D.ds.nom_v,'#4f8cff',1));
traces.push(arrows(D.ds.x,D.ds.y,D.ds.mod_u,D.ds.mod_v,'#ff6b6b',1));
traces.push({type:'scatter',x:D.trajectory.map(p=>p[0]),y:D.trajectory.map(p=>p[1]),mode:'lines',line:{color:'#ffffff',width:2},name:'trajectory'});
if(D.lidar){const px=[D.state[0]],py=[D.state[1]];for(let i=0;i<D.lidar.angles.length;i++){const r=D.lidar.ranges[i];px.push(D.state[0]+r*Math.cos(D.lidar.angles[i]));py.push(D.state[1]+r*Math.sin(D.lidar.angles[i]));}
  traces.push({type:'scatter',x:px,y:py,mode:'markers',marker:{color:'#e0b25e',size:2},name:'LiDAR'});}
traces.push({type:'scatter',x:[D.goal[0]],y:[D.goal[1]],mode:'markers',marker:{color:'#22c55e',size:12,symbol:'x'},name:'goal'});
traces.push({type:'scatter',x:[D.state[0]],y:[D.state[1]],mode:'markers',marker:{color:'#ffffff',size:10},name:'drone'});
traces.push({type:'scatter',x:D.centers.map(c=>c[0]),y:D.centers.map(c=>c[1]),mode:'markers',marker:{color:'#888',size:3},name:'basis centers'});
for(const o of D.obstacles){if(o.kind==='cylinder'){traces.push({type:'scatter',x:[o.center[0]],y:[o.center[1]],mode:'markers',marker:{color:'#c98a2b',size:o.radius*40},name:'obstacle'});}
  else{const [cx,cy]=o.center,[hx,hy]=o.half,a=o.angle||0,ca=Math.cos(a),sa=Math.sin(a);
    const cs=[[-hx,-hy],[hx,-hy],[hx,hy],[-hx,hy],[-hx,-hy]].map(([x,y])=>[cx+ca*x-sa*y,cy+sa*x+ca*y]);
    traces.push({type:'scatter',x:cs.map(p=>p[0]),y:cs.map(p=>p[1]),mode:'lines',line:{color:'#c98a2b',width:3},name:'obstacle'});}}
Plotly.newPlot('c',traces,{title:`${D.controller} · ${D.scene} · learned obstacle field`,paper_bgcolor:'#0b0f14',plot_bgcolor:'#0b0f14',font:{color:'#c9d1d9'},xaxis:{scaleanchor:'y',title:'x [m]'},yaxis:{title:'y [m]'},width:window.innerWidth,height:window.innerHeight},{responsive:true});
</script></body></html>"""


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
