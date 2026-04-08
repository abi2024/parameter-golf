#!/usr/bin/env python3
"""Generate HTML dashboard from experiment JSONs. Usage: python3 dashboard.py"""
import json, glob, os
from datetime import datetime

def main():
    exps = []
    for f in sorted(glob.glob("experiments/exp_*.json")):
        try:
            with open(f) as fh: exps.append(json.load(fh))
        except: pass

    succeeded = [e for e in exps if e.get("status") == "success"]
    best = min((e["metrics"]["val_bpb"] for e in succeeded if e["metrics"].get("val_bpb")), default=None)

    rows = ""
    for e in reversed(exps):
        m = e.get("metrics", {})
        st = {"success":"✅","crashed":"💥","no_metrics":"❓","diverged":"📈","over_budget":"📦"}.get(e.get("status",""),"❓")
        bpb = f"{m['val_bpb']:.4f}" if m.get("val_bpb") else "—"
        art = f"{m['artifact_bytes']/1e6:.2f}MB" if m.get("artifact_bytes") else "—"
        best_mark = " 🏆" if m.get("val_bpb") and m["val_bpb"] == best else ""
        over = " style='color:#f85149;font-weight:bold'" if m.get("artifact_bytes") and m["artifact_bytes"] > 16000000 else ""
        rows += f"<tr class='{e.get('status','')}'><td>{st}</td><td>{e.get('name','?')}</td><td style='font-weight:bold;color:#58a6ff'>{bpb}{best_mark}</td><td{over}>{art}</td><td>{e.get('duration_seconds',0)}s</td><td style='color:#8b949e;font-size:12px'>{e.get('timestamp','')[:16]}</td></tr>\n"

    series = []
    best_so_far = 999
    for e in exps:
        b = e.get("metrics",{}).get("val_bpb")
        if b and e.get("status") == "success":
            best_so_far = min(best_so_far, b)
            series.append({"i":len(series),"b":b,"best":best_so_far,"n":e.get("name","")})

    improvement = f"{((best - 1.2244) / 1.2244 * 100):.2f}%" if best else "—"
    success_rate = f"{len(succeeded)/len(exps)*100:.0f}%" if exps else "—"

    html = f"""<!DOCTYPE html><html><head><meta charset=utf-8><title>Parameter Golf Dashboard</title>
<style>*{{margin:0;padding:0;box-sizing:border-box}}body{{background:#0d1117;color:#e6edf3;font-family:system-ui;padding:20px}}
h1{{font-size:22px;margin-bottom:4px}}.sub{{color:#8b949e;font-size:13px;margin-bottom:20px}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin-bottom:20px}}
.stat{{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}}
.sv{{font-size:24px;font-weight:700}}.sl{{color:#8b949e;font-size:11px;text-transform:uppercase;margin-top:3px}}
table{{width:100%;border-collapse:collapse;background:#161b22;border:1px solid #30363d;border-radius:8px;overflow:hidden}}
th{{background:#1c2128;text-align:left;padding:8px 12px;font-size:11px;text-transform:uppercase;color:#8b949e}}
td{{padding:7px 12px;border-top:1px solid #30363d;font-size:13px}}tr:hover{{background:#1c2128}}
tr.crashed{{opacity:0.4}}canvas{{width:100%;height:200px;display:block;margin-bottom:20px;background:#161b22;border:1px solid #30363d;border-radius:8px}}
</style></head><body>
<h1>⛳ Parameter Golf Dashboard</h1>
<p class=sub>{len(exps)} experiments · Best: {f'{best:.4f}' if best else '—'} · Generated {datetime.now().strftime('%H:%M %b %d')}</p>
<div class=stats>
<div class=stat><div class=sv style=color:#58a6ff>{len(exps)}</div><div class=sl>Total</div></div>
<div class=stat><div class=sv style=color:#3fb950>{len(succeeded)}</div><div class=sl>Succeeded</div></div>
<div class=stat><div class=sv style=color:#f85149>{len(exps)-len(succeeded)}</div><div class=sl>Failed</div></div>
<div class=stat><div class=sv style=color:#3fb950>{f'{best:.4f}' if best else '—'}</div><div class=sl>Best BPB</div></div>
<div class=stat><div class=sv style=color:#d29922>{improvement}</div><div class=sl>vs Baseline</div></div>
<div class=stat><div class=sv style=color:#bc8cff>{success_rate}</div><div class=sl>Success Rate</div></div>
</div>
<canvas id=c></canvas>
<table><thead><tr><th></th><th>Experiment</th><th>val_bpb</th><th>Artifact</th><th>Time</th><th>When</th></tr></thead>
<tbody>{rows}</tbody></table>
<script>
const D={json.dumps(series)};const c=document.getElementById('c');const x=c.getContext('2d');
c.width=c.offsetWidth;c.height=200;const W=c.width,H=c.height,p={{t:20,r:20,b:25,l:55}};
if(D.length>0){{const bs=D.map(d=>d.b),mn=Math.min(...bs)-0.005,mx=Math.max(...bs,1.2244)+0.005;
const sx=i=>p.l+(i/Math.max(D.length-1,1))*(W-p.l-p.r),sy=v=>p.t+(1-(v-mn)/(mx-mn))*(H-p.t-p.b);
x.fillStyle='#161b22';x.fillRect(0,0,W,H);
x.strokeStyle='#f8514933';x.setLineDash([4,4]);x.beginPath();x.moveTo(p.l,sy(1.2244));x.lineTo(W-p.r,sy(1.2244));x.stroke();x.setLineDash([]);
x.fillStyle='#f85149';x.font='10px system-ui';x.fillText('Baseline 1.2244',p.l+4,sy(1.2244)-4);
D.forEach((d,i)=>{{x.fillStyle='#58a6ff44';x.beginPath();x.arc(sx(i),sy(d.b),3,0,Math.PI*2);x.fill()}});
x.strokeStyle='#3fb950';x.lineWidth=2;x.beginPath();D.forEach((d,i)=>{{const y=sy(d.best);i===0?x.moveTo(sx(i),y):x.lineTo(sx(i),y)}});x.stroke();
for(let i=0;i<=4;i++){{const v=mx-(i/4)*(mx-mn);x.fillStyle='#8b949e';x.font='10px system-ui';x.textAlign='right';x.fillText(v.toFixed(3),p.l-6,sy(v)+3)}}
}}else{{x.fillStyle='#8b949e';x.font='14px system-ui';x.fillText('No experiments yet',W/2-60,H/2)}}
</script></body></html>"""
    with open("dashboard.html","w", encoding="utf-8") as f: f.write(html)

    print(f"Dashboard: dashboard.html ({len(exps)} experiments, best: {best})")

if __name__=="__main__": main()