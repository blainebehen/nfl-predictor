"""
Re-tune the QB and EPA settings after switching to raw QB ratings.

I had decided (research/qb.py) that comparing each starter to his team's
usual QB didn't help, and to use the raw QB ratings instead. But the model
code still had the team-baseline version switched on, so the backtest
numbers and live ratings came from the version I'd rejected. This fixes
that and re-checks everything across the same six test windows: for each
window, tune on the seasons before it, then score it.

Result: raw QB ratings help in all six windows, by more than before. Team
EPA on top of that is basically zero (+0.0005 to -0.0002).
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import load_games
from elo import run_elo, log_loss, accuracy, season_mask
from features import build_qb_map, team_game_epa

g = load_games()
qm = build_qb_map(g, verbose=False)
em = team_game_epa()
K,H,RHO = 20,50,0.5
QS=[0,300,400,500,600,700,800,1000]; QA=[0.01,0.02,0.03,0.05]
ES=[0,50,100,150,200,300]; EA=[0.05,0.1,0.15,0.25]
cache={}
def run(qs,qa,es,ea):
    key=(qs,qa,es,ea)
    if key not in cache:
        E,S,_=run_elo(g,k=K,H=H,rho=RHO,mov=True,qb_map=qm,qb_scale=qs,qb_alpha=qa,epa_map=em,epa_scale=es,epa_alpha=ea)
        cache[key]=(E,S)
    return cache[key]
def best(mask, grid):
    return min(grid, key=lambda p: log_loss(*[x[mask] for x in run(*p)]))
WIN=[(2004,2009),(2010,2015),(2016,2018),(2019,2021),(2022,2025),(2019,2025)]
print('window    | noQB   QB(raw) QB+EPA | picked QB     picked EPA')
for lo,hi in WIN:
    tr=season_mask(g,max_season=lo-1); te=season_mask(g,lo,hi)
    q=best(tr,[(s,a,0,0.15) for s in QS for a in QA])
    e=best(tr,[(q[0],q[1],s,a) for s in ES for a in EA])
    L0=log_loss(*[x[te] for x in run(0,0.02,0,0.15)])
    Lq=log_loss(*[x[te] for x in run(*q)]); Le=log_loss(*[x[te] for x in run(*e)])
    print(f'{lo}-{hi} | {L0:.4f} {Lq:.4f} {Le:.4f} | {q[0]},{q[1]}  {e[2]},{e[3]}')
allm=season_mask(g)
q=best(allm,[(s,a,0,0.15) for s in QS for a in QA]); e=best(allm,[(q[0],q[1],s,a) for s in ES for a in EA])
for lab,p in [('noQB',(0,0.02,0,0.15)),('QB',q),('QB+EPA',e),('old 600/.02+200/.15',(600,0.02,200,0.15))]:
    E,S=run(*p); print(f'all games {lab:<22} {p}  L {log_loss(E,S):.4f} acc {accuracy(E,S):.4f}')
