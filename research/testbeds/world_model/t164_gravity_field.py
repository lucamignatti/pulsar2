"""W164: gravity-field self-imitation. The production mechanism (commit 49e88722) on AirLine.

No goals, no bank, no per-arena retargeting, no reachability band. Goal MASS is binary over the
occupancy pool - a state is a goal if the map does not KNOW it (twin disagreement >= p90) or thinks
it is GOOD but UNPROVEN (value above median, visits below median) - and the pull is the summed field

    phi(s) = sum over goals of exp( -d(s->g) / (GRAVITY_DECISIONS * localD) )

Each update, the rows the policy actually played whose phi is highest are imitated. Nothing is
stored between updates except the occupancy pool, which is what keeps a region pulling after the
arena that found it moved on.

Collection is PURE ON-POLICY: no commitment, no held actions, no injected exploration, no reward
shaping. Identical to W161 in every respect except which executed rows the imitation term
up-weights, and the dose is identical too (air_cycle.ppo subsamples 1024 region rows and
build_region normalises mean weight to TARGET_MEAN_WEIGHT, so row count is selectivity only).

Arms:
  gravity  the field above
  flat     CONTROL: identical plumbing, phi replaced by uniform noise. Isolates the field from the
           fact that we are imitating our own recent play at all.
"""
import argparse,copy,json,os,time
from pathlib import Path
import numpy as np
import torch

import air_line as A, air_cycle as C, air_adapter as AD
from frontier_core import MapEnsemble,ValueTwins,expectile_loss
import w163_value_map as VM
from quasimetric_map import Dataset
from frontier_core import train_fast as train_map

REPO=Path(__file__).resolve().parents[3]
ROOT=REPO/'research/results/world_model_w164_gravity_20260915'
BASE=REPO/'research/results/world_model_w160_air_baseline_20260914'
MAPDIR=REPO/'research/results/world_model_w160_air_map_20260914'
VALDIR=REPO/'research/results/world_model_w163_value_map_20260914'
ARMS=('gravity','flat')
ITERATIONS=2048;GUIDANCE_END=1024;FORK=511
MAP_STEPS=24;TWIN_STEPS=24;REPLAY=200_000

# --- the production constants, ported verbatim ---------------------------------------------
# Gravity range is SCALE-FREE: sigma is a fraction of the state distribution's OWN median pairwise
# distance, not a fixed decision count. In Rocket League 50 decisions x localD 0.47 = 23.5 units
# against a median pair distance of 35 -> ratio 0.67, which is where contrast measured 1.76. Stating
# it in decisions does not transfer between metrics with different spreads: the same 50 decisions in
# this toy gives sigma >> the whole distance distribution, and the field goes flat (contrast 1.09).
GRAVITY_FRAC=0.67
UNKNOWN_Q=0.90             # twin disagreement above this = the map does not know it
OCC=8192                   # occupancy pool ~ the C++ 65k/16.7k = several updates of retention
GOAL_CAP=512               # goals sampled per update (compute cap)
VISIT_BITS=12              # 4096 buckets vs 1024 pool rows/update, ~the C++ ratio
VISIT_DECAY=0.99
TOP_FRAC=0.10              # imitate the best tenth of what we played. Dose is UNAFFECTED by this
                           # (ppo subsamples 1024, weights normalise) - it is selectivity only.
DELTA_CAP=2.0;DELTA_SHARP=3.0;TARGET_MEAN_WEIGHT=25.0   # identical to W161, so doses match


class Gravity:
    def __init__(self,n,maps,seed,value=None,v1=None,v2=None,arm='gravity'):
        self.n=n;self.maps=maps;self.value=value;self.v1=v1;self.v2=v2;self.arm=arm
        self.rng=np.random.default_rng(9_640_000+seed)
        self.occ=None;self.occ_ctx=None;self.enabled=False;self.rows=None;self.stats={}
        g=torch.Generator().manual_seed(0xF00D+seed)
        m0=maps.members[0];L=m0.latent_dim+m0.symmetric_dim
        self.proj=torch.randn(L,VISIT_BITS,generator=g)
        self.visit=torch.zeros(1<<VISIT_BITS)
        self.localD=1.0

    def reset_update(self):
        self.rows=[];self.stats={}

    def push_occupancy(self,mo,ctx):
        self.occ=mo.copy() if self.occ is None else np.concatenate((self.occ,mo))[-OCC:]
        self.occ_ctx=ctx.copy() if self.occ_ctx is None else np.concatenate((self.occ_ctx,ctx))[-OCC:]

    def _key(self,z):
        b=(z@self.proj[:z.shape[1]]>0).long()
        return (b*(2**torch.arange(b.shape[1]))).sum(-1)

    @torch.no_grad()
    def observe_visits(self,mo,ctx):
        z=self.maps.encode(torch.from_numpy(mo),torch.from_numpy(ctx))
        self.visit.index_add_(0,self._key(z),torch.ones(len(z)))
        self.visit.mul_(VISIT_DECAY)

    @torch.no_grad()
    def measure_scale(self,mo,mn,ctx):
        """localD for reporting; sigma from the MEDIAN PAIRWISE distance, which is what sets contrast."""
        j=self.rng.choice(len(mo),min(2048,len(mo)),replace=False)
        z=self.maps.encode(torch.from_numpy(mo[j]),torch.from_numpy(ctx[j]))
        zn=self.maps.encode(torch.from_numpy(mn[j]),torch.from_numpy(ctx[j]))
        self.localD=0.9*self.localD+0.1*max(float(self.maps.distance(z,zn).mean()),1e-3)
        k=min(256,len(z));a=z[:k]
        pw=self.maps.distance(a.repeat_interleave(k,0),a.repeat(k,1))
        med=float(torch.quantile(pw,0.5))
        self.sigma=0.9*getattr(self,'sigma',med*GRAVITY_FRAC)+0.1*max(med*GRAVITY_FRAC,1e-3)

    @torch.no_grad()
    def build_goals(self):
        """Binary mass: unknown (twin disagreement high) OR good-but-unproven (V high, visits low)."""
        if self.occ is None or len(self.occ)<256:return False
        o=torch.from_numpy(self.occ);c=torch.from_numpy(self.occ_ctx)
        x=torch.cat((o,c),-1)
        v=torch.minimum(self.v1(x),self.v2(x)).squeeze(-1)
        dis=(self.v1(x)-self.v2(x)).squeeze(-1).abs()
        z=self.maps.encode(o,c)
        vis=self.visit[self._key(z)]
        unknown=dis>=torch.quantile(dis,UNKNOWN_Q)
        unproven=(v>=torch.quantile(v,0.5))&(vis<=torch.quantile(vis,0.5))
        m=(unknown|unproven)
        gi=torch.nonzero(m).flatten()
        if len(gi)<8:self.goalz=None;return False
        if len(gi)>GOAL_CAP:gi=gi[torch.randperm(len(gi))[:GOAL_CAP]]
        self.goalz=z[gi]
        self.mass_frac=float(m.float().mean())
        return True

    @torch.no_grad()
    def phi(self,obs,ctx):
        """Summed gravity: clumps pull harder, separate clumps combine."""
        z=self.maps.encode(torch.from_numpy(obs[:,:AD.MAP_OBS_DIM]),torch.from_numpy(ctx))
        sig=max(getattr(self,"sigma",1.0),1e-3)
        n,G=len(z),len(self.goalz)
        d=self.maps.distance(z.repeat_interleave(G,0),self.goalz.repeat(n,1)).reshape(n,G)
        return torch.exp(-d/sig).sum(1).numpy()

    def tick(self,env,obs,intents,act):
        if not self.enabled or getattr(self,'goalz',None) is None:return
        ctx=env.theta.astype(np.float32)
        p=self.rng.random(self.n).astype(np.float32) if self.arm=='flat' else self.phi(obs,ctx)
        for e in range(self.n):
            self.rows.append((obs[e].copy(),int(act[e]),int(intents[e]),float(p[e])))


def build_region(rows,stats):
    """Top TOP_FRAC by field value, uniform weight, normalised exactly as W161 did."""
    if not rows:return None,dict(entries=0,rows=0)
    phi=np.array([r[3] for r in rows],np.float32)
    k=max(8,int(len(rows)*TOP_FRAC))
    keep=np.argsort(-phi)[:k]
    o=np.stack([rows[i][0] for i in keep]);a=np.array([rows[i][1] for i in keep],np.int64)
    z=np.array([rows[i][2] for i in keep],np.int64)
    w=np.full(len(keep),TARGET_MEAN_WEIGHT,np.float32)
    contrast=float(np.percentile(phi,90)/max(np.percentile(phi,10),1e-9))
    return ((torch.from_numpy(o),torch.from_numpy(a),torch.from_numpy(z),torch.from_numpy(w)),
            dict(entries=int(k),rows=int(k),mean_weight=TARGET_MEAN_WEIGHT,
                 field_contrast=contrast,imitated_phi=float(phi[keep].mean()),
                 pool_phi=float(phi.mean()),**stats))


def run(seed,arm,root=None,iterations=None,guidance_end=None):
    root=root or ROOT;iterations=iterations or ITERATIONS;guidance_end=guidance_end or GUIDANCE_END
    out=root/f'seed{seed}_{arm}';out.mkdir(parents=True,exist_ok=False)
    snap=torch.load(BASE/f'seed{seed}_snapshot_{FORK}.pt',map_location='cpu',weights_only=False)
    agent=C.Agent();agent.load_state_dict(snap['actor'])
    opt=torch.optim.Adam(agent.parameters(),lr=1e-3);opt.load_state_dict(copy.deepcopy(snap['actor_optimizer']))
    rows=C.Rows(seed);rows.load_state_dict(snap['collector'])
    bank=C.Bank(seed);bank.rows=copy.deepcopy(snap['bank_rows']);bank.seen=snap['bank_seen'].copy()
    torch.set_rng_state(snap['torch_rng'])
    maps=MapEnsemble(AD.MAP_OBS_DIM,AD.CTX_DIM,k=4,seed=11)
    maps.load_state_dict(torch.load(MAPDIR/'map_ensemble.pt',map_location='cpu'))
    twins=ValueTwins(AD.MAP_OBS_DIM,AD.CTX_DIM);tw=torch.optim.Adam(twins.parameters(),lr=1e-3)
    vck=torch.load(VALDIR/'value_map.pt',map_location='cpu')
    vd=AD.MAP_OBS_DIM+AD.CTX_DIM
    _v1=VM.Value(vd);_v1.load_state_dict(vck['v1']);_v2=VM.Value(vd);_v2.load_state_dict(vck['v2'])
    _v1.eval();_v2.eval()
    for _p in list(_v1.parameters())+list(_v2.parameters()):_p.requires_grad_(False)
    valuefn=lambda x:torch.minimum(_v1(x),_v2(x))
    fr=Gravity(rows.n,maps,seed,value=valuefn,v1=_v1,v2=_v2,arm=arm)
    C.dump(out/'resume.json',dict(fork=str(BASE/f'seed{seed}_snapshot_{FORK}.pt'),seed=seed,arm=arm,
        mechanism='gravity field, binary goal mass, no bank/goals/band',
        gravity_frac=GRAVITY_FRAC,unknown_q=UNKNOWN_Q,occ=OCC,top_frac=TOP_FRAC,
        commitment=False,reward_shaping=False,withdrawn_at=guidance_end))
    hist=[];replay=None;t0=time.time()
    for it in range(iterations):
        fr.enabled=it<guidance_end;fr.reset_update()
        def override(t,env,obs,sampled):
            fr.tick(env,obs,rows.intents,sampled)
            return sampled,np.zeros(rows.n,bool)          # pure on-policy
        D,complete=rows.collect(agent,override=override)
        assert not D['committed'].any()
        bank.admit(complete)
        mo=D['obs'][:,:AD.MAP_OBS_DIM];mn=D['nxt'][:,:AD.MAP_OBS_DIM];mc=D['ctx']
        fr.push_occupancy(mo[::8],mc[::8]);fr.observe_visits(mo[::8],mc[::8])
        fr.measure_scale(mo,mn,mc)
        replay=(mo,mc,mn) if replay is None else tuple(np.concatenate((a,b))[-REPLAY:] for a,b in zip(replay,(mo,mc,mn)))
        if fr.enabled:
            train_map(maps.members[it%maps.k],Dataset(replay[0],replay[1],replay[2],np.zeros(len(replay[0]),np.int64)),
                      steps=MAP_STEPS,batch=512,lr=3e-4,seed=it)
            fr.build_goals()
        T=torch.from_numpy
        for _ in range(TWIN_STEPS):
            j=torch.randint(len(mo),(1024,))
            with torch.no_grad():
                tgt=T(D['reward'])[j]+T((~D['terminal']).astype(np.float32))[j]*twins.v_dag(T(mn)[j],T(mc)[j])
            z=torch.cat((T(mo)[j],T(mc)[j]),-1)
            loss=(twins.v_real(T(mo)[j],T(mc)[j])-T(D['ret'])[j]).pow(2).mean()
            loss=loss+expectile_loss(twins.dag1(z).squeeze(-1),tgt,twins.tau)
            loss=loss+expectile_loss(twins.dag2(z).squeeze(-1),tgt,twins.tau)
            tw.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(twins.parameters(),10.);tw.step()
        region,sil=build_region(fr.rows if fr.enabled else [],
                                dict(mass_frac=getattr(fr,'mass_frac',0.),localD=fr.localD,sigma=getattr(fr,'sigma',0.)))
        grad=C.ppo(agent,opt,D,bank,region)
        rec=dict(iteration=it,real_steps=rows.real_rows,
            train_return=float(D['ep_ret'].mean()) if len(D['ep_ret']) else 0.,
            strike_rate=float(complete['hit'].mean()) if complete else 0.,
            region_sil=sil,gradients=grad,committed_rows=0,reward_shaping=False)
        if it%32==0 or it in (guidance_end-1,iterations-1,iterations-2):
            rec['evaluation']=C.evaluate(agent,700000+seed*97+it,n=512 if it in (guidance_end-1,iterations-1) else 256)
            rec['profile']=C.action_profile(agent,710000+seed*31+it)
            print(seed,arm,it,{k:round(v,4) for k,v in rec['evaluation']['mid'].items()},
                  'sil',sil.get('entries',0),'contrast',round(sil.get('field_contrast',0),3),
                  'mass',round(sil.get('mass_frac',0),3),
                  'pop',round(rec['profile']['p_pop_when_grounded_near'],5),flush=True)
        hist.append(rec);C.dump(out/'history.json',hist)
        if it in (guidance_end-1,iterations-1):
            torch.save(dict(actor=agent.state_dict(),iteration=it,seed=seed,arm=arm),out/f'snapshot_{it}.pt')
    C.dump(out/'readout.json',dict(seed=seed,arm=arm,history=hist,wall_seconds=time.time()-t0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--core',type=int,required=True)
    p.add_argument('--output-root',type=Path);p.add_argument('--iterations',type=int,default=ITERATIONS)
    p.add_argument('--guidance-end',type=int,default=GUIDANCE_END)
    a=p.parse_args()
    os.sched_setaffinity(0,{a.core});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run(a.seed,a.arm,a.output_root,a.iterations,a.guidance_end)
