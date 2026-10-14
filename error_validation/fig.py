import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import numpy as np, json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sim, struct_max_core as smc, struct_max_errors as sme
from scipy.fftpack import dct
sim.SELFNOISE=True; sim.rng=np.random.default_rng(7)
dm=sim.dDMs; NR=200; J=150
data=[sim.realise(1.0) for _ in range(NR)]; bank=np.array([sim.realise(1.0,False) for _ in range(J)])
kc=smc.get_kc(dct(data[0],norm='ortho'))
dh=[];sd=[];lo=[];hi=[];sp0=[];ssp0=[];shr0=[]
for I in data:
    r=sme.dm_uncertainty(I,dm,bank,kc); dh.append(r['dm_hat']); sd.append(r['sigma_dm'])
    s=smc.structure_param(I,kc=kc); a,b=smc.dm_range_from_relative_uncertainty(s['sp'],s['relative_uncertainty'],dm); lo.append(a);hi.append(b)
    sp0.append(s['sp'][40]); ssp0.append(r['sigma_sp'][40]); shr0.append(s['uncertainty'][40]*s['sp'][40])
dh=np.array(dh);sd=np.array(sd);lo=np.array(lo);hi=np.array(hi)
print('kc',kc,'MC sd',dh.std(),'new',np.median(sd),'SHRINE hw',np.median((hi-lo)/2),'cov1',np.mean(abs(dh)<sd))
print('SP MC',np.std(sp0),'new',np.median(ssp0),'SHRINE',np.median(shr0))
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
I=data[0]; s=smc.structure_param(I,kc=kc)
ax[0].plot(dm,s['sp'],'k',label='SP')
ax[0].plot(dm,s['sp']*(1+s['relative_uncertainty']),'C3',label='SHRINE: SP + ||DS(Δ_d−Δ_0)||')
r=sme.dm_uncertainty(I,dm,bank,kc)
ax[0].fill_between(dm,s['sp']-r['sigma_sp'],s['sp']+r['sigma_sp'],color='C0',alpha=.3,label='new ±σ_SP (per-DM, absolute)')
ax[0].axhline(s['sp'].max(),ls=':',c='gray'); ax[0].set_xlabel('ΔDM'); ax[0].set_ylabel('structure parameter'); ax[0].legend(fontsize=7)
ax[1].hist(dh,bins=25,color='gray',alpha=.7,label=f'MC σ = {dh.std():.4f}')
x=np.linspace(-0.03,0.03,400); m=np.median(sd)
ax[1].plot(x,len(dh)*(np.diff(np.histogram(dh,25)[1])[0])*np.exp(-x**2/2/m**2)/np.sqrt(2*np.pi)/m,'C0',label=f'new σ = {m:.4f}')
ax[1].axvspan(np.median(lo),np.median(hi),color='C3',alpha=.15,label=f'SHRINE 68% ({np.median(lo):+.3f},{np.median(hi):+.3f})')
ax[1].set_xlim(-0.03,0.03); ax[1].set_xlabel('recovered ΔDM'); ax[1].legend(fontsize=7)
res=[json.load(open(f)) for f in ['res_amp0.5_sn1.json','res_amp1.0_sn0.json','res_amp1.0_sn1.json']]
lab=['amp .5\nself-noise','amp 1\nradiometer','amp 1\nself-noise']
rs=[];rd=[];ns=[];nd=[]
for o in res:
    o={k:np.array(v,float) for k,v in o.items() if isinstance(v,list)}
    mc=np.std(o['sp0']); md=np.std(o['dhat'])
    rs.append(np.median(o['shrine_abs0'])/mc); rd.append(np.nanmedian((o['shr_hi']-o['shr_lo'])/2)/md)
    ns.append(np.median(o['proj_local0'])/mc); nd.append(np.median(o['sig_dm'])/md)
xx=np.arange(3); w=.2
ax[2].bar(xx-1.5*w,rs,w,color='C3',alpha=.5,label='SHRINE σ_SP / MC'); ax[2].bar(xx-.5*w,rd,w,color='C3',label='SHRINE σ_DM / MC')
ax[2].bar(xx+.5*w,ns,w,color='C0',alpha=.5,label='new σ_SP / MC'); ax[2].bar(xx+1.5*w,nd,w,color='C0',label='new σ_DM / MC')
ax[2].axhline(1,c='k',lw=.8); ax[2].set_xticks(xx); ax[2].set_xticklabels(lab,fontsize=8); ax[2].set_ylabel('estimated / true scatter'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('shrine_error_diagnosis.png',dpi=140)
