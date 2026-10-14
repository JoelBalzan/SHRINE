import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import numpy as np, sim, struct_max_core as smc
from scipy.fftpack import dct
dm=sim.dDMs; i0=40; NT=sim.NT
for SN in [False, True]:
    sim.SELFNOISE=SN; sim.rng=np.random.default_rng(9)
    data=np.array([sim.realise(1.0) for _ in range(300)])
    kc=smc.get_kc(dct(data[0],norm='ortho')); fL=smc.lowpass_response(NT,kc,3); g=smc.highpass_eigenvalues(NT)*fL
    k=np.arange(NT); w=np.pi*k/(NT*sim.DT)                 # DCT mode angular freq (rad/us)
    SP=np.array([np.linalg.norm(g*dct(I,norm='ortho'),axis=1) for I in data])   # (R, ND)
    mcdiff=np.std(SP-SP[:,[i0]],axis=0)                                         # MC truth
    pred=[];shr=[];hw_new=[];hw_shr=[];dh=[]
    for r,I in enumerate(data[:100]):
        C=dct(I,norm='ortho'); sp=SP[r]; pk=int(np.argmax(sp)); dh.append(dm[pk])
        sig_n=1.4826*np.median(np.abs(C[pk,NT//4:]))      # white level from signal-free band
        a=g*fL*C[pk]; u=a/np.linalg.norm(a)               # structure direction at peak (DCT domain)
        s12=[]
        for d in range(len(dm)):
            dtau=(dm[d]-dm[pk])*sim.tau_unit               # per-channel delay difference (us)
            rho=np.mean(np.cos(np.outer(w,dtau)),axis=1)   # noise correlation between trials, per mode
            s12.append(sig_n*np.sqrt(np.sum((g*u)**2*2*(1-rho))))
        s12=np.array(s12)
        res=smc.structure_param(I,kc=kc); rel=res['relative_uncertainty']
        if r<100 and pk==i0: pred.append(s12); shr.append(rel*sp)
        ok=sp>=sp[pk]-s12; a_=pk
        while a_>0 and ok[a_-1]: a_-=1
        b_=pk
        while b_<len(dm)-1 and ok[b_+1]: b_+=1
        hw_new.append((dm[b_]-dm[a_])/2)
        lo,hi=smc.dm_range_from_relative_uncertainty(sp,rel,dm); hw_shr.append((hi-lo)/2)
    pred=np.median(pred,axis=0); shr=np.median(shr,axis=0)
    print(f"selfnoise={SN} kc={kc}")
    for j in [41,43,45,50,60]:
        print(f"  dDM={dm[j]-dm[i0]:+.3f}: MC std[SP(d)-SP(0)]={mcdiff[j]:.3f}  analytic={pred[j]:.3f}  SHRINE ||D1S(Δd-Δ0)||={shr[j]:.3f}")
    print(f"  DM: MC std(dhat)={np.std(dh):.4f}  new level-crossing hw={np.median(hw_new):.4f}  SHRINE hw={np.median(hw_shr):.4f}")
