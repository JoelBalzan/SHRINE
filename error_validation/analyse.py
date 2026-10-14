import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import sys, json
import numpy as np
from scipy.fftpack import dct, idct
import struct_max_core as smc
import sim

amp = float(sys.argv[1]); NR = int(sys.argv[2]); J = int(sys.argv[3])
sim.SELFNOISE = (sys.argv[4] == '1') if len(sys.argv) > 4 else True
NOSCALE = len(sys.argv) > 5 and sys.argv[5] == 'noscale'
dm = sim.dDMs; ND = len(dm); NT = sim.NT; i0 = int(np.argmin(abs(dm)))

data = np.array([sim.realise(amp) for _ in range(NR)])
bank = np.array([sim.realise(amp, with_signal=False) for _ in range(J)])   # "off-pulse" noise cubes
kc = smc.get_kc(dct(data[0], norm="ortho"))
fL = smc.lowpass_response(NT, kc, 3); hp = smc.highpass_eigenvalues(NT); g = hp*fL
frac = np.sum((1-fL)**2)/NT          # fraction of white-noise power that survives into Delta = (1-S) i

def peak_refine(y, x):
    k = int(np.argmax(y)); k = min(max(k, 1), len(y)-2)
    a, b, c = y[k-1], y[k], y[k+1]; den = a - 2*b + c
    off = 0.5*(a - c)/den if den != 0 else 0.0
    return x[k] + off*(x[1]-x[0])

def robust_std(x):
    return 1.4826*np.median(np.abs(x - np.median(x)))

out = dict(sp0=[], shrine_abs0=[], proj_white0=[], proj_local0=[], dhat=[], dhat_grid=[],
           shr_lo=[], shr_hi=[], sig_dm=[], lvl_lo=[], lvl_hi=[], neff=[])
for r in range(NR):
    I = data[r]
    res = smc.structure_param(I, kc=kc)
    sp, rel, unc = res["sp"], res["relative_uncertainty"], res["uncertainty"]
    lo, hi = smc.dm_range_from_relative_uncertainty(sp, rel, dm)
    out["shr_lo"].append(lo); out["shr_hi"].append(hi)
    out["sp0"].append(sp[i0]); out["shrine_abs0"].append(unc[i0]*sp[i0])
    out["dhat_grid"].append(dm[np.argmax(sp)]); out["dhat"].append(peak_refine(sp, dm))

    C = dct(I, norm="ortho")                       # (ND, NT)
    chat = fL[None]*C                              # smoothed signal estimate
    gc = g[None]*chat
    # (1) stationary white-noise projection: sigma_SP = sigma_w ||g^2 c|| / ||g c||
    sw = np.array([robust_std(C[d, NT//2:]) for d in range(ND)])
    proj_w = sw*np.linalg.norm(g[None]*gc, axis=1)/np.linalg.norm(gc, axis=1)
    out["proj_white0"].append(proj_w[i0])
    # (2) self-noise aware: time-domain weights w_d(t), local variance from residual
    W = idct(g[None]*gc, norm="ortho")/np.linalg.norm(gc, axis=1)[:, None]   # (ND, NT)
    # local noise variance from the signal-free band k > NT/4 (no burst power leaks there)
    hiband = np.zeros(NT); hiband[NT//4:] = 1.0
    Dhi = idct(hiband[None]*C, norm="ortho")
    ker = np.ones(31)/31
    var_t = np.array([np.convolve(Dhi[d]**2, ker, mode="same") for d in range(ND)])/0.75
    proj_l = np.sqrt(np.sum(W**2*var_t, axis=1))
    out["proj_local0"].append(proj_l[i0])
    # N_eff: ||g(1-fL) n|| / sigma_proj for white noise
    out["neff"].append(float((sw[i0]*np.linalg.norm(g*(1-fL)))**2/proj_w[i0]**2))

    # (3) DM error from noise bank: linear perturbation of SP curve, self-noise rescaled
    dpk = int(np.argmax(sp))
    scale = np.sqrt(var_t[dpk]/np.mean(var_t[dpk][:NT//4]))      # on-pulse / off-pulse noise ratio
    if NOSCALE: scale = np.ones(NT)
    dSP = np.einsum("jdt,dt->jd", bank*scale[None, None], W)        # (J, ND)
    # local quadratic model of SP around peak
    win = slice(max(dpk-3, 0), min(dpk+4, ND))
    pc = np.polyfit(dm[win], sp[win], 2)
    model = np.polyval(pc, dm)
    dd = np.array([peak_refine(model + dSP[j], dm) for j in range(J)])
    out["sig_dm"].append(float(np.std(dd)))
    # level-crossing interval using projected differential sigma
    sdiff = np.std(dSP - dSP[:, [dpk]], axis=0)
    ok = sp >= sp[dpk] - sdiff
    # contiguous region containing peak
    a = dpk
    while a > 0 and ok[a-1]: a -= 1
    b = dpk
    while b < ND-1 and ok[b+1]: b += 1
    out["lvl_lo"].append(float(dm[a])); out["lvl_hi"].append(float(dm[b]))

out = {k: [None if v is None else float(v) for v in vals] for k, vals in out.items()}
out["kc"] = kc; out["amp"] = amp
json.dump(out, open(f"res_amp{amp}_sn{int(sim.SELFNOISE)}.json", "w"))

o = {k: np.array(v, dtype=float) for k, v in out.items() if isinstance(v, list)}
print(f"amp={amp} kc={kc} NR={NR} J={J}")
print(f"  MC std SP(dDM=0)         : {np.std(o['sp0']):.4g}")
print(f"  SHRINE ||DS Delta||      : {np.median(o['shrine_abs0']):.4g}  (ratio {np.median(o['shrine_abs0'])/np.std(o['sp0']):.1f})")
print(f"  projected, white         : {np.median(o['proj_white0']):.4g}")
print(f"  projected, self-noise    : {np.median(o['proj_local0']):.4g}")
print(f"  sqrt(N_eff) (white)      : {np.sqrt(np.median(o['neff'])):.1f}")
print(f"  MC std dhat (refined)    : {np.std(o['dhat']):.4g}   grid: {np.std(o['dhat_grid']):.4g}  mean {np.mean(o['dhat']):.2e}")
hw = 0.5*(o['shr_hi']-o['shr_lo'])
print(f"  SHRINE half-width median : {np.nanmedian(hw):.4g}  (ratio {np.nanmedian(hw)/np.std(o['dhat']):.1f})")
print(f"  bank sigma_dm median     : {np.median(o['sig_dm']):.4g}")
lw = 0.5*(o['lvl_hi']-o['lvl_lo'])
print(f"  projected level-crossing : {np.median(lw):.4g}")
z = o['dhat']/o['sig_dm']
print(f"  coverage |dhat|<1 sig_dm : {np.mean(abs(z)<1):.2f}, <2: {np.mean(abs(z)<2):.2f}")
print(f"  SHRINE coverage          : {np.mean((o['shr_lo']<=0)&(o['shr_hi']>=0)):.2f}")
