import datetime, json, numpy as np
from math import radians, sin, cos, asin, sqrt
from pathlib import Path

C  = 3.0e5      # km/s ground-wave
FS = 100_000    # 100 kHz ⇒ 6 M rows/min
STATIONS = [dict(id="LON", lat=51.5072, lon=-0.1276),
            dict(id="PAR", lat=48.8566, lon= 2.3522)]

def hav_km(lat1, lon1, lat2, lon2):
    R=6371; dlat,dlon = map(radians,(lat2-lat1, lon2-lon1))
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def make_noise(rng,N,t):
    x=rng.normal(0,0.003,N)
    x+=0.002*np.sin(2*np.pi*50*t)
    x+=0.001*np.sin(2*np.pi*62*t)
    x+=0.001*np.sin(2*np.pi*38*t)
    x+=0.0015*np.sin(2*np.pi*25*t)
    return x.astype("f4")

def simulate(minutes:int,out_prefix:str,seed:int=0):
    rng=np.random.default_rng(seed)
    N=FS*60*minutes; t=np.arange(N)/FS
    waves={s["id"]:make_noise(rng,N,t) for s in STATIONS}
    events=[]
    # three clusters every minute
    base_times=[5,25,45] if minutes==1 else np.linspace(10,60*minutes-10,3*minutes)
    for base_t, (name,d_rng,nf) in zip(base_times,
        [("near",(20,50), (8,12)), ("mid",(100,200),(5,9)), ("far",(400,600),(3,6))]*minutes):
        nf=rng.integers(*nf)
        base_lat=rng.uniform(49,53); base_lon=rng.uniform(-2,4)
        for _ in range(nf):
            ev_t=base_t+rng.uniform(0,2)
            d=rng.uniform(*d_rng); bearing=rng.uniform(0,2*np.pi)
            ev_lat=base_lat+(d/111)*np.cos(bearing)
            ev_lon=base_lon+(d/111)*np.sin(bearing)/np.cos(radians(base_lat))
            amp=rng.uniform(0.4,1.0)/(1+d/50); freq=rng.uniform(3e3,9e3)
            events.append(dict(t=float(ev_t),lat=float(ev_lat),lon=float(ev_lon),
                               amp=float(amp),freq=float(freq),cluster=name))
            for s in STATIONS:
                dist=hav_km(ev_lat,ev_lon,s["lat"],s["lon"]); delay=dist/C
                i0=int((ev_t+delay)*FS); dur=int(FS*0.04)
                if i0>=N: continue
                subt=np.arange(dur)/FS
                burst=amp*np.sin(2*np.pi*freq*subt)*np.exp(-subt/0.003)/(1+dist/50)
                sl=slice(i0,min(i0+dur,N))
                waves[s["id"]][sl]+=burst[:sl.stop-sl.start]
    Path(out_prefix).parent.mkdir(parents=True,exist_ok=True)
    for s in STATIONS: np.save(f"{out_prefix}_{s['id']}.npy",waves[s['id']])
    meta=dict(fs=FS,utc_start=datetime.datetime.utcnow().isoformat(timespec="seconds")+"Z",
              stations=STATIONS,events=events)
    json.dump(meta,open(f"{out_prefix}_meta.json","w"),indent=2)
    print(f"Saved {len(events)} flashes → {out_prefix}_*.npy")
