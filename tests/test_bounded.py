import threading, time
from concurrent.futures import ThreadPoolExecutor
from autosorter.core.bounded import run_bounded

def test_order_independent_and_bounded():
    inflight=0; peak=0; lock=threading.Lock()
    def fn(x):
        nonlocal inflight, peak
        with lock:
            inflight+=1; peak=max(peak,inflight)
        time.sleep(0.05 if x%4==0 else 0.005)
        with lock: inflight-=1
        return x*2
    with ThreadPoolExecutor(4) as ex:
        out=list(run_bounded(ex,range(40),fn,max_inflight=8))
    assert sorted(i for i,_,_ in out)==list(range(40))
    assert all(r==i*2 and e is None for i,r,e in out)
    assert peak<=4

def test_errors_are_reported_not_raised():
    def fn(x):
        if x==3: raise ValueError('bad')
        return x
    with ThreadPoolExecutor(2) as ex:
        out={i:(r,e) for i,r,e in run_bounded(ex,range(6),fn,max_inflight=4)}
    assert isinstance(out[3][1],ValueError) and out[2]==(2,None) and len(out)==6

def test_cancel_stops_submitting():
    started=[]
    cancel=False
    def fn(x):
        started.append(x); time.sleep(0.02); return x
    with ThreadPoolExecutor(2) as ex:
        n=0
        for i,r,e in run_bounded(ex,range(1000),fn,max_inflight=4,should_cancel=lambda: cancel):
            n+=1
            if n==5: cancel=True
    assert len(started)<30
