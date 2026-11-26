"""Portable CSV/NumPy preprocessing and metrics for the frozen TabM packet."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

WEIGHTS=np.array([1.,1.1145828276911305,1.175343805006587,1.1241432712795814])
NA_TOKENS=frozenset(['','NA','N/A','None','NULL','NaN','nan','null','n/a','<NA>','#N/A','#NA',
                     '#N/A N/A','-1.#IND','1.#IND','-1.#QNAN','1.#QNAN','-NaN','-nan'])
MODEL=dict(d_out=4,n_blocks=2,d_block=256,dropout=.1,k=16,arch_type='tabm',
           start_scaling_init='random-signs',num_embeddings=None)
TRAIN=dict(lr=.002,weight_decay=.0003,batch_size=512,max_epochs=80,patience=12)


def require(ok,message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(50):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==49:
                raise
            time.sleep(.1)


def row_digest(rows,y):
    return hashlib.sha256(np.asarray(rows,dtype='<i8').tobytes()+np.asarray(y,dtype='<i8')[rows].tobytes()).hexdigest()


def array_digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def macro_f1(y,p):
    y,p=np.asarray(y,dtype=int),np.asarray(p,dtype=int)
    require(y.shape==p.shape and len(y)>0 and np.isin(y,range(4)).all() and np.isin(p,range(4)).all(),'Invalid classification arrays')
    m=np.bincount(4*y+p,minlength=16).reshape(4,4)
    den=m.sum(0)+m.sum(1)
    return float(np.divide(2.*np.diag(m),den,out=np.zeros(4),where=den>0).mean())


def softmax(logits):
    logits=np.asarray(logits,dtype=np.float64)
    exp=np.exp(logits-logits.max(axis=-1,keepdims=True))
    return exp/exp.sum(axis=-1,keepdims=True)


def blended(lr,p):
    return ((.75*np.asarray(lr)+.25*np.asarray(p))*WEIGHTS).argmax(1)


def load_table(path,schema,nrows,target=True):
    """Same raw parser policy, without a cloud pandas/sklearn dependency."""
    numeric=np.empty((nrows,len(schema['numeric'])),dtype=np.float64)
    categorical=np.empty((nrows,len(schema['categorical'])),dtype=object)
    ids=np.empty(nrows,dtype='U24')
    y=np.empty(nrows,dtype=np.int64) if target else None
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        require(set(schema['columns']).issubset(reader.fieldnames),'Raw schema differs')
        count=0
        for i,row in enumerate(reader):
            require(i<nrows,'Too many source rows')
            ids[i]=row['player_id']
            for j,col in enumerate(schema['numeric']):
                s=row[col]
                value=float(s) if s not in NA_TOKENS else np.nan
                numeric[i,j]=value if np.isfinite(value) else np.nan
            for j,col in enumerate(schema['categorical']):
                s=row[col]
                if s in NA_TOKENS:
                    s='__MISSING__'
                elif col=='preferred_game_mode':
                    s=str(int(float(s)))
                categorical[i,j]=sys.intern(s)
            if target:
                y[i]=int(row['segment'])
            count=i+1
    require(count==nrows,'Source row count differs')
    require(len(np.unique(ids))==nrows,'Duplicate source IDs')
    return dict(numeric=numeric,categorical=categorical,ids=ids,y=y)


def transform(table,rows,state,dtype=np.float32):
    rows=np.asarray(rows,dtype=int)
    raw=table['numeric'][rows]
    missing=np.isnan(raw)
    filled=np.where(missing,np.asarray(state['medians']),raw)
    indicators=missing[:,state['indicator_indices']].astype(np.float64)
    num=np.column_stack([filled,indicators])
    num=(num-np.asarray(state['scaler_mean']))/np.asarray(state['scaler_scale'])
    sizes=[len(c) for c in state['categories']]
    result=np.zeros((len(rows),num.shape[1]+sum(sizes)),dtype=dtype)
    result[:,:num.shape[1]]=num
    offset=num.shape[1]
    for j,categories in enumerate(state['categories']):
        lookup={c:k for k,c in enumerate(categories)}
        values=table['categorical'][rows,j]
        codes=np.array([lookup.get(v,-1) for v in values])
        keep=codes>=0
        result[np.flatnonzero(keep),offset+codes[keep]]=1
        offset+=len(categories)
    require(result.shape[1]==state['input_dim'] and np.isfinite(result).all(),'Invalid encoded features')
    return result


def lr_probability(table,rows,state):
    rows=np.asarray(rows,dtype=int)
    result=np.empty((len(rows),4),dtype=np.float64)
    for start in range(0,len(rows),2048):
        x=transform(table,rows[start:start+2048],state,dtype=np.float64)
        result[start:start+len(x)]=softmax(x@np.asarray(state['coef']).T+np.asarray(state['intercept']))
    return result


def check_bundle(bundle,expected_manifest=None):
    bundle=Path(bundle).resolve()
    if expected_manifest is not None:
        require(sha(bundle/'manifest.json')==expected_manifest,'Packet manifest differs')
    manifest=read(bundle/'manifest.json')
    expected=set(manifest['files'])|{'manifest.json'}
    actual={p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    require(actual==expected,'Packet inventory differs')
    for name,record in manifest['files'].items():
        path=(bundle/name).resolve()
        require(path.is_relative_to(bundle),'Packet member escapes bundle')
        require(path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],'Packet asset differs: '+name)
    return manifest


def pilot_gate(y,lr,p):
    base=(lr*WEIGHTS).argmax(1)
    tab=(p*WEIGHTS).argmax(1)
    oracle=np.where((base!=y)&(tab==y),tab,base)
    corrections=int(((base!=y)&(tab==y)).sum())
    result=dict(baseline=macro_f1(y,base),primary=macro_f1(y,blended(lr,p)),
                oracle=macro_f1(y,oracle),corrected=corrections,
                minimum_corrections=max(20,int(math.ceil(.01*(base!=y).sum()))),
                oracle_is_unattainable=True)
    result['passed']=bool(result['primary']>=result['baseline'] and
                          result['oracle']-result['baseline']>=.002 and corrections>=result['minimum_corrections'])
    return result
