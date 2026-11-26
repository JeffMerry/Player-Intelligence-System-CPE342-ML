"""Offline GPU-only full TabM experiment; CPU smoke never exports a candidate."""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.dont_write_bytecode=True
import numpy as np
import torch
import torch.nn.functional as F

from task2_tabm_common import (MODEL,TRAIN,WEIGHTS,require,sha,read,write,row_digest,array_digest,
    macro_f1,softmax,blended,load_table,transform,lr_probability,check_bundle,pilot_gate)


def load_vendor(bundle):
    for wheel in sorted((Path(bundle)/'vendor').glob('*.whl')):
        if str(wheel) not in sys.path:
            sys.path.insert(0,str(wheel))
    import tabm
    import rtdl_num_embeddings
    require(tabm.__version__=='0.0.3' and rtdl_num_embeddings.__version__=='0.0.12','Wrong vendor version')
    return tabm


def setup():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True


def make_model(tabm,n_features,seed,device):
    torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
    return tabm.TabM.make(n_num_features=n_features,**MODEL).to(device=device,dtype=torch.float32)


def member_loss(logits,y):
    require(logits.ndim==3 and logits.shape[1:]==(16,4),'Wrong member-logit shape')
    return F.cross_entropy(logits.flatten(0,1),y.repeat_interleave(logits.shape[1]))


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_pt(path,value):
    path=Path(path)
    temp=path.with_suffix('.tmp.pt')
    torch.save(value,temp)
    for attempt in range(50):
        try:
            temp.replace(path)
            return
        except PermissionError:
            if attempt==49:
                raise
            time.sleep(.1)


@torch.inference_mode()
def predict(model,x,device,batch=512):
    model.eval()
    p=np.empty((len(x),4),dtype=np.float32)
    for start in range(0,len(x),batch):
        logits=model(torch.from_numpy(x[start:start+batch]).to(device))
        p[start:start+len(logits)]=logits.softmax(-1).mean(1).cpu().numpy()
    require(np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6),'Invalid neural probabilities')
    return p


def update(out,phase,**kw):
    value=dict(phase=phase,pid=os.getpid(),unix_time=time.time(),**kw)
    write(Path(out)/'progress.json',value)
    print(json.dumps(value,allow_nan=False),flush=True)


def sample_rows(rows,y,count):
    rows=np.asarray(rows,dtype=int)
    selected=[]
    for c in range(4):
        selected.extend(rows[y[rows]==c][:max(1,count//4)].tolist())
    return np.asarray(selected,dtype=int)


def device_preflight(tabm,n_features,device):
    model=make_model(tabm,n_features,991,device)
    n=512 if device.type=='cuda' else 4
    x=torch.randn(n,n_features,device=device,dtype=torch.float32)
    y=torch.arange(n,device=device)%4
    optimizer=torch.optim.AdamW(model.parameters(),lr=TRAIN['lr'],weight_decay=TRAIN['weight_decay'])
    initial=state_hash(model.state_dict())
    optimizer.zero_grad(set_to_none=True)
    logits=model(x)
    loss=member_loss(logits,y)
    logs=logits.detach().cpu().numpy().astype(np.float64)
    lse=np.log(np.exp(logs-logs.max(-1,keepdims=True)).sum(-1))+logs.max(-1)
    target=np.broadcast_to(y.cpu().numpy()[:,None,None],(n,16,1))
    numpy_loss=float((lse-np.take_along_axis(logs,target,axis=-1)[...,0]).mean())
    require(abs(numpy_loss-float(loss.detach()))<1e-6,'Independent member CE differs')
    loss.backward()
    require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()),'Invalid preflight gradients')
    optimizer.step()
    require(state_hash(model.state_dict())!=initial,'Preflight weights did not update')
    result=dict(status='PASS',device=str(device),batch=n,shape=list(logits.shape),
                per_member_ce_numpy_difference=abs(numpy_loss-float(loss.detach())),
                parameters=sum(p.numel() for p in model.parameters()),weights_updated=True,
                gpu_peak_bytes=torch.cuda.max_memory_allocated(0) if device.type=='cuda' else None)
    del model,optimizer,x,y,logits,loss
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return result


def fit_stage(bundle,out,role,table,splits,roles,tabm,device,smoke=False,fixed_epochs=None,test=None):
    folder=Path(out)/role
    folder.mkdir(exist_ok=True)
    state=read(Path(bundle)/'lr_states'/f'{role}.json')
    plan=roles[role]
    y=splits['y']
    full_fit=np.asarray(splits[f'{role}_fit'])
    full_stop=np.asarray(splits[f'{role}_stop'])
    full_query=np.asarray(splits[f'{role}_query'])
    fit=sample_rows(full_fit,y,64) if smoke else full_fit
    stop=sample_rows(full_stop,y,32) if smoke and len(full_stop) else full_stop
    query=sample_rows(full_query,y,32) if smoke and len(full_query) else full_query
    query_table=table if test is None else test
    if test is not None:
        query=np.arange(len(test['ids']))
    require(state['fit_digest']==plan['fit_digest']==row_digest(full_fit,y),'Role preprocessing fit rows differ')
    require(not(set(fit)&set(stop)) and (test is not None or not(set(fit)&set(query))),'Neural fit/stop/query overlap')
    require(not(set(stop)&set(query)),'Stop/query overlap')
    if plan['select']:
        require(len(stop)>0 and fixed_epochs is None,'Selection stage needs separate stop rows')
    else:
        require(not len(stop) and fixed_epochs is not None,'Refit needs fixed epochs/no stop')
    train_cfg={**TRAIN,**(dict(max_epochs=2,patience=2,batch_size=32) if smoke else {})}
    target_epochs=train_cfg['max_epochs'] if plan['select'] else int(fixed_epochs)
    require(0<target_epochs<=train_cfg['max_epochs'],'Invalid epoch budget')
    identity=dict(packet_sha=sha(Path(bundle)/'manifest.json'),role=role,smoke=smoke,
                  model=MODEL,training=train_cfg,seed=plan['seed'],input_dim=state['input_dim'],
                  declared_fit_digest=plan['fit_digest'],actual_fit_digest=row_digest(fit,y),
                  actual_stop_digest=None if not len(stop) else row_digest(stop,y),
                  fixed_epochs=fixed_epochs,query_source='test' if test is not None else 'train')
    if (folder/'stage.json').exists():
        record=read(folder/'stage.json')
        require(record['identity']==identity and sha(folder/'best.pt')==record['checkpoint_sha'],'Completed stage identity/hash differs')
        return record
    xfit=transform(table,fit,state)
    if not smoke:
        require(array_digest(xfit)==plan['fit_feature_digest'],'Encoded training features differ from prepared input')
    xstop=transform(table,stop,state) if len(stop) else None
    lr_stop=lr_probability(table,stop,state) if len(stop) else None
    golden=np.load(Path(bundle)/'golden_lr.npz')
    if len(stop) and not smoke:
        np.testing.assert_allclose(lr_stop,golden[f'{role}_stop'],rtol=0,atol=1e-8)
    xquery=transform(query_table,query,state) if len(query) else None
    lr_query=lr_probability(query_table,query,state) if len(query) else None
    if len(query) and not smoke:
        np.testing.assert_allclose(lr_query,golden[f'{role}_query'],rtol=0,atol=1e-8)
    model=make_model(tabm,state['input_dim'],plan['seed'],device)
    initial_hash=state_hash(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=train_cfg['lr'],weight_decay=train_cfg['weight_decay'])
    tx=torch.from_numpy(xfit).to(device)
    ty=torch.from_numpy(y[fit]).long().to(device)
    history=[]; best_score=-1.; best_epoch=0; start_epoch=1
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
        require(last['identity']==identity,'Partial checkpoint identity differs')
        require(last['runtime']==dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device)),'Cannot resume optimizer under a different runtime')
        model.load_state_dict(last['state'])
        optimizer.load_state_dict(last['optimizer'])
        history,best_score,best_epoch=last['history'],last['best_score'],last['best_epoch']
        initial_hash=last['initial_state_hash']
        start_epoch=last['epoch']+1
        torch.set_rng_state(last['rng_cpu'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state(last['rng_cuda'].cpu(),0)
    for epoch in range(start_epoch,target_epochs+1):
        model.train()
        generator=torch.Generator().manual_seed(plan['seed']+epoch)
        order=torch.randperm(len(fit),generator=generator)
        loss_total=0.; seen=0
        started=time.perf_counter()
        for start in range(0,len(fit),train_cfg['batch_size']):
            idx=order[start:start+train_cfg['batch_size']].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=member_loss(model(tx[idx]),ty[idx])
            require(bool(torch.isfinite(loss)),'Nonfinite training loss')
            loss.backward(); optimizer.step()
            loss_total+=float(loss.detach())*len(idx); seen+=len(idx)
        require(seen==len(fit),'Not all training rows processed')
        info=dict(epoch=epoch,train_member_ce=loss_total/seen,processed_rows=seen,
                  shuffle_digest=array_digest(fit[order.numpy()]),seconds=time.perf_counter()-started)
        if plan['select']:
            p=predict(model,xstop,device)
            value=macro_f1(y[stop],blended(lr_stop,p))
            info['stop_primary_macro_f1']=value
            if value>best_score:
                best_score,best_epoch=value,epoch
                save_pt(folder/'best.pt',dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        identity=identity,epoch=epoch,initial_state_hash=initial_hash))
        history.append(info)
        save_pt(folder/'last.pt',dict(state=model.state_dict(),optimizer=optimizer.state_dict(),identity=identity,
                epoch=epoch,history=history,best_score=best_score,best_epoch=best_epoch,initial_state_hash=initial_hash,
                runtime=dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device)),
                rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state(0) if device.type=='cuda' else None))
        write(folder/'history.json',history)
        update(out,'epoch',role=role,**info)
        if plan['select'] and epoch-best_epoch>=train_cfg['patience']:
            break
    if not plan['select']:
        best_epoch=target_epochs
        save_pt(folder/'best.pt',dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                identity=identity,epoch=target_epochs,initial_state_hash=initial_hash))
    checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    require(checkpoint['epoch']==best_epoch,'Selected checkpoint epoch differs')
    model.load_state_dict(checkpoint['state'])
    require(state_hash(checkpoint['state'])!=initial_hash,'Training did not change model weights')
    if len(stop):
        stop_p=predict(model,xstop,device)
        require(macro_f1(y[stop],blended(lr_stop,stop_p))==best_score,'Saved best checkpoint does not reproduce stop F1')
        np.savez_compressed(folder/'stop.npz',rows=stop,tabm=stop_p,lr=lr_stop)
    if len(query):
        query_p=predict(model,xquery,device)
        np.savez_compressed(folder/'query.npz',rows=query,tabm=query_p,lr=lr_query,
                            primary=blended(lr_query,query_p),ids=query_table['ids'][query])
    # Saved proper-role sample enables bounded CPU inference replay after cloud return.
    replay_x=xquery if len(query) else xstop
    replay_rows=query if len(query) else stop
    require(replay_x is not None and len(replay_x)>0,'No replay query')
    take=np.arange(min(32,len(replay_x)))
    model.eval()
    with torch.inference_mode():
        replay_logits=model(torch.from_numpy(replay_x[take]).to(device))
        replay_p=replay_logits.softmax(-1).mean(1).cpu().numpy()
        logits=replay_logits.cpu().numpy()
    numpy_p=softmax(logits).mean(1)
    require(np.max(np.abs(numpy_p-replay_p))<1e-6,'Independent softmax/member averaging differs')
    np.savez_compressed(folder/'replay.npz',rows=replay_rows[take],x=replay_x[take],logits=logits,
                        probability=replay_p,numpy_probability=numpy_p,
                        ids=query_table['ids'][replay_rows[take]] if len(query) else table['ids'][replay_rows[take]])
    record=dict(identity=identity,selected_epoch=best_epoch,executed_epochs=len(history),
                best_stop_macro_f1=best_score if plan['select'] else None,fit_count=len(fit),
                stop_count=len(stop),query_count=len(query),checkpoint_sha=sha(folder/'best.pt'),
                history_sha=sha(folder/'history.json'),initial_state_hash=initial_hash,
                trained_state_hash=state_hash(checkpoint['state']),replay_sha=sha(folder/'replay.npz'),
                runtime=dict(torch=str(torch.__version__),numpy=np.__version__,device=str(device)),
                gpu_peak_bytes=torch.cuda.max_memory_allocated(0) if device.type=='cuda' else None)
    write(folder/'stage.json',record)
    del model,optimizer,tx,ty,xfit,xquery,xstop
    gc.collect()
    if device.type=='cuda':
        torch.cuda.empty_cache()
    return record


def merge_csv(reference,predictions):
    with Path(reference).open(encoding='utf-8-sig',newline='') as f:
        original=list(csv.reader(f))
    require(original[0]==['id','task1','task2','task3','task4','task5'] and len(original)==len(predictions)+1,'Reference schema differs')
    result=[r.copy() for r in original]
    require([r[0] for r in result[1:]]==[f'ANS{i:05d}' for i in range(1,len(predictions)+1)],'ANS order differs')
    require(np.isin(predictions,range(4)).all(),'Invalid Task2 output')
    for row,value in zip(result[1:],predictions):
        row[2]=str(int(value))
    require(all(all(a[j]==b[j] for j in [0,1,3,4,5]) for a,b in zip(original,result)),'Non-target string changed')
    return result


def run(bundle,out,mode='run'):
    bundle,out=Path(bundle).resolve(),Path(out).resolve()
    require(mode in ['run','smoke'],'Wrong mode')
    manifest=check_bundle(bundle)
    require(not out.is_relative_to(bundle),'Outputs must be outside immutable packet')
    setup()
    if mode=='run':
        require(torch.cuda.is_available(),'Full TabM run requires CUDA; CPU allowed only for smoke')
        require(torch.cuda.get_device_properties(0).total_memory>=4*2**30,'Insufficient actual GPU VRAM')
    device=torch.device('cuda:0' if mode=='run' else 'cpu')
    recipe=read(bundle/'recipe.json')
    require(recipe['model']==MODEL and recipe['training']==TRAIN,'Recipe differs')
    if (out/'report.json').exists():
        report=read(out/'report.json')
        require(report['packet_sha']==sha(bundle/'manifest.json') and report['mode']==mode,'Completed run identity differs')
        return report
    require(not out.exists() or (out/'run_meta.json').exists() or not any(out.iterdir()),'Output folder is not owned by this run')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run_meta.json').exists():
        require(read(out/'run_meta.json')['packet_sha']==sha(bundle/'manifest.json'),'Run packet changed')
    else:
        write(out/'run_meta.json',dict(packet_sha=sha(bundle/'manifest.json'),mode=mode,torch=str(torch.__version__),
              numpy=np.__version__,cuda=torch.version.cuda,device=str(device),
              visible_gpu_count=torch.cuda.device_count(),gpu_names=[torch.cuda.get_device_name(k) for k in range(torch.cuda.device_count())]))
    tabm=load_vendor(bundle)
    roles=read(bundle/'roles.json')
    splits=np.load(bundle/'splits.npz',allow_pickle=False)
    table=load_table(bundle/'train.csv',read(bundle/'schema.json'),recipe['train_rows'])
    np.testing.assert_array_equal(table['ids'],splits['ids'])
    np.testing.assert_array_equal(table['y'],splits['y'])
    preflight=device_preflight(tabm,roles['pilot']['input_dim'],device)
    write(out/'preflight.json',preflight)
    fit_stage(bundle,out,'pilot',table,splits,roles,tabm,device,smoke=mode=='smoke')
    pilot=np.load(out/'pilot/query.npz')
    gate=pilot_gate(table['y'][pilot['rows']],pilot['lr'],pilot['tabm'])
    write(out/'pilot_gate.json',gate)
    report=dict(mode=mode,packet_sha=sha(bundle/'manifest.json'),pilot_gate=gate,
                development_passed=False,audit_passed=False,eligible=False,public_score=None,
                neural_full_quality_result=False,test_table_parsed=False)
    if mode=='smoke':
        # Exercise a fixed-epoch fresh refit independently from pilot epoch selection.
        fit_stage(bundle,out,'outer0',table,splits,roles,tabm,device,smoke=True,fixed_epochs=2)
        report['status']='SMOKE_ONLY_NOT_A_QUALITY_RESULT'
        write(out/'report.json',report)
        update(out,'completed_smoke',public_score=None)
        return report
    if not gate['passed']:
        report['status']='PILOT_SCREEN_FAILED_NO_FIVEFOLD_OR_EXPORT'
        write(out/'report.json',report)
        update(out,'pilot_stopped',gate=gate)
        return report
    dev,fold=splits['dev'],splits['fold']
    baseline=splits['baseline']; primary=np.zeros(len(dev),dtype=int)
    for k in range(5):
        selected=fit_stage(bundle,out,f'inner{k}',table,splits,roles,tabm,device)
        fit_stage(bundle,out,f'outer{k}',table,splits,roles,tabm,device,fixed_epochs=selected['selected_epoch'])
        values=np.load(out/f'outer{k}/query.npz')
        q=np.flatnonzero(fold==k)
        np.testing.assert_array_equal(values['rows'],dev[q])
        np.testing.assert_array_equal((values['lr']*WEIGHTS).argmax(1),baseline[q])
        primary[q]=values['primary']
    y=splits['y'][dev]
    fg=np.array([macro_f1(y[fold==k],primary[fold==k])-macro_f1(y[fold==k],baseline[fold==k]) for k in range(5)])
    gain=macro_f1(y,primary)-macro_f1(y,baseline)
    se=float(fg.std(ddof=1)/np.sqrt(5))
    passed=bool(gain>=.002 and fg.mean()>se)
    report.update(status='DEVELOPMENT_COMPLETED',neural_full_quality_result=True,
                  baseline_macro_f1=macro_f1(y,baseline),primary_macro_f1=macro_f1(y,primary),
                  gain=gain,paired_fold_gains=fg.tolist(),fold_gain_se=se,development_passed=passed)
    np.savez_compressed(out/'oof.npz',primary=primary,baseline=baseline,target=y,fold=fold)
    write(out/'frozen_development_decision.json',report)
    if passed:
        selected=fit_stage(bundle,out,'dev_inner',table,splits,roles,tabm,device)
        fit_stage(bundle,out,'development',table,splits,roles,tabm,device,fixed_epochs=selected['selected_epoch'])
        values=np.load(out/'development/query.npz')
        np.testing.assert_array_equal((values['lr']*WEIGHTS).argmax(1),splits['audit_baseline'])
        audit_y=splits['y'][splits['audit']]
        report.update(audit_macro_f1=macro_f1(audit_y,values['primary']),baseline_audit_macro_f1=macro_f1(audit_y,splits['audit_baseline']))
        report['audit_passed']=report['audit_macro_f1']>=report['baseline_audit_macro_f1']-.001
        report['eligible']=bool(report['audit_passed'])
        if report['eligible']:
            test=load_table(bundle/'test.csv',read(bundle/'schema.json'),recipe['test_rows'],target=False)
            np.testing.assert_array_equal(test['ids'],splits['test_ids'])
            fit_stage(bundle,out,'final',table,splits,roles,tabm,device,fixed_epochs=selected['selected_epoch'],test=test)
            values=np.load(out/'final/query.npz')
            pred=values['primary']
            require(sha(bundle/'reference.csv')==recipe['reference']['sha256'],'Frozen scored reference changed')
            rows=merge_csv(bundle/'reference.csv',pred)
            path=out/'submission_task2_tabm_unscored.csv'
            require(not path.exists(),'Refuse to overwrite candidate')
            with path.open('x',encoding='utf-8',newline='') as f:
                csv.writer(f).writerows(rows)
            write(out/'candidate.json',dict(task=2,reference=recipe['reference'],submission=path.name,
                  sha256=sha(path),public_score=None,unchanged_columns=['id','task1','task3','task4','task5']))
            report['test_table_parsed']=True
    write(out/'report.json',report)
    update(out,'completed',**{k:v for k,v in report.items() if k not in ['packet_sha']})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['run','smoke'],default='run')
    args=parser.parse_args()
    run(args.bundle,args.output,args.mode)
