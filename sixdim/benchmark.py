"""Frozen, tool-free student benchmark. No inference without --allow-network."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .core import Model, POLICY, DRAFT_TASK, LABELS, fingerprint, input_evidence, validate
from .io import load, write, unique, image_path


def freeze(rows, image_root):
    unique(rows)
    if not rows: raise ValueError('Empty test set')
    frozen=[]
    for r in rows:
        if r.get('split')!='test' or not r.get('event_group_id'):
            raise ValueError('Use the split command test manifest')
        if r.get('fake_cls') not in LABELS: raise ValueError('Missing benchmark label')
        path=image_path(image_root,r['image_path'])
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if r.get('image_sha256') and r['image_sha256']!=digest:
            raise ValueError('Image changed since split')
        frozen.append({k:r[k] for k in ('sample_id','event_group_id','text','image_path','fake_cls','split')})
        frozen[-1].update(image_sha256=digest, grouping_status=r.get('grouping_status','provisional'))
    return frozen


def run(rows, image_root, model, output, run_id):
    unique(rows)
    # Validate the entire frozen set before making the first request.
    checked=freeze(rows,image_root)
    if fingerprint(checked)!=fingerprint(rows): raise ValueError('Use an unchanged frozen manifest')
    protocol=fingerprint({'system':POLICY,'task':DRAFT_TASK,'temperature':0,'max_tokens':4096,'tools':False,'backend':getattr(model,'protocol',{'backend':'openai-compatible'})})
    test_hash=fingerprint(rows)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    with open(output,'x',encoding='utf-8') as out:
        for row in rows:
            record=dict(row,run_id=run_id,model=model.model,protocol_hash=protocol,test_hash=test_hash,status='failed')
            try:
                # Ground-truth labels, paths and event metadata never enter the prompt.
                payload={'sample_id':row['sample_id'],'text':row['text'],'evidence':input_evidence(row['text'])}
                record['final']=validate(model.ask(DRAFT_TASK,payload,str(image_path(image_root,row['image_path']))),{'I0','T0'})
                record['status']='ok'
            except Exception as exc:
                record['error']=str(exc)
            record['model_provenance']=getattr(model,'provenance',{'served_model':model.model})
            record['inference_backend']=getattr(model,'protocol',{'backend':'openai-compatible'})
            out.write(json.dumps(record,ensure_ascii=False)+'\n');out.flush()
            print(row['sample_id'],record['status'],flush=True)


def score(rows):
    matrix=[[0]*4 for _ in LABELS]; failures=[0]*4; insufficient=0
    for row in rows:
        i=LABELS.index(row['fake_cls'])
        try:
            report=validate(row['final'],{'I0','T0'}) if row.get('status')=='ok' else None
        except (ValueError,KeyError,TypeError): report=None
        if report is None: failures[i]+=1;continue
        matrix[i][LABELS.index(report['benchmark_pred'])]+=1
        insufficient+=report['factual_verdict']=='insufficient'
    def metrics(m,missing):
        f=[]
        for i in range(len(m)):
            denominator=sum(m[i])+missing[i]+sum(r[i] for r in m)
            f.append(2*m[i][i]/denominator if denominator else 0)
        return {'accuracy':sum(m[i][i] for i in range(len(m)))/len(rows),'macro_f1':sum(f)/len(f),'per_class_f1':f}
    binary=[[0,0],[0,0]]
    for i in range(4):
        for j in range(4): binary[int(i!=0)][int(j!=0)]+=matrix[i][j]
    return {'total':len(rows),'failures':sum(failures),'valid_report_rate':1-sum(failures)/len(rows),
            'factual_insufficient_count':insufficient,'labels':LABELS,'confusion_matrix':matrix,
            'four_class':metrics(matrix,failures),'binary':metrics(binary,[failures[0],sum(failures[1:])]),
            'note':'Failures count as false negatives, not silently discarded. Structural validity is not reasoning correctness.'}


def compare(base, tuned):
    unique(base);unique(tuned)
    if not base or not tuned: raise ValueError('Empty predictions')
    b={r['sample_id']:r for r in base};t={r['sample_id']:r for r in tuned}
    if b.keys()!=t.keys():raise ValueError('Prediction sets differ; missing rows cannot be dropped')
    for field in ('test_hash','protocol_hash'):
        values={r.get(field) for r in base+tuned}
        if len(values)!=1 or None in values:raise ValueError('Mismatched or missing '+field)
    for sid in b:
        for field in ('fake_cls','text','image_sha256','event_group_id','image_path','split','grouping_status'):
            if b[sid].get(field)!=t[sid].get(field):raise ValueError('Sample content changed: '+sid)
    ordered=sorted(b)
    expected=fingerprint([{k:b[s][k] for k in ('sample_id','event_group_id','text','image_path','fake_cls','split','image_sha256','grouping_status')} for s in b])
    if expected!=base[0]['test_hash']: raise ValueError('Incomplete or altered frozen test predictions')
    before=score(base);after=score(tuned)
    def correct(r):
        return r.get('status')=='ok' and r.get('final',{}).get('benchmark_pred')==r['fake_cls'] and _valid(r)
    improved=sum(not correct(b[s]) and correct(t[s]) for s in ordered)
    regressed=sum(correct(b[s]) and not correct(t[s]) for s in ordered)
    return {'base':before,'tuned':after,'accuracy_delta':after['four_class']['accuracy']-before['four_class']['accuracy'],
            'macro_f1_delta':after['four_class']['macro_f1']-before['four_class']['macro_f1'],
            'wrong_to_correct':improved,'correct_to_wrong':regressed,
            'event_grouping_reviewed':all(r.get('grouping_status')=='reviewed' for r in base),
            'note':'Descriptive single-run comparison only. No causal or statistical significance claim; audit six-dimensional evidence separately.'}


def _valid(row):
    try:validate(row['final'],{'I0','T0'});return True
    except (ValueError,KeyError,TypeError):return False


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze');f.add_argument('--input',required=True);f.add_argument('--image-root',required=True);f.add_argument('--output',required=True)
    r=sub.add_parser('run');r.add_argument('--input',required=True);r.add_argument('--image-root',required=True);r.add_argument('--output',required=True)
    r.add_argument('--base-url');r.add_argument('--model',required=True);r.add_argument('--backend',choices=['api','local'],default='api');r.add_argument('--adapter');r.add_argument('--run-id',required=True);r.add_argument('--allow-network',action='store_true');r.add_argument('--execute-local',action='store_true')
    c=sub.add_parser('compare');c.add_argument('--base',required=True);c.add_argument('--tuned',required=True);c.add_argument('--output',required=True)
    a=p.parse_args()
    if a.command=='freeze':write(a.output,freeze(load(a.input),a.image_root))
    elif a.command=='run':
        if Path(a.output).exists():raise FileExistsError(a.output)
        rows=load(a.input)
        if fingerprint(freeze(rows,a.image_root))!=fingerprint(rows):raise ValueError('Use a frozen manifest')
        if a.backend=='local':
            if not a.execute_local:p.error('Local weights/inference require --execute-local')
            from .local_model import LocalModel
            model=LocalModel(a.model,a.adapter)
        else:
            if a.adapter:p.error('--adapter is for local backend; API must serve its adapter explicitly')
            if not a.allow_network or not a.base_url:p.error('API inference requires --allow-network and --base-url')
            model=Model(a.base_url,a.model,os.environ.get('MAR_API_KEY','EMPTY'))
        run(rows,a.image_root,model,a.output,a.run_id)
    else:
        result=compare(load(a.base),load(a.tuned));Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        with open(a.output,'x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
        print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
