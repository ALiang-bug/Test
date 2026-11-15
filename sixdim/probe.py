"""Teacher interface probes only: provisional samples never become SFT data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .core import Model,DRAFT_TASK,POLICY,validate,input_evidence,fingerprint
from .io import load,unique,image_path,write
from .semantic import audit
from .evidence_review import review_report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run');run.add_argument('--candidates',required=True);run.add_argument('--before',required=True)
    run.add_argument('--image-root',required=True);run.add_argument('--output',required=True)
    run.add_argument('--base-url',default='http://127.0.0.1:19887/v1');run.add_argument('--model',default='Qwen3-VL-32B-Instruct-FP8');run.add_argument('--allow-network',action='store_true')
    cmp=sub.add_parser('compare');cmp.add_argument('--before',required=True);cmp.add_argument('--after',required=True);cmp.add_argument('--output',required=True)
    a=p.parse_args();before=load(a.before);unique(before)
    if not before:raise ValueError('Empty reference probe')
    if Path(a.output).exists():raise FileExistsError(a.output)
    if a.command=='run':
        if not a.allow_network:p.error('Explicit --allow-network required')
        candidates=load(a.candidates);unique(candidates);mapping={r['sample_id']:r for r in candidates}
        chosen=[mapping[r['sample_id']] for r in before]
        paths=[image_path(a.image_root,r['image_path']) for r in chosen]
        if any(r.get('split') not in ('train','dev') for r in chosen):raise ValueError('No test probes for teacher tuning')
        model=Model(a.base_url,a.model,os.environ.get('MAR_TEACHER_API_KEY','EMPTY'))
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        with open(a.output,'x',encoding='utf-8') as out:
            for row,path in zip(chosen,paths):
                payload={'sample_id':row['sample_id'],'text':row['text'],'evidence':input_evidence(row['text'])}
                result={'sample_id':row['sample_id'],'purpose':'interface_probe_only','grouping_status':row['grouping_status'],
                        'review_status':'pending','teacher_model':a.model,'status':'failed','input':payload,
                        'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prompt_hash':fingerprint([POLICY,DRAFT_TASK])}
                try:
                    result['report']=validate(model.ask(DRAFT_TASK,payload,str(path)),{'I0','T0'})
                    result['semantic']=audit(result['report'])
                    result['evidence_review']=review_report(model,payload,result['report'],str(path),result['semantic'])
                    result['status']='ok'
                    result['machine_review_pass']=result['semantic']['semantic_pass'] and result['evidence_review']['review']['overall_pass']
                except Exception as exc:result['error']=str(exc)
                out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    else:
        after=load(a.after);unique(after);mapping={r['sample_id']:r for r in after}
        if set(mapping)!={r['sample_id'] for r in before}:raise ValueError('Probe sample IDs differ')
        result=[]
        for old in before:
            new=mapping[old['sample_id']]
            if old.get('teacher_model')!=new.get('teacher_model'):raise ValueError('Teacher model differs')
            for field in ('image_sha256','input'):
                if field in old and field in new and old[field]!=new[field]:raise ValueError('Probe input changed: '+field)
            entry={'sample_id':old['sample_id'],'before_status':old.get('status'),'after_status':new.get('status'),
                   'note':('Image/text evidence hashes checked; model weights are not independently hashed.' if all(k in old and k in new for k in ('image_sha256','input')) else 'Old probe omitted image/input hashes: same ID/model only; full input equality and correctness require human verification.')}
            for name,row in [('before',old),('after',new)]:
                try:
                    report=validate(row['report'],{'I0','T0'})
                    entry[name]={'semantic':audit(report),'report':report,'evidence_review':row.get('evidence_review'), 'machine_review_pass':row.get('machine_review_pass')}
                except (KeyError,ValueError,TypeError) as exc:entry[name]={'error':str(exc)}
            result.append(entry)
        write(a.output,result)
        print(json.dumps([{'sample_id':r['sample_id'],'before':r['before'].get('semantic'),'after':r['after'].get('semantic')} for r in result],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
