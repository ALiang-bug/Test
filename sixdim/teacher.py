"""Blinded teacher generation, machine QC and hash-bound HUMAN review.
No network is used unless --allow-network is explicitly supplied. Offline replay
is provided for validating plumbing, not for asserting annotation correctness.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from .core import Model,REPORT,validate,fingerprint,input_evidence,DRAFT_TASK,CRITIQUE_TASK
from .io import load,write,unique,image_path

CHECKS=('evidence_grounded','no_fabricated_facts','dimension_boundaries','citations_verified','intent_restraint')


def local_evidence(sample):
    return input_evidence(sample['text'])


def evidence_for(sample):
    evidence=local_evidence(sample)+copy.deepcopy(sample.get('evidence',[]))
    unique(evidence,'id')
    for e in evidence[2:]:
        if e.get('kind') not in ('web','web_snippet','document','analogy'):
            raise ValueError('Unrecognized external evidence kind')
        if not isinstance(e.get('excerpt'),str) or not e['excerpt'].strip() or not e.get('source'):
            raise ValueError('External evidence requires excerpt and source')
    return evidence


def annotation_hash(row):
    return fingerprint({k:row.get(k) for k in ('sample_id','event_group_id','split','text','image_path','image_sha256','evidence','report','draft','critique','changed_claims','teacher_model','source_fingerprint','student_model','stage','observation_report','grouping_status','fake_cls')})


def check(row):
    errors=[];warnings=[]
    try:
        evidence=row.get('evidence',[]); unique(evidence,'id')
        validate(row['report'],{e['id'] for e in evidence})
        if row.get('stage')=='A':validate(row['observation_report'],{'I0','T0'})
        if row.get('stage')=='B':
            validate(row['draft'],{'I0','T0'})
            for k in ('issues','changes','unresolved'):
                if not isinstance(row['critique'].get(k),list): raise ValueError('Invalid critique')
            if not isinstance(row.get('changed_claims'),list):raise ValueError('Missing changed_claims')
            for c in row['changed_claims']:
                if not isinstance(c,dict) or not c.get('claim') or c.get('action') not in ('retain','retract','revise','defer'):
                    raise ValueError('Invalid change record')
                if not isinstance(c.get('evidence_ids'),list) or not set(c['evidence_ids']).issubset({e['id'] for e in evidence}):
                    raise ValueError('Invalid change citation')
        if row.get('fake_cls') and row['report']['benchmark_pred']!=row['fake_cls']:
            warnings.append('Teacher disagrees with source label; requires adjudication, not label-forced rationalization')
        if row['report']['factual_verdict']!='insufficient' and len(evidence)<=2:
            warnings.append('Definite factual verdict without external evidence; human must inspect visible/logical support')
    except (ValueError,KeyError,TypeError,AttributeError) as exc:errors.append(str(exc))
    return {'structural_pass':not errors,'errors':errors,'warnings':warnings,'human_review_required':True}


class Replay:
    def __init__(self,responses):
        self.responses=responses;self.model='offline-fixture';self.calls=[]
    def ask(self,task,payload,image=None):
        phase=payload['phase'];sid=payload['sample_id']
        self.calls.append({'phase':phase,'offline':True})
        return copy.deepcopy(self.responses[sid][phase])


def generate(sample,model,image_root,stage='A',verifier=None):
    if sample.get('split') not in ('train','dev') or sample.get('grouping_status')!='reviewed':
        raise ValueError('Only reviewed event-grouped train/dev inputs allowed; never test')
    if not sample.get('event_group_id'):raise ValueError('Missing event group')
    path=image_path(image_root,sample['image_path'])
    if stage=='B' and (not sample.get('draft') or not sample.get('student_model')):
        raise ValueError('Stage B requires actual student draft and student_model provenance')
    # Only the explicit allowlist reaches the teacher; never original fake_cls or gt_answers.
    payload={'sample_id':sample['sample_id'],'text':sample['text']}
    row={k:copy.deepcopy(sample[k]) for k in ('sample_id','event_group_id','split','grouping_status','text','image_path','image_sha256','fake_cls','student_model') if k in sample}
    import hashlib
    row['image_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    if sample.get('image_sha256') and sample['image_sha256']!=row['image_sha256']:raise ValueError('Image changed since split audit')
    row.update(stage=stage,review_status='pending',teacher_model=getattr(model,'model','unknown'),source_fingerprint=fingerprint(sample))
    evidence=evidence_for(sample);row['evidence']=evidence
    if stage=='A':
        initial=model.ask(DRAFT_TASK,dict(payload,phase='observation',evidence=evidence[:2]),str(path))
        validate(initial,{'I0','T0'})
        row['observation_report']=initial
        row['report']=model.ask('Verify atomic claims using supplied sources. Preserve unknowns; do not invent evidence. Return report: '+json.dumps(REPORT),dict(payload,phase='verification',observation_report=initial,evidence=evidence),str(path))
    else:
        row['draft']=copy.deepcopy(sample['draft']);validate(row['draft'],{'I0','T0'})
        row['critique']=model.ask(CRITIQUE_TASK,dict(payload,phase='critique',draft=row['draft'],evidence=evidence),str(path))
        refined=model.ask('Return {"report": <complete report>, "changed_claims":[{"claim":"...","action":"retain|retract|revise|defer","evidence_ids":[]}]}. Report schema: '+json.dumps(REPORT),dict(payload,phase='refinement',draft=row['draft'],critique=row['critique'],evidence=evidence),str(path))
        row['report']=refined['report'];row['changed_claims']=refined['changed_claims']
    row['qc']=check(row)
    if row['qc']['structural_pass']:
        row['verifier_model']=getattr(verifier or model,'model','unknown')
        row['verifier']=(verifier or model).ask('In a separate review pass check the report against image/text/evidence; no ground-truth label is supplied. Return {"pass":true,"issues":[]}. This pass cannot replace human review.',dict(payload,phase='review',report=row['report'],observation_report=row.get('observation_report'),draft=row.get('draft'),critique=row.get('critique'),changed_claims=row.get('changed_claims'),evidence=evidence),str(path))
        v=row['verifier']
        if type(v.get('pass')) is not bool or not isinstance(v.get('issues'),list):raise ValueError('Invalid verifier output')
    row['annotation_hash']=annotation_hash(row)
    return row


def apply_reviews(rows,reviews):
    unique(rows);unique(reviews)
    by_id={x['sample_id']:x for x in reviews}
    if set(by_id)-{x['sample_id'] for x in rows}:raise ValueError('Review references unknown sample')
    result=[]
    for original in rows:
        row=copy.deepcopy(original);review=by_id.get(row['sample_id'])
        if review:
            if review.get('annotation_hash')!=annotation_hash(row):raise ValueError('Stale or mismatched review hash')
            if review.get('decision') not in ('approved','rejected') or not review.get('reviewer') or not review.get('reviewed_at'):
                raise ValueError('Review requires decision, human reviewer and timestamp')
            if any(type(review.get('checks',{}).get(c)) is not bool for c in CHECKS):raise ValueError('Every review requires all boolean checklist judgments')
            if review['decision']=='approved':
                if not check(row)['structural_pass']:raise ValueError('Cannot approve structurally invalid annotation')
                if any(review.get('checks',{}).get(c) is not True for c in CHECKS):raise ValueError('All human checks required')
                if row.get('fake_cls') and row['fake_cls']!=row['report']['benchmark_pred'] and not review.get('adjudication'):
                    raise ValueError('Label disagreement needs explicit human adjudication')
                if row.get('verifier',{}).get('pass') is False and not review.get('adjudication'):
                    raise ValueError('Verifier disagreement needs human adjudication')
            row['review_status']=review['decision'];row['human_review']=review
        result.append(row)
    return result


def summary(rows):
    n=len(rows); valid=sum(bool(x.get('qc',{}).get('structural_pass')) for x in rows)
    reviewed=[x for x in rows if x.get('human_review')]
    good=sum(all(x['human_review'].get('checks',{}).get(c) is True for c in CHECKS) for x in reviewed)
    critical=sum(x['human_review'].get('checks',{}).get('no_fabricated_facts') is False for x in reviewed)
    count=len(reviewed)
    return {'total':n,'structural_pass':valid,
            'generation_failures':sum(x.get('generation_status')=='failed' for x in rows),
            'human_reviewed':count,'human_approved':sum(x.get('review_status')=='approved' for x in rows),
            'structural_pass_rate':valid/n if n else None,
            'human_checklist_pass_rate':good/count if count else None,
            'human_critical_error_rate':critical/count if count else None,
            'pilot_ready_for_scale':bool(n>=500 and count>=100 and valid/n>=.98 and good/count>=.90 and critical/count<=.05),
            'note':'Checklist rates require genuine human judgments; machine success is not factual correctness. Pilot sample must be representative.'}


def pilot_rows(rows,limit,seed):
    from collections import defaultdict
    import random
    if limit>=len(rows):return list(rows)
    buckets=defaultdict(list)
    for row in rows:buckets[(row.get('split'),row.get('fake_cls','unknown'))].append(row)
    rng=random.Random(seed);counts={k:int(len(v)*limit/len(rows)) for k,v in buckets.items()}
    remaining=limit-sum(counts.values())
    order=sorted(buckets,key=lambda k:(-(len(buckets[k])*limit/len(rows)-counts[k]),str(k)))
    for k in order[:remaining]:counts[k]+=1
    result=[]
    for k in sorted(buckets,key=str):
        bucket=sorted(buckets[k],key=lambda x:x['sample_id']);rng.shuffle(bucket);result+=bucket[:counts[k]]
    rng.shuffle(result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['generate','drafts','review','qc'])
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--image-root');p.add_argument('--stage',choices=['A','B'],default='A')
    p.add_argument('--responses');p.add_argument('--reviews');p.add_argument('--allow-network',action='store_true')
    p.add_argument('--base-url');p.add_argument('--model');p.add_argument('--verifier-model');p.add_argument('--verifier-base-url');p.add_argument('--limit',type=int,default=500);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();rows=load(a.input);unique(rows)
    if Path(a.output).exists():raise FileExistsError(a.output)
    if a.command in ('generate','drafts'):
        if not a.image_root:p.error('--image-root required')
        if a.limit<1:p.error('--limit must be positive')
        if a.responses:
            if a.allow_network:p.error('Replay and network are mutually exclusive')
            model=Replay(json.loads(Path(a.responses).read_text(encoding='utf-8')))
        else:
            if not a.allow_network or not a.base_url or not a.model:
                p.error('Live generation requires explicit --allow-network, --base-url, --model; default makes zero API calls')
            model=Model(a.base_url,a.model,os.environ.get('MAR_API_KEY' if a.command=='drafts' else 'MAR_TEACHER_API_KEY','EMPTY'))
        verifier=None
        if a.verifier_model:
            if not a.allow_network:p.error('Verifier network also requires --allow-network')
            verifier=Model(a.verifier_base_url or a.base_url,a.verifier_model,os.environ.get('MAR_VERIFIER_API_KEY',os.environ.get('MAR_TEACHER_API_KEY','EMPTY')))
        if any(x.get('split') not in ('train','dev') or x.get('grouping_status')!='reviewed' for x in rows):
            p.error('All inputs must have reviewed train/dev event groups; no test/excluded samples')
        outputs=[]
        Path(a.output).parent.mkdir(parents=True,exist_ok=True)
        output_handle=open(a.output,'x',encoding='utf-8')
        for row in pilot_rows(rows,a.limit,a.seed):
            try:
                start=len(model.calls);verifier_start=len(verifier.calls) if verifier else 0
                if a.command=='drafts':
                    if row.get('split') not in ('train','dev') or row.get('grouping_status')!='reviewed':raise ValueError('Reviewed training/development grouping required')
                    image=image_path(a.image_root,row['image_path'])
                    draft=model.ask(DRAFT_TASK,{'sample_id':row['sample_id'],'text':row['text'],'phase':'student_draft','evidence':local_evidence(row)},str(image))
                    validate(draft,{'I0','T0'})
                    out={k:copy.deepcopy(row[k]) for k in ('sample_id','event_group_id','grouping_status','split','text','image_path','image_sha256','fake_cls','evidence') if k in row}
                    out.update(draft=draft,student_model=model.model,review_status='pending')
                else:
                    out=generate(row,model,a.image_root,a.stage,verifier)
                out.update(generation_status='ok',calls=model.calls[start:],verifier_calls=verifier.calls[verifier_start:] if verifier else [],
                           generation_config={'seed':a.seed,'limit':a.limit,'mode':'offline_replay' if a.responses else 'explicit_network','input_fingerprint':fingerprint(rows)})
                outputs.append(out)
            except Exception as exc:
                outputs.append({'sample_id':row['sample_id'],'split':row.get('split'),'review_status':'pending',
                                'generation_status':'failed','error':str(exc),'qc':{'structural_pass':False}})
            output_handle.write(json.dumps(outputs[-1],ensure_ascii=False)+'\n');output_handle.flush()
        output_handle.close()
    elif a.command=='review':
        if not a.reviews:p.error('--reviews required')
        outputs=apply_reviews(rows,load(a.reviews))
    else:
        outputs=[]
        for row in rows:
            row=copy.deepcopy(row);row['qc']=check(row);outputs.append(row)
    if a.command not in ('generate','drafts'):write(a.output,outputs)
    print(json.dumps(summary(outputs)))

if __name__=='__main__':main()
