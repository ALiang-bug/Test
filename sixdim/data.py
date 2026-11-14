"""Human-reviewed annotations to train/dev SFT, with shared inference prompts."""
import argparse
import json
from .core import POLICY,validate,DRAFT_TASK,CRITIQUE_TASK,REFINE_TASK,input_evidence
from .teacher import annotation_hash,CHECKS,check
from .io import load,write,image_path,unique


def export(records,image_root,stage,split='train'):
    if split not in ('train','dev'):raise ValueError('Test data must never enter SFT')
    unique(records);output=[]
    for record in records:
        review=record.get('human_review',{})
        if record.get('split')!=split or record.get('review_status')!='approved' or record.get('grouping_status')!='reviewed':
            raise ValueError('Only human-approved event-grouped annotations in the requested split are exportable')
        if review.get('decision')!='approved' or review.get('annotation_hash')!=annotation_hash(record) or not review.get('reviewer') or not review.get('reviewed_at'):
            raise ValueError('Missing or stale human review')
        if any(review.get('checks',{}).get(c) is not True for c in CHECKS) or not check(record)['structural_pass']:
            raise ValueError('Annotation has not passed all checks')
        if record.get('stage')!=stage or not record.get('event_group_id'):raise ValueError('Stage or event mismatch')
        image=image_path(image_root,record['image_path'])
        import hashlib
        if not record.get('image_sha256') or hashlib.sha256(image.read_bytes()).hexdigest()!=record['image_sha256']:
            raise ValueError('Image changed or missing reviewed image hash')
        evidence=record['evidence'];report=validate(record['report'],{e['id'] for e in evidence})
        meta={'images':[str(image)],'sample_id':record['sample_id'],'event_group_id':record['event_group_id'],
              'stage':stage,'split':split,'image_sha256':record['image_sha256'],'review_status':'approved','annotation_hash':annotation_hash(record)}
        def add(task,payload,target):
            output.append(dict(meta,messages=[{'role':'system','content':POLICY},
                {'role':'user','content':'<image>\n'+task+'\nINPUT:\n'+json.dumps(payload,ensure_ascii=False)},
                {'role':'assistant','content':json.dumps(target,ensure_ascii=False)}]))
        news={'sample_id':record['sample_id'],'text':record['text'],'evidence':input_evidence(record['text'])}
        if stage=='A':
            for d in report['dimensions']:
                selected=[e for e in evidence if e['id'] in d['evidence_ids']]
                add('Answer this evidence-grounded dimension question.',dict(news,dimension=d['dimension'],question=d['question'],evidence=selected),d)
            observation=record.get('observation_report')
            if observation is None:raise ValueError('Stage A needs a blinded observation report for draft supervision')
            validate(observation,{'I0','T0'});add(DRAFT_TASK,news,observation)
        elif stage=='B':
            if not record.get('draft') or not record.get('student_model'):raise ValueError('Actual student draft provenance required')
            add(CRITIQUE_TASK,{'news':news,'draft':record['draft'],'evidence':evidence},record['critique'])
            add(REFINE_TASK,{'news':news,'draft':record['draft'],'critique':record['critique'],'evidence':evidence},report)
        else:raise ValueError('Unknown stage')
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--image-root',required=True);p.add_argument('--stage',choices=['A','B'],required=True)
    p.add_argument('--split',choices=['train','dev'],default='train');a=p.parse_args()
    rows=export(load(a.input),a.image_root,a.stage,a.split);write(a.output,rows)
    print(json.dumps({'rows':len(rows),'stage':a.stage,'split':a.split}))

if __name__=='__main__':main()
