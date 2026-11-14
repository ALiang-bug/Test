import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from sixdim.core import REPORT,DIMENSIONS
from sixdim.split import partition
from sixdim.teacher import Replay,generate,apply_reviews,annotation_hash,CHECKS,summary
from sixdim.data import export
from sixdim.train import prepare,assistant_labels,language_targets


def report():
    r=copy.deepcopy(REPORT)
    r.update(claims=['A verifiable claim'],factual_verdict='insufficient',media_status='unknown',
             benchmark_pred='original',mechanisms=[],expression_strategies=[],verification_questions=[])
    for d in r['dimensions']:d.update(status='insufficient',evidence_ids=['I0','T0'])
    return r


def sample(sid,split='train'):
    return {'sample_id':sid,'event_group_id':'event-'+sid,'grouping_status':'reviewed',
            'image_path':sid+'.png','text':'Unique news about '+sid,'split':split,'fake_cls':'original'}


def review(row):
    return {'sample_id':row['sample_id'],'annotation_hash':annotation_hash(row),'decision':'approved',
            'reviewer':'offline-test-human','reviewed_at':'2026-11-14', 'checks':dict.fromkeys(CHECKS,True)}


def responses():
    return {'observation':report(),'verification':report(),'review':{'pass':True,'issues':[]},
            'critique':{'issues':[],'changes':[],'unresolved':[]},
            'refinement':{'report':report(),'changed_claims':[{'claim':'A claim','action':'defer','evidence_ids':['T0']}]},
            'student_draft':report()}


class PreparationTests(unittest.TestCase):
    def test_event_overlap_removed_from_pool_deterministic_and_inputs_unchanged(self):
        train=[sample('a'),sample('b'),sample('c'),sample('d')];test=[sample('test','test')]
        before=copy.deepcopy(train)
        mapping=[{'sample_id':x['sample_id'],'event_id':'same' if x['sample_id'] in ('a','test') else x['sample_id'],
                  'review_status':'approved'} for x in train+test]
        out,audit=partition(train,test,mapping)
        self.assertEqual([x['sample_id'] for x in out['excluded']],['a'])
        self.assertEqual(out,partition(train,test,mapping)[0]);self.assertEqual(train,before)
        self.assertTrue(audit['event_grouping_reviewed'])
        groups=[{x['event_group_id'] for x in out[k]} for k in ('train','dev','test')]
        self.assertFalse(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
        with self.assertRaises(ValueError):partition(train,test,mapping[:-1])

    def test_identical_media_connects_different_texts_and_provisional_is_not_certified(self):
        with tempfile.TemporaryDirectory() as d:
            a=sample('a');b=sample('b','test')
            for x in (a,b):(Path(d)/x['image_path']).write_bytes(b'identical-image')
            out,audit=partition([a],[b],image_root=d)
            self.assertEqual(len(out['excluded']),1);self.assertFalse(audit['event_grouping_reviewed'])
            self.assertEqual(out['test'][0]['grouping_status'],'provisional')

    def test_teacher_blinded_review_hash_and_export_gate(self):
        class Fake(Replay):
            def ask(self,task,payload,image=None):
                self.assert_blinded(payload)
                return super().ask(task,payload,image)
            def assert_blinded(self,payload):
                assert 'fake_cls' not in json.dumps(payload) and 'gt_answers' not in json.dumps(payload)
        with tempfile.TemporaryDirectory() as d:
            x=sample('x');x['gt_answers']='Real';(Path(d)/'x.png').write_bytes(b'image')
            model=Fake({'x':responses()});row=generate(x,model,d)
            self.assertEqual(row['review_status'],'pending')
            with self.assertRaises(ValueError):export([row],d,'A')
            approved=apply_reviews([row],[review(row)])[0]
            self.assertEqual(len(export([approved],d,'A')),7)
            approved['observation_report']['claims']=['tampered after review']
            with self.assertRaises(ValueError):export([approved],d,'A')
            approved=apply_reviews([row],[review(row)])[0]
            (Path(d)/'x.png').write_bytes(b'changed-image')
            with self.assertRaises(ValueError):export([approved],d,'A')
            x['split']='test';calls=len(model.calls)
            with self.assertRaises(ValueError):generate(x,model,d)
            self.assertEqual(calls,len(model.calls))

    def test_qc_gate_never_promotes_machine_approval(self):
        rows=[{'qc':{'structural_pass':True},'review_status':'pending'} for _ in range(500)]
        self.assertFalse(summary(rows)['pilot_ready_for_scale'])
        for row in rows[:100]:
            row['human_review']={'checks':dict.fromkeys(CHECKS,True)}
            row['review_status']='approved'
        self.assertTrue(summary(rows)['pilot_ready_for_scale'])
        for row in rows[:6]:row['human_review']['checks']['no_fabricated_facts']=False
        self.assertFalse(summary(rows)['pilot_ready_for_scale'])

    def test_loss_mask_and_language_only_lora(self):
        self.assertEqual(assistant_labels([1,2,3,4],[1,2]),[-100,-100,3,4])
        with self.assertRaises(ValueError):assistant_labels([1,2,3],[9])
        with self.assertRaises(ValueError):assistant_labels([1,2],[1,2])
        names=['model.visual.blocks.0.q_proj','model.language_model.layers.0.self_attn.q_proj','lm_head']
        self.assertEqual(language_targets(names),[names[1]])
        with self.assertRaises(ValueError):language_targets([names[0]])

    def test_end_to_end_offline_cli_A_B_and_no_training_imports(self):
        root=Path(__file__).resolve().parents[1]
        def call(*args,ok=True):
            result=subprocess.run([sys.executable,'-B','-m',*args],cwd=root,text=True,capture_output=True)
            if ok:self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            else:self.assertNotEqual(result.returncode,0)
            return result
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            def dump(path,rows):path.write_text('\n'.join(json.dumps(x) for x in rows),encoding='utf-8')
            records=[sample('x'),sample('y','dev')]
            for x in records:(d/x['image_path']).write_bytes(b'fixture-not-used-for-tokenization')
            inp=d/'input.jsonl';dump(inp,records)
            replay=d/'responses.json';replay.write_text(json.dumps({x['sample_id']:responses() for x in records}))
            raw=d/'raw.jsonl'
            call('sixdim.teacher','generate','--input',str(inp),'--output',str(raw),'--image-root',str(d),'--responses',str(replay))
            generated=[json.loads(l) for l in raw.read_text().splitlines()]
            human=d/'review.jsonl';dump(human,[review(x) for x in generated]);approved=d/'approved.jsonl'
            call('sixdim.teacher','review','--input',str(raw),'--output',str(approved),'--reviews',str(human))
            approved_rows=[json.loads(l) for l in approved.read_text().splitlines()]
            a_train=d/'A.train.jsonl';a_dev=d/'A.dev.jsonl'
            for split,path in [('train',a_train),('dev',a_dev)]:
                source=d/(split+'.jsonl');dump(source,[x for x in approved_rows if x['split']==split])
                call('sixdim.data','--input',str(source),'--output',str(path),'--image-root',str(d),'--stage','A','--split',split)
            plan=call('sixdim.train','--train',str(a_train),'--dev',str(a_dev),'--stage','A','--model','local-model','--output-dir',str(d/'never-trained'))
            self.assertEqual(json.loads(plan.stdout)['mode'],'dry_run');self.assertFalse((d/'never-trained').exists())
            # Default invocation cannot silently make teacher API calls.
            call('sixdim.teacher','generate','--input',str(inp),'--output',str(d/'forbidden.jsonl'),'--image-root',str(d),ok=False)
            self.assertFalse((d/'forbidden.jsonl').exists())
            drafts=d/'drafts.jsonl'
            call('sixdim.teacher','drafts','--input',str(inp),'--output',str(drafts),'--image-root',str(d),'--responses',str(replay))
            braw=d/'B.raw.jsonl'
            call('sixdim.teacher','generate','--stage','B','--input',str(drafts),'--output',str(braw),'--image-root',str(d),'--responses',str(replay))
            b_rows=[json.loads(l) for l in braw.read_text().splitlines()]
            b_approved=apply_reviews(b_rows,[review(x) for x in b_rows])
            btrain=export([b_approved[0]],d,'B');bdev=export([b_approved[1]],d,'B','dev')
            atrain=[json.loads(l) for l in a_train.read_text().splitlines()]
            mixed,manifest=prepare(btrain,bdev,'B',atrain)
            self.assertEqual(len(mixed),3);self.assertEqual(manifest['replay_rows'],1)
            self.assertEqual(json.loads(btrain[1]['messages'][-1]['content']),report())
            bp=d/'B.train.jsonl';bd=d/'B.dev.jsonl';dump(bp,btrain);dump(bd,bdev)
            adapter=d/'stage-a-adapter';adapter.mkdir();(adapter/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'local-model'}))
            bplan=call('sixdim.train','--train',str(bp),'--dev',str(bd),'--stage','B','--model','local-model','--output-dir',str(d/'B-never-trained'),'--replay',str(a_train),'--init-adapter',str(adapter))
            self.assertEqual(json.loads(bplan.stdout)['replay_rows'],1);self.assertFalse((d/'B-never-trained').exists())
            with self.assertRaises(ValueError):prepare(btrain,bdev,'B',None)
            leaked=copy.deepcopy(bdev);leaked[0]['event_group_id']=btrain[0]['event_group_id']
            with self.assertRaises(ValueError):prepare(btrain,leaked,'B',atrain)
            with self.assertRaises(ValueError):prepare(atrain,[json.loads(l) for l in a_dev.read_text().splitlines()],'A',heldout=[records[0]])

if __name__=='__main__':unittest.main()
