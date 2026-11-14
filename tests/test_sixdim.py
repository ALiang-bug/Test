import copy
import json
import tempfile
import unittest
from pathlib import Path
from sixdim.core import Analyzer, Memory, REPORT, DIMENSIONS, validate
from sixdim.__main__ import evaluate
from sixdim.data import export
from sixdim.teacher import apply_reviews,annotation_hash,CHECKS


def report(verdict='insufficient', label='original'):
    r=copy.deepcopy(REPORT)
    r['claims']=['A claim']; r['factual_verdict']=verdict; r['media_status']='unknown'
    r['benchmark_pred']=label; r['mechanisms']=[]; r['expression_strategies']=[]; r['verification_questions']=[]
    for d in r['dimensions']: d['status']='insufficient'
    return r


class Tests(unittest.TestCase):
    def test_schema_rejects_unknown_citation_and_duplicate_dimension(self):
        r=report(); r['dimensions'][0]['evidence_ids']=['invented']
        with self.assertRaises(ValueError): validate(r,set())
        r=report(); r['dimensions'][0]['dimension']=DIMENSIONS[1]
        with self.assertRaises(ValueError): validate(r,set())

    def test_pipeline_blinds_ground_truth_and_preserves_correct_draft(self):
        class Fake:
            def ask(self,task,payload,image=None):
                assert 'fake_cls' not in json.dumps(payload)
                if 'domain tags' in task: return {'domains':['Society'],'persona':'Fact checker'}
                if 'Audit draft' in task: return {'issues':[],'changes':[],'unresolved':[]}
                return report()
        result=Analyzer(Fake()).analyze({'sample_id':'1','text':'claim','fake_cls':'mismatch'},'unused')
        self.assertEqual(result['draft'],result['final'])
        self.assertEqual(len(result['history']),1)

    def test_memory_excludes_same_event_and_redacts_verdict(self):
        class Embed:
            def rank(self,q,docs): return list(range(len(docs)))
        cases=[{'sample_id':str(i),'event_group_id':str(i),'summary':'similar event','verification_steps':['check date'],
                'sources':['https://example.org/evidence'],'split':'train','review_status':'approved','label':'fake'} for i in range(2)]
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'memory.jsonl'; p.write_text('\n'.join(json.dumps(c) for c in cases))
            m=Memory(p,Embed()); original=m.version
            result=m.search('query',{'sample_id':'0','event_group_id':'0'})
            with self.assertRaises(ValueError): m.search('query',{'sample_id':'new'})
            self.assertEqual(len(result),1); self.assertNotIn('label',result[0]); self.assertEqual(original,m.version)
            cases[0]['split']='test'; p.write_text(json.dumps(cases[0]))
            with self.assertRaises(ValueError): Memory(p,Embed())

    def test_failed_samples_remain_in_accuracy_denominator(self):
        records=[{'sample_id':'1','fake_cls':'original','status':'ok','final':report()},
                 {'sample_id':'2','fake_cls':'mismatch','status':'failed'}]
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'eval.json'; evaluate(records,path)
            r=json.loads(path.read_text())
            self.assertEqual(r['four_class']['accuracy_all_inputs'],.5)
            self.assertEqual(r['failed'],1); self.assertEqual(r['factual_insufficient_count'],1)
            with self.assertRaises(FileExistsError): evaluate(records,path)

    def test_export_gates_review_split_and_exposes_cited_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'x.jpg').write_bytes(b'test')
            r=report(); r['dimensions'][0]['evidence_ids']=['E1']
            record={'sample_id':'x','event_group_id':'g','split':'train','review_status':'approved','image_path':'x.jpg',
                    'image_sha256':__import__('hashlib').sha256(b'test').hexdigest(),
                    'text':'claim','report':r,'observation_report':report(),'stage':'A','grouping_status':'reviewed',
                    'evidence':[{'id':'E1','excerpt':'fact'}]}
            record=apply_reviews([record],[{'sample_id':'x','annotation_hash':annotation_hash(record),'decision':'approved',
                    'reviewer':'test-human','reviewed_at':'2026-11-14','checks':dict.fromkeys(CHECKS,True)}])[0]
            rows=export([record],d,'A'); self.assertEqual(len(rows),7)
            self.assertIn('E1',rows[0]['messages'][1]['content'])
            record['split']='test'
            with self.assertRaises(ValueError): export([record],d,'A')

if __name__=='__main__': unittest.main()
