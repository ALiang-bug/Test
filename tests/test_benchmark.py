import copy
import json
from pathlib import Path
import tempfile
import unittest
from sixdim.benchmark import freeze, run, compare, score
from sixdim.core import REPORT, fingerprint, validate
import subprocess
import sys


def report(label):
    r=copy.deepcopy(REPORT)
    r.update(claims=[], mechanisms=[], expression_strategies=[], verification_questions=[],
             factual_verdict='insufficient',media_status='unknown',benchmark_pred=label)
    for d in r['dimensions']:d.update(status='insufficient',evidence_ids=['I0','T0'])
    return r


class BenchmarkTests(unittest.TestCase):
    def test_frozen_pair_no_label_leakage_failures_and_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'image.png').write_bytes(b'fixture')
            rows=[{'sample_id':str(i),'event_group_id':'e'+str(i),'split':'test','grouping_status':'reviewed',
                   'image_path':'image.png','text':'A claim','fake_cls':label}
                  for i,label in enumerate(('original','mismatch'))]
            frozen=freeze(rows,root)
            class Fake:
                model='local-fixture'
                def __init__(self,fail=False):self.fail=fail
                def ask(self,task,payload,image):
                    self_test=set(payload)
                    if self_test!={'sample_id','text','evidence'}:raise AssertionError('Label leak')
                    if self.fail and payload['sample_id']=='1':raise RuntimeError('fixture failure')
                    return report('original' if payload['sample_id']=='0' else 'mismatch')
            before=root/'before.jsonl';after=root/'after.jsonl'
            run(frozen,root,Fake(True),before,'base');run(frozen,root,Fake(),after,'adapter')
            base=[json.loads(x) for x in before.read_text().splitlines()]
            tuned=[json.loads(x) for x in after.read_text().splitlines()]
            result=compare(base,tuned)
            self.assertEqual(result['accuracy_delta'],.5)
            self.assertEqual(result['wrong_to_correct'],1)
            self.assertEqual(result['base']['failures'],1)
            # Failed mismatch contributes a false negative to the macro score.
            self.assertEqual(result['base']['four_class']['macro_f1'],.25)
            with self.assertRaises(ValueError):compare(base,tuned[:1])
            changed=copy.deepcopy(tuned);changed[0]['protocol_hash']='other'
            with self.assertRaises(ValueError):compare(base,changed)
            changed=copy.deepcopy(tuned);changed[0]['text']='tampered'
            with self.assertRaises(ValueError):compare(base,changed)
            (root/'image.png').write_bytes(b'changed')
            with self.assertRaises(ValueError):run(frozen,root,Fake(),root/'never.jsonl','bad')
            self.assertFalse((root/'never.jsonl').exists())

    def test_definite_answers_need_citations_and_execute_gate(self):
        r=report('original');r['dimensions'][0].update(status='supported',evidence_ids=[])
        with self.assertRaises(ValueError):validate(r,{'I0','T0'})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'image.png').write_bytes(b'image')
            rows=freeze([{'sample_id':'1','event_group_id':'e','split':'test','image_path':'image.png',
                          'text':'claim','fake_cls':'original'}],root)
            source=root/'test.jsonl';source.write_text(json.dumps(rows[0])+'\n')
            command=[sys.executable,'-B','-m','sixdim.benchmark','run','--backend','local',
                     '--input',str(source),'--image-root',str(root),'--output',str(root/'never.jsonl'),
                     '--model','missing-local-model','--run-id','test']
            result=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('--execute-local',result.stderr)
            self.assertFalse((root/'never.jsonl').exists())

    def test_invalid_report_is_failure_even_with_correct_label(self):
        bad={'status':'ok','fake_cls':'original','final':{'benchmark_pred':'original'}}
        self.assertEqual(score([bad])['failures'],1)

if __name__=='__main__':unittest.main()
