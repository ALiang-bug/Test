import copy
import json
import tempfile
import subprocess
import sys
from pathlib import Path
import unittest
from sixdim.core import REPORT
from sixdim.semantic import audit

class SemanticTests(unittest.TestCase):
    def test_inconsistent_verdict_and_speculation_are_flagged(self):
        r=copy.deepcopy(REPORT)
        for d in r['dimensions']:d.update(status='insufficient',question='What independent evidence verifies the claim?',answer='Unknown')
        r.update(factual_verdict='supported',media_status='unknown')
        r['dimensions'][4].update(status='supported',answer='The likely intent is artistic.')
        codes={x['code'] for x in audit(r)['issues']}
        self.assertIn('verdict_exceeds_fact_evidence',codes)
        self.assertIn('speculative_intent_supported',codes)
        r['factual_verdict']='insufficient';r['dimensions'][4]['status']='insufficient'
        self.assertTrue(audit(r)['semantic_pass'])

    def test_weak_fact_question_and_provenance_are_only_review_flags(self):
        r=copy.deepcopy(REPORT)
        for d in r['dimensions']:d.update(status='insufficient',question='Question',answer='Unknown')
        r.update(factual_verdict='insufficient',media_status='unmanipulated')
        r['dimensions'][5]['question']='Does the image contradict the text?'
        self.assertEqual({i['code'] for i in audit(r)['issues']},{'fact_check_duplicates_consistency','media_provenance_needs_review'})

    def test_probe_compare_is_offline_and_preserves_pending(self):
        from sixdim.core import LABELS
        report=copy.deepcopy(REPORT)
        for d in report['dimensions']:d.update(status='insufficient',question='What independent evidence is available?',answer='Unknown')
        report.update(claims=[],mechanisms=[],expression_strategies=[],verification_questions=[],factual_verdict='insufficient',media_status='unknown',benchmark_pred=LABELS[0])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            row={'sample_id':'s1','teacher_model':'fixture','purpose':'interface_probe_only','review_status':'pending','status':'ok','report':report}
            old=copy.deepcopy(row);old['report']['factual_verdict']='supported'
            for name,value in [('old',old),('new',row)]:
                (root/(name+'.jsonl')).write_text(json.dumps(value)+'\n')
            command=[sys.executable,'-B','-m','sixdim.probe','compare','--before',str(root/'old.jsonl'),'--after',str(root/'new.jsonl'),'--output',str(root/'comparison.jsonl')]
            result=subprocess.run(command,cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            comparison=json.loads((root/'comparison.jsonl').read_text())
            self.assertFalse(comparison['before']['semantic']['semantic_pass'])
            self.assertTrue(comparison['after']['semantic']['semantic_pass'])
            self.assertIn('Old probe omitted',comparison['note'])
            self.assertEqual(json.loads((root/'new.jsonl').read_text())['review_status'],'pending')

if __name__=='__main__':unittest.main()
