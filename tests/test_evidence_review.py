import copy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from sixdim.core import REPORT
from sixdim import probe
from sixdim.core import DIMENSIONS
from sixdim.evidence_review import review_report,validate_review


def review():
    return {'claim_checks':[{'claim_id':'C1','claim':'A person rides.','source_quote':'A person rides.',
            'assessment':'pass','evidence_ids':['T0'],'reason':'Explicit assertion, not inferred truth.'}],
            'dimension_checks':[{'dimension':d,'assessment':'pass','evidence_ids':['I0','T0'],
                                 'reason':'Draft preserves uncertainty.'} for d in DIMENSIONS],
            'issues':[],'overall_pass':True}


class EvidenceReviewTests(unittest.TestCase):
    def test_full_coverage_and_consistent_verdict(self):
        self.assertTrue(validate_review(review(),['A person rides.'],{'I0','T0'})['overall_pass'])
        for mutate in ('missing_dimension','duplicate_claim','wrong_claim','unknown_ref','false_pass'):
            value=review()
            if mutate=='missing_dimension':value['dimension_checks'].pop()
            elif mutate=='duplicate_claim':value['claim_checks']*=2
            elif mutate=='wrong_claim':value['claim_checks'][0]['claim']='Invented claim'
            elif mutate=='unknown_ref':value['dimension_checks'][0]['evidence_ids']=['invented']
            else:value['dimension_checks'][0]['assessment']='revise'
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):
                validate_review(value,['A person rides.'],{'I0','T0'})

    def test_review_receives_image_and_blind_original_claims(self):
        class Fake:
            model='fixture'
            def ask(self,task,payload,image):
                assert image=='original.png'
                assert set(payload)=={'sample_id','text','evidence','numbered_claims','draft','rule_flags'}
                assert payload['numbered_claims']==[{'claim_id':'C1','claim':'A person rides.'}]
                assert 'fake_cls' not in payload
                return review()
        result=review_report(Fake(),{'sample_id':'s','text':'A person rides.',
            'evidence':[{'id':'I0'},{'id':'T0'}]}, {'claims':['A person rides.']},'original.png',{})
        self.assertTrue(result['review']['overall_pass'])
        self.assertIn('report_hash',result)

    def test_nonpassing_review_is_retained_not_auto_rewrite(self):
        class Fake:
            model='fixture'
            def ask(self,*args):
                r=review();r['dimension_checks'][0]['assessment']='revise';r['overall_pass']=False
                r['issues']=[{'code':'text_induced_identity','dimension':'image_perception',
                    'claim_ids':['C1'],'evidence_ids':['I0'],'detail':'Identity cannot be established.'}]
                return r
        original={'claims':['A person rides.']};before=copy.deepcopy(original)
        result=review_report(Fake(),{'text':'A person rides.','evidence':[{'id':'I0'},{'id':'T0'}]},original,'image',{})
        self.assertFalse(result['review']['overall_pass']);self.assertEqual(original,before)

    def test_invented_quote_and_missing_source_quote_fail(self):
        class Fake:
            model='fixture'
            def __init__(self,quote):self.quote=quote
            def ask(self,*args):
                r=review();r['claim_checks'][0]['source_quote']=self.quote;return r
        for quote in ('invented source',''):
            with self.subTest(quote=quote),self.assertRaises(ValueError):
                review_report(Fake(quote),{'text':'A person rides.','evidence':[{'id':'I0'},{'id':'T0'}]},
                              {'claims':['A person rides.']},'image',{})

    def test_probe_calls_twice_and_keeps_rejected_report_pending(self):
        report=copy.deepcopy(REPORT)
        report.update(claims=['A person rides.'],mechanisms=[],expression_strategies=[],verification_questions=[],
                      factual_verdict='insufficient',media_status='unknown',benchmark_pred='original')
        for d in report['dimensions']:d.update(status='insufficient',evidence_ids=['I0','T0'],question='What independent evidence verifies this?',answer='Unknown')
        calls=[]
        class Fake:
            model='fixture'
            def __init__(self,*args):pass
            def ask(self,task,payload,image):
                calls.append(payload)
                if len(calls)==1:return copy.deepcopy(report)
                value=review();value['dimension_checks'][0]['assessment']='revise';value['overall_pass']=False
                return value
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'image.png').write_bytes(b'fixture')
            row={'sample_id':'s','split':'train','grouping_status':'provisional','image_path':'image.png',
                 'text':'A person rides.','fake_cls':'original'}
            (root/'candidates.jsonl').write_text(json.dumps(row)+'\n')
            (root/'before.jsonl').write_text(json.dumps({'sample_id':'s'})+'\n')
            args=['probe','run','--candidates',str(root/'candidates.jsonl'),'--before',str(root/'before.jsonl'),
                  '--image-root',str(root),'--output',str(root/'out.jsonl'),'--allow-network','--model','fixture']
            with patch('sys.argv',args),patch('sixdim.probe.Model',Fake),patch('builtins.print'):
                probe.main()
            result=json.loads((root/'out.jsonl').read_text())
            self.assertEqual(len(calls),2)
            self.assertTrue(all('fake_cls' not in payload for payload in calls))
            self.assertEqual(result['status'],'ok')
            self.assertFalse(result['machine_review_pass'])
            self.assertEqual(result['review_status'],'pending')
            self.assertEqual(result['report'],report)

if __name__=='__main__':unittest.main()
