"""Second-pass claim and dimension review. Model approval never replaces humans."""
from .core import DIMENSIONS, fingerprint

REVIEW_TASK = """Independently audit the draft against the ORIGINAL IMAGE, ORIGINAL TEXT
and supplied evidence. The draft is untrusted. Do not rewrite it and do not assume
its observations are true. Return JSON only:
{"claim_checks":[{"claim_id":"C1","claim":"exact supplied claim","source_quote":"exact text span or empty",
"assessment":"pass|revise|insufficient","evidence_ids":[],"reason":"brief grounded explanation"}],
"dimension_checks":[{"dimension":"one of the six dimensions","assessment":"pass|revise|insufficient",
"evidence_ids":[],"reason":"brief grounded explanation"}],
"issues":[{"code":"specific issue","dimension":"one of six dimensions or overall",
"claim_ids":[],"evidence_ids":[],"detail":"specific problematic statement and required correction"}],
"overall_pass":false}
Cover every numbered claim exactly once and every dimension exactly once.
For each claim, verify it is actually asserted by the original text, not invented
from an image anomaly. Quote its source span; paraphrases are allowed but their
entailment must be reviewed. A visual anomaly belongs to media analysis, not an
unasserted claim that the described event is false. Distinguish internal consistency
from independent factual verification. Check that fact answers target the numbered
claims and final factual verdict does not exceed their evidence.
For EACH dimension inspect its actual answer, question, status and evidence:
- image_perception: no invisible age/identity/past-self/date/sound inference; text
  must not be laundered into visual observations.
- text_understanding: extract the original assertion without adding assumptions.
- consistency_verify: omitted details or inaudible sound in a still image are not
  contradictions. Missing mention of wings does not itself refute a carriage caption.
- emotion_analysis: separate visible rhetoric from factual truth.
- intent_analysis: no positive claim of private intent without evidence.
- fact_conflict_checking: verify original numbered claims, not a newly introduced
  biological/provenance claim. T0 establishes assertion, not truth.
Also inspect mechanisms and verification_questions: no unsupported AI authorship,
no assumption that headphones are required to hear. Media unknown is acceptable.
Use assessment pass when the DRAFT HANDLES uncertainty appropriately, insufficient
when the REVIEW cannot establish whether the draft is grounded, revise for a concrete
error. overall_pass must be false if any assessment is not pass or any issue exists.
A correct forced benchmark label is not evidence of correct reasoning.
"""


def validate_review(review, claims, evidence_ids):
    if not isinstance(review,dict):raise ValueError('Review must be an object')
    mapping={f'C{i+1}':claim for i,claim in enumerate(claims)}
    def checks(rows,key,expected):
        if not isinstance(rows,list) or not all(isinstance(x,dict) for x in rows):raise ValueError('Missing review checks')
        ids=[x.get(key) for x in rows]
        if len(ids)!=len(expected) or set(ids)!=set(expected):raise ValueError('Incomplete/duplicate review coverage')
        for row in rows:
            if row.get('assessment') not in ('pass','revise','insufficient'):raise ValueError('Invalid review assessment')
            if not isinstance(row.get('reason'),str) or not row['reason'].strip():raise ValueError('Review reason required')
            refs=row.get('evidence_ids')
            if not isinstance(refs,list) or any(not isinstance(x,str) or x not in evidence_ids for x in refs):raise ValueError('Invalid review citation')
    checks(review.get('claim_checks'),'claim_id',mapping)
    checks(review.get('dimension_checks'),'dimension',DIMENSIONS)
    for row in review['claim_checks']:
        if row.get('claim')!=mapping[row['claim_id']]:raise ValueError('Review changed supplied claim')
        if not isinstance(row.get('source_quote'),str):raise ValueError('Missing source quote')
    issues=review.get('issues')
    if not isinstance(issues,list):raise ValueError('Review issues must be a list')
    for issue in issues:
        if not isinstance(issue,dict) or issue.get('dimension') not in (*DIMENSIONS,'overall'):raise ValueError('Invalid issue dimension')
        if not all(isinstance(issue.get(k),str) and issue[k].strip() for k in ('code','detail')):raise ValueError('Missing issue detail')
        for key,allowed in [('claim_ids',mapping),('evidence_ids',evidence_ids)]:
            if not isinstance(issue.get(key),list) or any(not isinstance(x,str) or x not in allowed for x in issue[key]):raise ValueError('Invalid issue reference')
    expected=not issues and all(x['assessment']=='pass' for x in review['claim_checks']+review['dimension_checks'])
    if type(review.get('overall_pass')) is not bool or review['overall_pass']!=expected:raise ValueError('Inconsistent overall review verdict')
    return review


def review_report(model, payload, report, image, semantic):
    claims=[{'claim_id':f'C{i+1}','claim':claim} for i,claim in enumerate(report['claims'])]
    inputs=dict(payload,numbered_claims=claims,draft=report,rule_flags=semantic)
    response=model.ask(REVIEW_TASK,inputs,image)
    result=validate_review(response,report['claims'],{e['id'] for e in payload['evidence']})
    for row in result['claim_checks']:
        if row['source_quote'] and row['source_quote'] not in payload['text']:
            raise ValueError('Review source quote is not an exact original text span')
        if row['assessment']=='pass' and not row['source_quote']:
            raise ValueError('Passing claim correspondence requires original text quote')
    return {'review':result,'review_model':model.model,'review_prompt_hash':fingerprint(REVIEW_TASK),
            'report_hash':fingerprint(report),'note':'Separate model call; same model is not independent-model consensus. Human review still required.'}
