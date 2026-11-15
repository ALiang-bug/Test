"""Second-pass claim and dimension review. Model approval never replaces humans."""
from .core import DIMENSIONS, fingerprint

REVIEW_TASK = """Independently audit the draft against the ORIGINAL IMAGE, ORIGINAL TEXT
and supplied evidence. The draft is untrusted. Do not rewrite it and do not assume
its observations are true. Return JSON only:
{"source_checks":[{"claim_id":"C1","claim":"exact supplied claim","source_quote":"exact text span or empty",
"assessment":"pass|revise|insufficient","evidence_ids":[],"reason":"brief grounded explanation"}],
"fact_checks":[{"claim_id":"C1","claim":"exact supplied claim","verdict":"supported|refuted|insufficient",
"assessment":"pass|revise|insufficient","evidence_ids":[],"reason":"evaluate how the draft handles factual evidence"}],
"dimension_checks":[{"dimension":"one of the six dimensions","assessment":"pass|revise|insufficient",
"evidence_ids":[],"reason":"brief grounded explanation"}],
"issues":[{"code":"specific issue","dimension":"one of six dimensions or overall",
"claim_ids":[],"evidence_ids":[],"draft_path":"dimensions.image_perception.answer","draft_quote":"exact nonempty draft substring","detail":"specific problematic statement and required correction"}],
"overall_pass":false}
Cover every numbered claim exactly once and every dimension exactly once.
SOURCE CHECKS judge ONLY textual entailment/correspondence. A claim explicitly
asserted in T0 can pass source correspondence even if its truth is unknown.
FACT CHECKS separately return the real-world evidence verdict, plus an assessment
of whether the DRAFT handled that evidence correctly. A correctly uncertain draft
can have verdict insufficient AND assessment pass. Never fail source correspondence
merely because sound/identity/history cannot be verified from an image.
For every issue quote an exact nonempty problematic substring from the draft and
provide its path: dimensions.<dimension>.answer or .question, claims.<zero-based-index>,
mechanisms.<index>, expression_strategies.<index>, verification_questions.<index>,
factual_verdict or media_status. No invented quotes or paraphrased quotations.
Point to the actual mistake: do not say a draft avoids age inference while its
answer says younger version of herself. Colors and image style can be observable;
subjective framing is not automatically an error. Cite why a detail is unsupported.
Every revise assessment needs a matching quoted issue. Do not fabricate issues to
justify an uncertain review. Recheck AI-generation claims in mechanisms.
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


def validate_review(review, claims, evidence_ids, report=None):
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
    checks(review.get('source_checks'),'claim_id',mapping)
    checks(review.get('fact_checks'),'claim_id',mapping)
    checks(review.get('dimension_checks'),'dimension',DIMENSIONS)
    for row in review['source_checks']:
        if row.get('claim')!=mapping[row['claim_id']]:raise ValueError('Review changed supplied claim')
        if not isinstance(row.get('source_quote'),str):raise ValueError('Missing source quote')
    for row in review['fact_checks']:
        if row.get('claim')!=mapping[row['claim_id']]:raise ValueError('Fact check changed supplied claim')
        if row.get('verdict') not in ('supported','refuted','insufficient'):raise ValueError('Invalid factual evidence verdict')
    issues=review.get('issues')
    if not isinstance(issues,list):raise ValueError('Review issues must be a list')
    for issue in issues:
        if not isinstance(issue,dict) or issue.get('dimension') not in (*DIMENSIONS,'overall'):raise ValueError('Invalid issue dimension')
        if not all(isinstance(issue.get(k),str) and issue[k].strip() for k in ('code','detail','draft_path','draft_quote')):raise ValueError('Missing issue detail')
        for key,allowed in [('claim_ids',mapping),('evidence_ids',evidence_ids)]:
            if not isinstance(issue.get(key),list) or any(not isinstance(x,str) or x not in allowed for x in issue[key]):raise ValueError('Invalid issue reference')
        if report is not None:
            value=draft_value(report,issue['draft_path'])
            if not isinstance(value,str) or issue['draft_quote'] not in value:
                raise ValueError('Issue quote is not an exact substring at the supplied draft path')
    for row in review['dimension_checks']:
        if row['assessment']=='revise' and not any(i['dimension']==row['dimension'] for i in issues):
            raise ValueError('Dimension revision requires a quoted matching issue')
    for row in review['source_checks']+review['fact_checks']:
        if row['assessment']=='revise' and not any(row['claim_id'] in i['claim_ids'] for i in issues):
            raise ValueError('Claim revision requires a quoted matching issue')
    expected=not issues and all(x['assessment']=='pass' for x in review['source_checks']+review['fact_checks']+review['dimension_checks'])
    if type(review.get('overall_pass')) is not bool or review['overall_pass']!=expected:raise ValueError('Inconsistent overall review verdict')
    return review


def review_report(model, payload, report, image, semantic):
    claims=[{'claim_id':f'C{i+1}','claim':claim} for i,claim in enumerate(report['claims'])]
    inputs=dict(payload,numbered_claims=claims,draft=report,rule_flags=semantic)
    response=model.ask(REVIEW_TASK,inputs,image)
    result=validate_review(response,report['claims'],{e['id'] for e in payload['evidence']},report)
    for row in result['source_checks']:
        if row['source_quote'] and row['source_quote'] not in payload['text']:
            raise ValueError('Review source quote is not an exact original text span')
        if row['assessment']=='pass' and not row['source_quote']:
            raise ValueError('Passing claim correspondence requires original text quote')
    return {'review':result,'review_model':model.model,'review_prompt_hash':fingerprint(REVIEW_TASK),
            'report_hash':fingerprint(report),'note':'Separate model call; same model is not independent-model consensus. Human review still required.'}


def draft_value(report,path):
    parts=path.split('.')
    if parts[0]=='dimensions' and len(parts)==3 and parts[1] in DIMENSIONS and parts[2] in ('answer','question'):
        return next(d[parts[2]] for d in report['dimensions'] if d['dimension']==parts[1])
    if parts[0] in ('claims','mechanisms','expression_strategies','verification_questions') and len(parts)==2 and parts[1].isdigit():
        try:return report[parts[0]][int(parts[1])]
        except (IndexError,KeyError):raise ValueError('Invalid draft list reference')
    if path in ('factual_verdict','media_status'):return report[path]
    raise ValueError('Invalid draft issue path')
