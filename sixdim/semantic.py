"""Conservative semantic triage; heuristics are not factual verification."""
import re


def audit(report):
    issues=[]
    dims={d['dimension']:d for d in report['dimensions']}
    fact=dims['fact_conflict_checking']
    if fact['status']=='insufficient' and report['factual_verdict']!='insufficient':
        issues.append({'code':'verdict_exceeds_fact_evidence','severity':'error',
                       'detail':'Fact checking is insufficient but final factual verdict is definite.'})
    intent=dims['intent_analysis']
    if intent['status']=='supported' and re.search(r'\b(likely|possibly|appears|suggests?|may|might|perhaps)\b',intent['answer'],re.I):
        issues.append({'code':'speculative_intent_supported','severity':'warning',
                       'detail':'Speculative wording with supported intent requires human evidence review.'})
    if re.search(r'contradict|conflict|consistent|align',fact['question'],re.I) and not re.search(r'external|independent|source|real.world|verif|evidence',fact['question'],re.I):
        issues.append({'code':'fact_check_duplicates_consistency','severity':'warning',
                       'detail':'Fact question may only repeat cross-modal consistency.'})
    if report['media_status']=='unmanipulated':
        issues.append({'code':'media_provenance_needs_review','severity':'warning',
                       'detail':'Require positive provenance; absence of visible artifacts is insufficient.'})
    return {'semantic_pass':not issues,'issues':issues,
            'note':'Flags require independent/model/human review. No flags does not establish correctness.'}
