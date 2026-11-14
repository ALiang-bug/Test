"""Conservative event/duplicate grouping before QA generation; never modifies inputs."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from .core import fingerprint
from .io import load,write,unique,image_path


def normalized(text):
    return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def partition(train,test,event_map=None,image_root=None,dev_fraction=.1,seed=42):
    if not 0<dev_fraction<1: raise ValueError('dev fraction must be between zero and one')
    rows=[]
    for origin,items in [('pool',train),('test',test)]:
        for x in items:
            x=dict(x)
            x.setdefault('sample_id',x.get('image_path',''))
            if not isinstance(x.get('text'),str) or not normalized(x['text']): raise ValueError('Missing news text')
            x['_origin']=origin; rows.append(x)
    unique(rows)
    mapping={}
    if event_map is not None:
        unique(event_map)
        mapping={x['sample_id']:x for x in event_map}
        if set(mapping)!={x['sample_id'] for x in rows}: raise ValueError('Event map must cover every input exactly')
        if any(not x.get('event_id') or x.get('review_status')!='approved' for x in event_map):
            raise ValueError('Event map requires approved event_id for every sample')
    parent=list(range(len(rows)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[find(j)]=find(i)
    keys={}; links=[]; hashes={}
    for i,x in enumerate(rows):
        identifiers=[('text',normalized(x['text']))]
        # A reviewed event map is required to certify semantic event grouping.
        if mapping: identifiers.append(('event',mapping[x['sample_id']]['event_id']))
        if image_root:
            path=image_path(image_root,x['image_path'])
            digest=hashlib.sha256(path.read_bytes()).hexdigest(); hashes[x['sample_id']]=digest
            identifiers.append(('image',digest))
        for kind,key in identifiers:
            if (kind,key) in keys:
                other=keys[(kind,key)]; union(i,other)
                links.append({'a':rows[other]['sample_id'],'b':x['sample_id'],'reason':kind})
            else: keys[(kind,key)]=i
    groups=defaultdict(list)
    for i,x in enumerate(rows): groups[find(i)].append(x)
    output={'train':[],'dev':[],'test':[],'excluded':[]}; pool=[]
    for members in groups.values():
        gid='event-'+fingerprint(sorted(x['sample_id'] for x in members))[:20]
        has_test=any(x['_origin']=='test' for x in members)
        for x in members:
            x['event_group_id']=gid
            x['grouping_status']='reviewed' if mapping else 'provisional'
            if x['sample_id'] in hashes: x['image_sha256']=hashes[x['sample_id']]
        if has_test:
            for x in members:
                split='test' if x['_origin']=='test' else 'excluded'
                x['split']=split
                if split=='excluded': x['exclusion_reason']='overlap_with_test'
                output[split].append(x)
        else: pool.append((gid,members))
    # Balance groups by their majority class; groups, not rows, are assigned.
    strata=defaultdict(list)
    for gid,members in sorted(pool):
        majority=Counter(x.get('fake_cls','unknown') for x in members).most_common(1)[0][0]
        strata[majority].append((gid,members))
    rng=random.Random(seed)
    for entries in strata.values():
        rng.shuffle(entries)
        n=round(len(entries)*dev_fraction)
        if len(entries)>1: n=max(1,min(n,len(entries)-1))
        for index,(_,members) in enumerate(entries):
            split='dev' if index<n else 'train'
            for x in members: x['split']=split; output[split].append(x)
    for items in output.values():
        for x in items: x.pop('_origin')
        items.sort(key=lambda x:x['sample_id'])
    audit={'seed':seed,'dev_fraction':dev_fraction,'source_fingerprint':fingerprint({'train':train,'test':test}),
           'event_map_fingerprint':fingerprint(event_map) if event_map is not None else None,
           'event_grouping_reviewed':bool(mapping),'image_hashes_checked':bool(image_root),
           'near_duplicate_semantic_audit':'Requires human event map; normalized exact text and optional byte hashes only',
           'counts':{s:len(v) for s,v in output.items()},'links':links}
    return output,audit


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train-input',required=True);p.add_argument('--test-input',required=True)
    p.add_argument('--event-map');p.add_argument('--image-root');p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--dev-fraction',type=float,default=.1)
    a=p.parse_args(); dest=Path(a.output_dir)
    if dest.exists(): raise FileExistsError(dest)
    outputs,audit=partition(load(a.train_input),load(a.test_input),load(a.event_map) if a.event_map else None,a.image_root,a.dev_fraction,a.seed)
    dest.mkdir(parents=True)
    for split,items in outputs.items(): write(dest/(split+'.jsonl'),items)
    template=[{'sample_id':x['sample_id'],'event_id':x['event_group_id'],'review_status':'pending'} for items in outputs.values() for x in items]
    write(dest/'event_map.template.jsonl',template)
    (dest/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit['counts']))

if __name__=='__main__':main()
