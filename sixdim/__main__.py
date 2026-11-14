import argparse
import json
import os
import time
from pathlib import Path
from .core import Analyzer, Model, Embeddings, Memory, Web, LABELS


def main():
    p=argparse.ArgumentParser(description='Independent six-dimensional workflow; outputs are never overwritten')
    p.add_argument('command',choices=['run','evaluate'])
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--image-root')
    p.add_argument('--base-url',default='http://127.0.0.1:19887/v1')
    p.add_argument('--model',default='Qwen3-VL-8B-Instruct')
    p.add_argument('--embedding-model')
    p.add_argument('--embedding-device',default='cpu')
    p.add_argument('--memory')
    p.add_argument('--web',action='store_true')
    p.add_argument('--proxy')
    p.add_argument('--limit',type=int)
    p.add_argument('--rounds',type=int,default=1,choices=[1,2])
    a=p.parse_args()
    source=Path(a.input).read_text(encoding='utf-8')
    samples=json.loads(source) if source.lstrip().startswith('[') else [json.loads(l) for l in source.splitlines() if l.strip()]
    if a.command=='evaluate':
        evaluate(samples,a.output)
        return
    if not a.image_root:
        p.error('--image-root is required')
    if (a.web or a.memory) and not a.embedding_model:
        p.error('--embedding-model is required for retrieval')
    if a.limit is not None:
        if a.limit<1: p.error('--limit must be positive')
        samples=samples[:a.limit]
    model=Model(a.base_url,a.model,os.environ.get('MAR_API_KEY','EMPTY'))
    embed=Embeddings(a.embedding_model,a.embedding_device) if a.embedding_model else None
    memory=Memory(a.memory,embed) if a.memory else None
    analyzer=Analyzer(model,Web(embed,a.proxy) if a.web else None,memory,a.rounds)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    with open(a.output,'x',encoding='utf-8') as out:
        for index,original in enumerate(samples):
            sample=dict(original)
            sample.setdefault('sample_id',sample.get('image_path',str(index)))
            start=time.monotonic(); call_start=len(model.calls)
            record={'sample_id':sample['sample_id'],'fake_cls':sample.get('fake_cls'),'status':'failed'}
            try:
                image=Path(a.image_root)/sample['image_path']
                if not image.is_file(): raise FileNotFoundError(image)
                record.update(analyzer.analyze(sample,str(image)))
                record['status']='ok'
            except Exception as exc:
                record['error']=str(exc)
            record.update(elapsed_seconds=time.monotonic()-start,calls=model.calls[call_start:])
            out.write(json.dumps(record,ensure_ascii=False)+'\n'); out.flush()


def evaluate(records,output):
    if not records: raise ValueError('Empty evaluation input')
    seen=set(); matrix=[[0]*4 for _ in LABELS]; failed=0; abstained=0
    for r in records:
        if r['sample_id'] in seen: raise ValueError('Duplicate sample ID')
        seen.add(r['sample_id'])
        if r.get('fake_cls') not in LABELS: raise ValueError('Unknown/missing ground truth')
        try:
            pred=r['final']['benchmark_pred'] if r.get('status')=='ok' else None
        except (KeyError,TypeError): pred=None
        if pred not in LABELS:
            failed+=1; continue
        matrix[LABELS.index(r['fake_cls'])][LABELS.index(pred)]+=1
        abstained+=r['final'].get('factual_verdict')=='insufficient'
    def metrics(m):
        fs=[]
        for i in range(len(m)):
            tp=m[i][i]; den=sum(m[i])+sum(row[i] for row in m)
            fs.append(2*tp/den if den else 0)
        return {'macro_f1_successful':sum(fs)/len(fs),'accuracy_all_inputs':sum(m[i][i] for i in range(len(m)))/len(records)}
    binary=[[0,0],[0,0]]
    for i,row in enumerate(matrix):
        for j,n in enumerate(row): binary[int(i!=0)][int(j!=0)]+=n
    report={'total':len(records),'failed':failed,'success_rate':1-failed/len(records),
            'factual_insufficient_count':abstained,'confusion_matrix':matrix,'labels':LABELS,
            'four_class':metrics(matrix),'binary':metrics(binary),
            'note':'Macro F1 uses successful predictions only; failures retained in all-input accuracy. Factual abstention is distinct from benchmark prediction.'}
    with open(output,'x',encoding='utf-8') as f: json.dump(report,f,ensure_ascii=False,indent=2)

if __name__=='__main__': main()
