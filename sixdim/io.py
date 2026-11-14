"""Small, strict JSON utilities shared by offline preparation commands."""
import json
from pathlib import Path


def load(path):
    text=Path(path).read_text(encoding='utf-8-sig')
    rows=json.loads(text) if text.lstrip().startswith('[') else [json.loads(x) for x in text.splitlines() if x.strip()]
    if not isinstance(rows,list) or not all(isinstance(x,dict) for x in rows):
        raise ValueError('Expected JSON array or JSONL objects')
    return rows


def write(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def unique(rows,key='sample_id'):
    ids=[x.get(key) for x in rows]
    if any(not isinstance(x,str) or not x.strip() for x in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Missing or duplicate '+key)


def image_path(root,relative):
    base=Path(root).resolve(); path=(base/relative).resolve()
    if not path.is_relative_to(base): raise ValueError('Image path escapes image root')
    if not path.is_file(): raise FileNotFoundError(path)
    return path
