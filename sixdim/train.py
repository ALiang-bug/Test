"""Offline preflight by default. Training requires --execute-training explicitly."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from .core import fingerprint
from .io import load

TARGETS={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}


def assistant_labels(full,prefix):
    if len(prefix)>=len(full) or full[:len(prefix)]!=prefix:
        raise ValueError('Chat template is not prefix-stable; refuse incorrect loss masking')
    return [-100]*len(prefix)+full[len(prefix):]


def validate_rows(rows,split,stages):
    if not rows:raise ValueError('Empty SFT dataset')
    seen=set()
    for row in rows:
        if row.get('split')!=split or row.get('stage') not in stages or row.get('review_status')!='approved':
            raise ValueError('Unexpected split/stage or missing approval')
        if not row.get('event_group_id') or not row.get('sample_id') or not row.get('annotation_hash'):
            raise ValueError('Missing SFT provenance')
        messages=row.get('messages',[])
        if [m.get('role') for m in messages]!=['system','user','assistant']:
            raise ValueError('Only single-target system/user/assistant examples supported')
        if any(not isinstance(m.get('content'),str) or not m['content'].strip() for m in messages):
            raise ValueError('Empty message')
        if messages[1]['content'].count('<image>')!=1:raise ValueError('Exactly one image placeholder required')
        if len(row.get('images',[]))!=1 or not Path(row['images'][0]).is_file():raise ValueError('Missing single local image')
        import hashlib
        if hashlib.sha256(Path(row['images'][0]).read_bytes()).hexdigest()!=row.get('image_sha256'):
            raise ValueError('SFT image hash mismatch')
        key=fingerprint({'sample_id':row['sample_id'],'messages':messages})
        if key in seen:raise ValueError('Duplicate SFT example')
        seen.add(key)
        json.loads(messages[-1]['content'])


def prepare(train,dev,stage,replay=None,seed=42,heldout=None):
    validate_rows(train,'train',{stage});validate_rows(dev,'dev',{stage})
    training_groups={x['event_group_id'] for x in train}
    development_groups={x['event_group_id'] for x in dev}
    training_ids={x['sample_id'] for x in train}
    development_ids={x['sample_id'] for x in dev}
    if training_groups & development_groups or training_ids & development_ids:raise ValueError('Train/dev event or sample leakage')
    mixed=list(train);n=0
    if stage=='B':
        if not replay:raise ValueError('Stage B requires Stage A replay data (25% target)')
        validate_rows(replay,'train',{'A'})
        if development_groups & {x['event_group_id'] for x in replay} or development_ids & {x['sample_id'] for x in replay}:
            raise ValueError('Replay overlaps development set')
        n=round(len(train)/3)
        if len(replay)<n:raise ValueError('Not enough distinct Stage A rows for 25% replay')
        mixed+=random.Random(seed).sample(replay,n)
    elif replay:raise ValueError('Replay is only used in Stage B')
    if heldout is not None:
        if not heldout:raise ValueError('Empty held-out manifest')
        unique_heldout={x.get('sample_id') for x in heldout}
        if None in unique_heldout or len(unique_heldout)!=len(heldout) or any(x.get('split')!='test' or not x.get('event_group_id') for x in heldout):
            raise ValueError('Held-out manifest requires unique test samples with event IDs')
        hids={x['sample_id'] for x in heldout};hgroups={x['event_group_id'] for x in heldout}
        if any(x['sample_id'] in hids or x['event_group_id'] in hgroups for x in mixed+dev):
            raise ValueError('Training/development overlaps frozen held-out manifest')
    random.Random(seed).shuffle(mixed)
    return mixed,{'train_rows':len(mixed),'dev_rows':len(dev),'replay_rows':n,
                  'actual_replay_fraction':n/len(mixed),'train_fingerprint':fingerprint(mixed),
                  'dev_fingerprint':fingerprint(dev),'stage':stage,'seed':seed,
                  'heldout_audit':bool(heldout),'stages':dict(Counter(x['stage'] for x in mixed))}


class Collator:
    def __init__(self,processor,max_length):self.processor,self.max_length=processor,max_length
    def encode(self,row):
        from PIL import Image
        from copy import deepcopy
        messages=deepcopy(row['messages'])
        messages[1]['content']=[{'type':'image'}, {'type':'text','text':messages[1]['content'].replace('<image>','',1)}]
        with Image.open(row['images'][0]) as raw: image=raw.convert('RGB')
        full_text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
        prefix_text=self.processor.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True)
        full=self.processor(text=[full_text],images=[image],return_tensors='pt',padding=False)
        prefix=self.processor(text=[prefix_text],images=[image],return_tensors='pt',padding=False)
        ids=full['input_ids'][0].tolist();prompt=prefix['input_ids'][0].tolist()
        if len(ids)>self.max_length:raise ValueError('Overlength sample '+row['sample_id']+'; do not truncate image/assistant supervision')
        return full,assistant_labels(ids,prompt)
    def __call__(self,rows):
        import torch
        encoded=[self.encode(row) for row in rows];width=max(len(labels) for _,labels in encoded)
        pad=self.processor.tokenizer.pad_token_id
        if pad is None:raise ValueError('Tokenizer must define pad_token_id')
        ids=[];masks=[];labels=[]
        for full,target in encoded:
            n=width-len(target)
            ids.append(full['input_ids'][0].tolist()+[pad]*n)
            masks.append([1]*len(target)+[0]*n);labels.append(target+[-100]*n)
        batch={'input_ids':torch.tensor(ids),'attention_mask':torch.tensor(masks),'labels':torch.tensor(labels)}
        for key in ('pixel_values','image_grid_thw'):
            if any(key not in full for full,_ in encoded):raise ValueError('Processor omitted '+key)
            batch[key]=torch.cat([full[key] for full,_ in encoded],dim=0)
        return batch


def language_targets(names):
    selected=[n for n in names if '.language_model.' in '.'+n and n.rsplit('.',1)[-1] in TARGETS]
    if not selected or any('visual' in n for n in selected):raise ValueError('No safe language-only Qwen LoRA target modules')
    return selected


def execute(a,train,dev,manifest):
    # Imports and weights are deliberately beyond the explicit execution gate.
    import importlib.metadata
    import torch
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,Trainer,TrainingArguments,set_seed
    from peft import LoraConfig,get_peft_model,PeftModel
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for this training entry point')
    if a.precision=='bf16' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16 unsupported; explicitly select fp16 if appropriate')
    output=Path(a.output_dir)
    if output.exists():raise FileExistsError(output)
    set_seed(a.seed)
    dtype=torch.bfloat16 if a.precision=='bf16' else torch.float16
    processor=AutoProcessor.from_pretrained(a.model,local_files_only=True,min_pixels=a.min_pixels,max_pixels=a.max_pixels)
    collator=Collator(processor,a.max_length)
    # Validate every tokenized example before optimizer/training. No silent dropping.
    for row in train+dev:collator.encode(row)
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.model,local_files_only=True,torch_dtype=dtype)
    model.config.use_cache=False
    if a.init_adapter:
        model=PeftModel.from_pretrained(model,a.init_adapter,is_trainable=True,local_files_only=True)
    else:
        targets=language_targets([name for name,module in model.named_modules() if isinstance(module,torch.nn.Linear)])
        model=get_peft_model(model,LoraConfig(r=a.rank,lora_alpha=a.alpha,lora_dropout=.05,bias='none',
                                            task_type='CAUSAL_LM',target_modules=targets))
    trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    if not trainable or any('lora_' not in n or 'visual' in n or 'merger' in n for n in trainable):
        raise ValueError('Only language LoRA parameters may be trainable')
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    output.mkdir(parents=True)
    manifest.update(model=a.model,init_adapter=a.init_adapter,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    configuration=vars(a),versions={x:importlib.metadata.version(x) for x in ('torch','transformers','peft','accelerate')})
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    args=TrainingArguments(output_dir=str(output),num_train_epochs=a.epochs,max_steps=a.max_steps,learning_rate=a.learning_rate,
        per_device_train_batch_size=a.batch_size,per_device_eval_batch_size=1,gradient_accumulation_steps=a.accumulation,
        bf16=a.precision=='bf16',fp16=a.precision=='fp16',gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},remove_unused_columns=False,
        eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,logging_steps=10,
        report_to='none',seed=a.seed,data_seed=a.seed,dataloader_num_workers=0,
        warmup_ratio=.03,ddp_find_unused_parameters=False)
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=dev,data_collator=collator)
    trainer.train()
    trainer.save_model(str(output/'adapter'));processor.save_pretrained(str(output/'adapter'))
    trainer.save_state()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',required=True);p.add_argument('--dev',required=True)
    p.add_argument('--stage',choices=['A','B'],required=True);p.add_argument('--model',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--init-adapter');p.add_argument('--replay')
    p.add_argument('--heldout');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rank',type=int,default=32);p.add_argument('--alpha',type=int,default=64)
    p.add_argument('--epochs',type=float,default=2);p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--accumulation',type=int,default=16)
    p.add_argument('--max-length',type=int,default=8192);p.add_argument('--min-pixels',type=int,default=12544)
    p.add_argument('--max-pixels',type=int,default=1003520);p.add_argument('--precision',choices=['bf16','fp16'],default='bf16')
    p.add_argument('--max-steps',type=int,default=-1,help='Positive optimizer steps for a smoke run; -1 uses epochs')
    p.add_argument('--execute-training',action='store_true');a=p.parse_args()
    if a.max_steps==0 or a.max_steps < -1:p.error('--max-steps must be -1 or positive')
    if any(v<=0 for v in (a.rank,a.alpha,a.epochs,a.learning_rate,a.batch_size,a.accumulation,a.max_length,a.min_pixels,a.max_pixels)) or a.min_pixels>a.max_pixels:
        p.error('Training dimensions, rates and budgets must be positive and ordered')
    if a.execute_training and not a.heldout:p.error('Actual training requires --heldout frozen test manifest for leakage audit')
    if a.stage=='B' and not a.init_adapter:p.error('Stage B must continue a Stage A adapter')
    if a.stage=='A' and a.init_adapter:p.error('Stage A starts a new adapter')
    if a.init_adapter:
        cfg=Path(a.init_adapter)/'adapter_config.json'
        if not cfg.is_file():p.error('Missing adapter_config.json')
        adapter=json.loads(cfg.read_text(encoding='utf-8'))
        base=adapter.get('base_model_name_or_path')
        if base!=a.model and Path(str(base)).resolve()!=Path(a.model).resolve():p.error('Adapter base model mismatch')
    if Path(a.output_dir).exists():raise FileExistsError(a.output_dir)
    train,manifest=prepare(load(a.train),load(a.dev),a.stage,load(a.replay) if a.replay else None,a.seed,load(a.heldout) if a.heldout else None)
    manifest.update(mode='execute' if a.execute_training else 'dry_run',
                    note='Dry-run checks data, paths, splits and replay only; no model loads/tokenization/forward/backward.')
    print(json.dumps(manifest,indent=2))
    if a.execute_training:execute(a,train,load(a.dev),manifest)

if __name__=='__main__':main()
