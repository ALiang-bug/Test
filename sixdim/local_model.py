"""Local BF16 base/LoRA inference, imported only on explicit benchmark execution."""
import json
import hashlib
import importlib.metadata
from pathlib import Path
from .core import POLICY


class LocalModel:
    def __init__(self, path, adapter=None, min_pixels=12544, max_pixels=262144):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Local benchmark requires a BF16-capable CUDA GPU')
        self.model=str(path)
        self.calls=[]
        def digest(file):
            h=hashlib.sha256()
            with Path(file).open('rb') as stream:
                for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
            return h.hexdigest()
        self.provenance={'base_path':str(Path(path).resolve()),
                         'base_config_sha256':digest(Path(path)/'config.json'),
                         'adapter_path':str(Path(adapter).resolve()) if adapter else None}
        if adapter:
            weights=Path(adapter)/'adapter_model.safetensors'
            if not weights.is_file():raise ValueError('Expected safetensors LoRA adapter')
            self.provenance['adapter_sha256']=digest(weights)
        self.processor=AutoProcessor.from_pretrained(path,local_files_only=True,
            min_pixels=min_pixels,max_pixels=max_pixels)
        if adapter:
            config=json.loads((Path(adapter)/'adapter_config.json').read_text(encoding='utf-8'))
            base=config.get('base_model_name_or_path')
            if base!=str(path) and Path(str(base)).resolve()!=Path(path).resolve():
                raise ValueError('Adapter base model does not match benchmark base')
        self.network=Qwen3VLForConditionalGeneration.from_pretrained(path,local_files_only=True,
            torch_dtype=torch.bfloat16).to('cuda')
        if adapter:
            from peft import PeftModel
            self.network=PeftModel.from_pretrained(self.network,adapter,local_files_only=True)
        self.network.eval()
        self.protocol={'backend':'transformers','dtype':'bf16','min_pixels':min_pixels,'max_pixels':max_pixels,
                       'do_sample':False,'max_input_tokens':8192,
                       'versions':{x:importlib.metadata.version(x) for x in ('torch','transformers','peft')}}

    def ask(self, task, payload, image=None):
        import torch
        from PIL import Image
        if not image:raise ValueError('Image required')
        messages=[{'role':'system','content':POLICY},{'role':'user','content':[
            {'type':'image'},{'type':'text','text':task+'\nINPUT:\n'+json.dumps(payload,ensure_ascii=False)}]}]
        with Image.open(image) as raw:rgb=raw.convert('RGB')
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=[rgb],return_tensors='pt').to('cuda')
        length=inputs['input_ids'].shape[1]
        if length>8192:raise ValueError('Overlength benchmark input; no silent truncation')
        with torch.inference_mode():
            output=self.network.generate(**inputs,do_sample=False,max_new_tokens=4096)
        generated=output[0,length:]
        self.calls.append({'model':self.model,'prompt_tokens':length,'completion_tokens':len(generated)})
        if len(generated)==4096:raise ValueError('Generation reached token limit; report may be truncated')
        return json.loads(self.processor.decode(generated,skip_special_tokens=True))
