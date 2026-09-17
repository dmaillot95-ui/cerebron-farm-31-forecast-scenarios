#!/usr/bin/env python3
import os, json, pathlib, subprocess, hashlib
PREFERRED=['/generate','/chat','/predict','/respond','/infer','/run']
def run(cmd, timeout=240): return subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
def payload_for(spec,prompt):
    payload={}; prompt_set=False
    for p in spec.get('parameters',[]):
        name=p.get('name',''); lname=name.lower(); required=bool(p.get('required',False)); default=p.get('default'); typ=(p.get('type') or {}).get('type')
        if lname in {'message','prompt','text','query','input','instruction','user_message'}: payload[name]=prompt; prompt_set=True
        elif lname in {'chat_history','history','messages'}: payload[name]=[]
        elif lname in {'max_new_tokens','max_tokens','maximum_new_tokens'}: payload[name]=900
        elif lname=='temperature': payload[name]=0.1
        elif lname=='top_p': payload[name]=0.9
        elif lname=='top_k': payload[name]=40
        elif lname in {'system','system_prompt'}: payload[name]='Evidence-grounded forecasting and scenario analysis. Forecast is not fact.'
        elif required and default is None:
            if typ=='string' and not prompt_set: payload[name]=prompt; prompt_set=True
            else: return None
    return payload if prompt_set else None
def extract(raw):
    raw=raw.strip()
    try:
        obj=json.loads(raw)
        if isinstance(obj,dict):
            for k in ('Response','response','text','output','message'):
                if isinstance(obj.get(k),str): return obj[k].strip()
    except Exception: pass
    return raw
def invoke(space,prompt):
    info=run(['hf-gradio','info',space],120)
    if info.returncode!=0: return False,'',{'stage':'info','error':(info.stderr or info.stdout)[-1200:]}
    try: api=json.loads(info.stdout)
    except Exception as e: return False,'',{'stage':'decode','error':repr(e)}
    endpoints=list(api.items()); endpoints.sort(key=lambda kv:(PREFERRED.index(kv[0]) if kv[0] in PREFERRED else 99,kv[0]))
    errors=[]
    for endpoint,spec in endpoints:
        p=payload_for(spec,prompt)
        if p is None: continue
        pred=run(['hf-gradio','predict',space,endpoint,json.dumps(p,ensure_ascii=False)],240)
        if pred.returncode==0 and (pred.stdout or '').strip():
            text=extract(pred.stdout)
            if text: return True,text,{'stage':'predict','endpoint':endpoint,'sha256':hashlib.sha256(text.encode()).hexdigest()}
        errors.append((pred.stderr or pred.stdout)[-700:])
    return False,'',{'stage':'predict','error':' | '.join(errors[-3:]) or 'No compatible endpoint'}
role=os.environ.get('ROLE','UNKNOWN_ROLE'); model=os.environ.get('MODEL','huggingface-projects/llama-3.2-3B-Instruct'); mission=pathlib.Path('MISSION.md').read_text(encoding='utf-8')
prompt=f'''You are role {role} in CEREBRON Farm 31 Forecast Scenarios.\n{mission}\nReturn an auditable analysis with explicit assumptions, time horizon, uncertainty, base rates, failure modes, evidence status, calibration needs and what would falsify the forecast or scenario. Do not present scenarios as facts or model outputs as observations.'''
ok,text,meta=invoke(model,prompt)
out={'role':role,'model':model,'inference_success':bool(ok),'status':'UNREVIEWED_EXTERNAL_AGENT_OUTPUT' if ok else 'EXTERNAL_INFERENCE_FAILED','result':text if ok else None,'error':None if ok else meta.get('error'),'meta':meta}
pathlib.Path('results').mkdir(exist_ok=True); pathlib.Path(f'results/{role}.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'role':role,'inference_success':bool(ok),'model':model}))
