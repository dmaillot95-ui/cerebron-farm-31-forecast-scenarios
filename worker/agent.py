import json, os, pathlib
from gradio_client import Client

role = os.environ.get('ROLE','UNKNOWN_ROLE')
model = os.environ.get('MODEL','huggingface-projects/llama-3.2-3B-Instruct')
mission = pathlib.Path('MISSION.md').read_text(encoding='utf-8')
prompt = f'''You are role {role} in CÉRÉBRON Farm 31 Forecast Scenarios.\nFollow the mission exactly.\n{mission}\nReturn an auditable analysis with explicit assumptions, horizon, uncertainty, failure modes, evidence status and what would falsify the forecast/scenario. Do not present scenarios as facts.'''

out = {'role': role, 'model': model, 'success': False}
try:
    c = Client(model)
    result = c.predict(prompt, api_name='/chat')
    out['success'] = True
    out['result'] = result
except Exception as e:
    out['error'] = repr(e)

pathlib.Path('results').mkdir(exist_ok=True)
pathlib.Path(f'results/{role}.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
