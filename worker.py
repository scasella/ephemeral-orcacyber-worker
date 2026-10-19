"""Runpod queue worker for the pinned OrcaCyber GGUF and DFlash2 draft."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import requests
import runpod
from huggingface_hub import hf_hub_download

ROOT = Path('/tmp/orcacyber')
ROOT.mkdir(exist_ok=True)
MODEL_CACHE = Path(os.environ.get('ORCA_MODEL_CACHE', '/runpod-volume/huggingface-cache/hub'))
REVISION = 'fe6ba411d0941f61643b6406614875b292982142'
MODEL = 'OrcaCyber-Nano-1.0-AI-Defense-Mix-15.7G.gguf'
DRAFT = 'OrcaCyber-Nano-1.0-AI-Defense-DFlash2-Q8_0.gguf'
process = None


def download(name, expected=None):
    cached = MODEL_CACHE / 'models--orcarouter--OrcaCyber-Nano-1.0' / 'snapshots' / REVISION / name
    if cached.is_file():
        # Select only the immutable revision we tested, never an arbitrary snapshot.
        print(json.dumps({'model_file': name, 'source': 'runpod-cache'}), flush=True)
        return cached
    target = ROOT / name
    if target.exists():
        return target
    if os.environ.get('ORCA_REQUIRE_CACHE') == '1':
        raise RuntimeError('Pinned model file missing from Runpod model cache')
    print(json.dumps({'model_file': name, 'source': 'download-fallback'}), flush=True)
    # Xet downloads chunks concurrently; a single HTTP stream made cold starts slow.
    os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'
    hf_hub_download(repo_id='orcarouter/OrcaCyber-Nano-1.0', filename=name,
                    revision=REVISION, local_dir=ROOT, token=os.environ['HF_TOKEN'])
    if expected:
        digest = hashlib.sha256()
        with target.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            target.unlink()
            raise RuntimeError('Model checksum mismatch')
    return target


def ensure_server(job):
    global process
    if process is not None and process.poll() is None:
        return
    runpod.serverless.progress_update(job, 'Downloading pinned OrcaCyber weights')
    target = download(MODEL, '4214beb4c3ee2323d97d19a5c6a3327f25d969c7db26956e2fc917cf297dc289')
    runpod.serverless.progress_update(job, 'Downloading DFlash2 draft')
    draft = download(DRAFT)
    runpod.serverless.progress_update(job, 'Loading OrcaCyber and DFlash2 on CUDA')
    command = ['/app/llama-server', '-m', str(target), '-md', str(draft),
               '--spec-type', 'draft-dflash', '--spec-draft-n-max', '5',
               '-ngl', '99', '-ngld', '99', '-np', '1', '-c', '16384',
               '--host', '127.0.0.1', '--port', '8080', '--alias', 'orca-cyber',
               '--jinja', '--chat-template-file', os.environ.get('ORCA_CHAT_TEMPLATE', '/tmp/orca-chat-template.jinja'),
               '--chat-template-kwargs', '{"enable_thinking":false}', '--reasoning', 'off',
               '--temp', '0.6', '--top-p', '0.95', '--top-k', '20', '--no-webui']
    environment = {k: v for k, v in os.environ.items() if k not in ('HF_TOKEN', 'RUNPOD_API_KEY')}
    log_path = ROOT / 'server.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
    for _ in range(240):
        if process.poll() is not None:
            raise RuntimeError('llama-server startup failed: ' + log_path.read_text()[-4000:])
        try:
            if requests.get('http://127.0.0.1:8080/health', timeout=2).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    process.terminate()
    raise RuntimeError('llama-server startup timed out')


def handler(job):
    body = job.get('input')
    if not isinstance(body, dict) or body.get('model') != 'orca-cyber' or not isinstance(body.get('messages'), list):
        raise ValueError('Expected an OrcaCyber chat completion request')
    maximum = body.get('max_tokens', 2048)
    if not isinstance(maximum, int) or not 1 <= maximum <= 4096:
        raise ValueError('max_tokens must be between 1 and 4096')
    ensure_server(job)
    request = {k: v for k, v in body.items() if k in ('model', 'messages', 'tools', 'tool_choice', 'temperature', 'top_p', 'max_tokens')}
    streaming = body.get('stream') is True
    request.update(stream=streaming, cache_prompt=True, chat_template_kwargs={'enable_thinking': False})
    if streaming:
        request['stream_options'] = {'include_usage': True}
    with requests.post('http://127.0.0.1:8080/v1/chat/completions', json=request,
                       stream=streaming, timeout=(10, 540)) as response:
        if not response.ok:
            raise RuntimeError(f'Inference returned HTTP {response.status_code}')
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip()
        backend = {'gpu': gpu, 'speculation': 'draft-dflash', 'revision': REVISION}
        if not streaming:
            yield {'chat': response.json(), 'backend': backend}
            return
        sequence = 0
        for line in response.iter_lines(chunk_size=1):
            if not line.startswith(b'data: '):
                continue
            payload = line[6:]
            if payload == b'[DONE]':
                yield {'seq': sequence, 'done': True, 'backend': backend}
                return
            chunk = json.loads(payload)
            if 'error' in chunk:
                raise RuntimeError('Inference stream failed')
            yield {'seq': sequence, 'chunk': chunk}
            sequence += 1
        raise RuntimeError('Inference stream ended before completion')


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler, 'return_aggregate_stream': True})
