"""Run the pinned Vision-OPD evaluator against an existing model API."""
import argparse
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / 'vendor' / 'eval'
BENCHES = {
    'vstar': 'vstar.json', 'zoombench': 'zoombench.json',
    'hrbench-4k': 'hr_bench_4k.json', 'hrbench-8k': 'hr_bench_8k.json',
    'mme-realworld': 'MME_RealWorld.json', 'mme-realworld-cn': 'MME_RealWorld_CN.json',
    'ulvr-validation': '../ulvr/validation.json',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(path):
    text = Path(path).read_text(encoding='utf-8')
    if text.lstrip().startswith('['):
        return json.loads(text)
    return [json.loads(x) for x in text.splitlines() if x.strip()]

def uid(row, benchmark):
    if row.get('sample_uid'):
        return str(row['sample_uid'])
    for key in ('uid', 'index', 'question_id', 'id'):
        if row.get(key) is not None and str(row[key]):
            return f'{benchmark}:{key}:{row[key]}'
    obj = {'benchmark': benchmark, 'images': row.get('images') or [], 'query': row.get('query', '')}
    return 'sha1:' + hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def validate_answers(path, expected, benchmark):
    rows = load(path)
    ids = [uid(r, benchmark) for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != {uid(r, benchmark) for r in expected}:
        raise ValueError(f'{path}: missing, extra or duplicate answers')
    failed = [r for r in rows if not str(r.get('model_answer', '')).strip()
              or str(r.get('model_answer', '')).startswith(('[API_ERROR]', '[FUTURE_ERROR]'))]
    if failed:
        raise ValueError(f'{len(failed)} inference failures in {path}; rerun before judging')

def prepare(benchmark, data_dir):
    path = data_dir / BENCHES[benchmark]
    if path.exists():
        return path
    from huggingface_hub import HfApi
    spec = importlib.util.spec_from_file_location('official_prepare', VENDOR / 'prepare_data.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    original_download = mod.snapshot_download
    revisions = {}
    def pinned_download(repo_id, **kwargs):
        info = HfApi().dataset_info(repo_id)
        revisions[repo_id] = info.sha
        return original_download(repo_id, revision=info.sha, **kwargs)
    mod.snapshot_download = pinned_download
    data_dir.mkdir(parents=True, exist_ok=True)
    if benchmark == 'vstar':
        rows = mod.prepare_vstar(data_dir)
    elif benchmark == 'zoombench':
        rows = mod.prepare_zoombench(data_dir)
    elif benchmark.startswith('hrbench'):
        rows = mod.prepare_hrbench(data_dir, benchmark)
    else:
        # Dispatch via official main to retain its actual repository mapping.
        old_argv = sys.argv
        try:
            sys.argv = ['prepare_data.py', '--benchmark', benchmark, '--data_dir', str(data_dir)]
            mod.main()
            rows = json.loads(path.read_text())
        finally:
            sys.argv = old_argv
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    path.with_suffix('.provenance.json').write_text(json.dumps(revisions, indent=2))
    return path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='base')
    p.add_argument('--model', default=os.getenv('OPENAI_MODEL_ID'))
    p.add_argument('--api-base', default=os.getenv('API_BASE'))
    p.add_argument('--judge-api-base', default=os.getenv('JUDGE_API_BASE'))
    p.add_argument('--judge-model', default=os.getenv('JUDGE_MODEL'))
    p.add_argument('--checkpoint', default='', help='Checkpoint path or immutable HF revision for provenance')
    p.add_argument('--benchmarks', default='vstar')
    p.add_argument('--limit', type=int, default=0, help='0=full benchmark; positive=first N smoke samples')
    p.add_argument('--max-tokens', type=int, default=32768)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--thinking', choices=['True', 'False'], default='False')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--judge-only', action='store_true')
    p.add_argument('--infer-only', action='store_true', help='Save predictions without requesting judge; grade later with --judge-only')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    p.add_argument('--runs-dir', type=Path, default=ROOT / 'runs')
    args = p.parse_args()
    if args.judge_only and args.infer_only:
        p.error('judge-only and infer-only are mutually exclusive')
    # The upstream judge silently disables mathruler when dependencies are missing.
    # Require it for real evaluations to keep the protocol consistent across models.
    if not args.prepare_only and args.model != 'fixture-model':
        from mathruler.grader import grade_answer
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.tag) or args.tag in ('.', '..'):
        p.error('tag must be a safe directory name')
    if args.limit < 0 or args.max_tokens < 1 or args.workers < 1:
        p.error('invalid limit/tokens/workers')
    if not args.prepare_only and not all([args.model, args.api_base]):
        p.error('set API_BASE and OPENAI_MODEL_ID')
    if not args.prepare_only and not args.infer_only and not all([args.judge_api_base, args.judge_model]):
        p.error('set JUDGE_API_BASE and JUDGE_MODEL for grading')
    tag = args.tag + (f'_n{args.limit}' if args.limit else '')
    run_dir = args.runs_dir.resolve() / tag
    benches = args.benchmarks.split(',')
    if any(b not in BENCHES for b in benches):
        p.error('unsupported benchmark')
    for bench in benches:
        data_path = prepare(bench, args.data_dir.resolve())
        rows = load(data_path)
        if args.prepare_only:
            print(f'{bench}: {len(rows)} examples at {data_path}')
            continue
        # Check the actual served ID before recording a checkpoint-tagged evaluation.
        if args.model != 'fixture-model' and not args.judge_only:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'), base_url=args.api_base)
            served = {m.id for m in client.models.list().data}
            if args.model not in served:
                raise ValueError(f'Requested model {args.model!r} not found in served IDs: {sorted(served)}')
        selected = rows[:args.limit] if args.limit else rows
        if not selected:
            raise ValueError('empty benchmark')
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / f'{bench}.json'
        manifest = {
            'tag': tag, 'benchmark': bench, 'n': len(selected), 'smoke': bool(args.limit),
            'model': args.model, 'checkpoint': args.checkpoint, 'api_base': args.api_base,
            'judge_model': args.judge_model, 'judge_api_base': args.judge_api_base,
            'temperature': 0, 'max_tokens': args.max_tokens, 'thinking': args.thinking,
            'benchmark_sha256': sha(data_path), 'evaluator_commit': '06860e69b5ed9dc24e96ca5c855f3a4ef25976aa',
            'judge_sha256': sha(VENDOR / 'judge_qwenlm.py'),
            'dependencies': {p: version(p) for p in ['openai','pillow','tqdm','mathruler','sympy','pylatexenc'] if importlib.util.find_spec(p.replace('pillow','PIL'))},
            'note': 'API judge failures raise instead of being scored No; all grading rules retained',
        }
        meta_path = run_dir / f'{bench}.manifest.json'
        if meta_path.exists():
            previous = json.loads(meta_path.read_text())
            changed = {k for k in previous.keys() | manifest.keys() if previous.get(k) != manifest.get(k)}
            adding_first_judge = (args.judge_only and changed <= {'judge_model', 'judge_api_base'}
                                  and previous.get('judge_model') is None and previous.get('judge_api_base') is None)
            if changed and not adding_first_judge:
                raise ValueError(f'{meta_path}: settings changed; choose a new tag to avoid mixing results')
        meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        input_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2))
        def call(script, *options):
            subprocess.run([sys.executable, str(VENDOR / script), *map(str, options)], cwd=run_dir, check=True)
        if not args.judge_only:
            call('infer.py', '--benchmark', bench, '--benchmark_json', input_path,
                 '--out_dir', 'model_answer', '--model_name', tag,
                 '--api_base', args.api_base, '--api_key', os.getenv('OPENAI_API_KEY', 'EMPTY'),
                 '--model_id', args.model, '--max_tokens', args.max_tokens,
                 '--parallel_workers', args.workers, '--enable_thinking', args.thinking)
        validate_answers(run_dir / 'model_answer' / bench / f'{tag}_answer.jsonl', selected, bench)
        if args.infer_only:
            continue
        call('judge_qwenlm.py', '--benchmark', bench, '--model', tag,
             '--api_base', args.judge_api_base, '--api_key', os.getenv('JUDGE_API_KEY', 'EMPTY'),
             '--judge_model', args.judge_model)
        validate_answers(run_dir / 'judge' / bench / f'{tag}_answer.jsonl', selected, bench)
        judges = load(run_dir / 'judge' / bench / f'{tag}_answer.jsonl')
        if any(str(r.get('judge', '')).strip().lower() not in ('yes', 'no') for r in judges):
            raise ValueError('judge returned values other than Yes/No; inspect before reporting accuracy')
        call('cal_acc.py', '--benchmark', bench, '--judge_json',
             run_dir / 'judge' / bench / f'{tag}_answer.jsonl', '--benchmark_json', input_path)
    print(f'Results: {run_dir}')

if __name__ == '__main__':
    main()
