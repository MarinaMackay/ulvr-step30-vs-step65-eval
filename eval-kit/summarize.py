"""Compare identical question sets and sample paired successes/failures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from run import load, uid

def correct(row):
    return str(row['judge']).strip().lower() == 'yes'

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval-dir', type=Path, required=True, help='runs/ directory, or existing Vision-OPD/eval for --legacy')
    p.add_argument('--models', nargs='+', default=['base', 'step_30', 'step_65'])
    p.add_argument('--benchmarks', default='vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn')
    p.add_argument('--out', type=Path, default=Path('report'))
    p.add_argument('--legacy', action='store_true', help='existing evaluator outputs without kit manifests; protocol unverified')
    p.add_argument('--sample', type=int, default=3, help='cases per benchmark/category/change group')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stats, category_stats, cases = [], [], []
    protocol_fields = ['judge_model', 'judge_api_base', 'temperature', 'max_tokens', 'thinking',
                       'benchmark_sha256', 'judge_sha256', 'n', 'smoke', 'evaluator_commit', 'dependencies']
    for bench in args.benchmarks.split(','):
        data, manifests = {}, []
        for model in args.models:
            folder = args.eval_dir if args.legacy else args.eval_dir / model
            path = folder / 'judge' / bench / f'{model}_answer.jsonl'
            rows = load(path)
            ids = [uid(r, bench) for r in rows]
            if not rows or len(ids) != len(set(ids)):
                raise ValueError(f'{path}: empty or duplicate question IDs')
            if any(str(r.get('judge', '')).strip().lower() not in ('yes', 'no')
                   or not str(r.get('model_answer', '')).strip()
                   or str(r.get('model_answer', '')).startswith(('[API_ERROR]', '[FUTURE_ERROR]')) for r in rows):
                raise ValueError(f'{path}: failed inference or malformed judge result')
            data[model] = dict(zip(ids, rows))
            if not args.legacy:
                manifests.append(json.loads((folder / f'{bench}.manifest.json').read_text()))
        if manifests:
            ref = {k: manifests[0][k] for k in protocol_fields}
            if any({k: m[k] for k in protocol_fields} != ref for m in manifests):
                raise ValueError(f'{bench}: model protocols differ')
            if any(m['n'] != len(data[model]) for m, model in zip(manifests, args.models)):
                raise ValueError(f'{bench}: incomplete results')
        first = data[args.models[0]]
        if any(set(records) != set(first) for records in data.values()):
            raise ValueError(f'{bench}: question sets differ; refusing a misleading comparison')
        for records in data.values():
            for key in first:
                if (records[key].get('query'), records[key].get('response')) != (first[key].get('query'), first[key].get('response')):
                    raise ValueError(f'{bench}/{key}: question or gold answer differs')
        counts = {m: sum(correct(r) for r in data[m].values()) for m in args.models}
        stats.append({'benchmark': bench, 'n': len(first),
                      'correct': counts, 'accuracy': {m: 100*c/len(first) for m,c in counts.items()}})
        cats = defaultdict(list)
        for key, row in first.items():
            cats[str(row.get('category') or row.get('source_dataset') or 'unknown')].append(key)
        for cat, keys in sorted(cats.items()):
            category_stats.append({'benchmark': bench, 'category': cat, 'n': len(keys),
                                   'accuracy': {m: 100*sum(correct(data[m][k]) for k in keys)/len(keys) for m in args.models}})
        if len(args.models) >= 2:
            before, after = args.models[-2:]
            for cat, keys in sorted(cats.items()):
                buckets = defaultdict(list)
                for key in sorted(keys):
                    a,b = correct(data[before][key]),correct(data[after][key])
                    group = 'regression' if a and not b else 'improvement' if b and not a else 'both_wrong' if not a and not b else 'both_correct'
                    row = first[key]
                    buckets[group].append({'benchmark': bench, 'category': cat, 'sample_uid': key,
                        'query': row.get('query'), 'gold': row.get('response'), 'images': row.get('images'),
                        'before_model': before, 'after_model': after, 'group': group,
                        'before_answer': data[before][key].get('model_answer'),
                        'after_answer': data[after][key].get('model_answer'),
                        'before_judge_source': data[before][key].get('judge_source'),
                        'after_judge_source': data[after][key].get('judge_source')})
                for group, items in sorted(buckets.items()):
                    if group == 'both_correct': continue
                    rng = random.Random(f'{args.seed}:{bench}:{cat}:{group}')
                    rng.shuffle(items)
                    cases.extend(items[:args.sample])
    n = sum(s['n'] for s in stats)
    macro = {m: sum(s['accuracy'][m] for s in stats)/len(stats) for m in args.models}
    micro = {m: 100*sum(s['correct'][m] for s in stats)/n for m in args.models}
    report = {'protocol_verified': not args.legacy, 'models': args.models, 'benchmarks': stats,
              'categories': category_stats, 'macro_accuracy': macro, 'micro_accuracy': micro,
              'n_total': n, 'warning': 'legacy protocols unverified' if args.legacy else ''}
    (args.out/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (args.out/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cases))
    labels = ['benchmark', 'n', *args.models] + (['delta_pp'] if len(args.models)>1 else [])
    lines = ['# 评测对比', '', '协议核对：'+('未核对（导入旧结果）' if args.legacy else '通过；仍需确认 checkpoint 本身的来源'), '',
             '| '+' | '.join(labels)+' |', '| '+' | '.join(['---']*len(labels))+' |']
    for s in stats + [{'benchmark':'macro（等权）','n':'—','accuracy':macro}, {'benchmark':'micro（题数加权）','n':n,'accuracy':micro}]:
        vals = [s['benchmark'],str(s['n']),*[f"{s['accuracy'][m]:.2f}%" for m in args.models]]
        if len(args.models)>1: vals.append(f"{s['accuracy'][args.models[-1]]-s['accuracy'][args.models[-2]]:+.2f}")
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['', '## Category', '', '| benchmark | category | n | '+' | '.join(args.models)+' |',
              '| --- | --- | --- | '+' | '.join(['---']*len(args.models))+' |']
    for s in category_stats:
        vals=[s['benchmark'],s['category'].replace('|','/'),str(s['n']),*[f"{s['accuracy'][m]:.2f}%" for m in args.models]]
        lines.append('| '+' | '.join(vals)+' |')
    lines += ['', '本实验准备工具从原始 Parquet 补回 MME category；若导入的旧结果为 unknown，需另补原始分类元数据。',
              '', '案例按 benchmark/category/变化方向分层抽取，固定随机种子；cases.jsonl 保留原始回答、题目、图片路径和 judge 来源。']
    (args.out/'comparison.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines[:len(stats)+12]))

if __name__ == '__main__':
    main()
