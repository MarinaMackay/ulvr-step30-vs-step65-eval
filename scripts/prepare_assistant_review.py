"""Run unchanged upstream grading rules and export semantic cases as PENDING.

Writes only under assistant-review; never represents pending cases as incorrect.
Run again after more benchmark inference files have completed.
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/ulvr_vopd_4b_seed42_20261018'
spec = importlib.util.spec_from_file_location('official_judge', ROOT/'eval-kit/vendor/eval/judge_qwenlm.py')
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)

def main():
    summary=[]
    for tag in ('base','step_30','step_65'):
        source=RUN/'eval'/tag
        for manifest in sorted(source.glob('*.manifest.json')):
            meta=json.loads(manifest.read_text());bench=meta['benchmark']
            answer=source/'model_answer'/bench/f'{tag}_answer.jsonl'
            if not answer.exists():
                continue
            rows=[json.loads(line) for line in answer.read_text().splitlines() if line.strip()]
            if len(rows)!=meta['n'] or any(str(x.get('model_answer','')).startswith(('[API_ERROR]','[FUTURE_ERROR]')) for x in rows):
                continue
            work=RUN/'assistant-review'/tag
            work.mkdir(parents=True,exist_ok=True)
            link=work/'model_answer'
            if not link.exists():
                link.symlink_to(source/'model_answer',target_is_directory=True)
            destination=work/'judge'/bench/f'{tag}_answer.jsonl'
            if destination.exists():
                continue
            pending=[]
            def queue(prompts,*args,**kwargs):
                pending.extend(prompts)
                return ['PENDING_ASSISTANT_REVIEW']*len(prompts)
            judge.judge_via_api=queue
            os.chdir(work)
            sys.argv=['official_judge','--benchmark',bench,'--model',tag,
                      '--api_base','assistant-review-no-network','--judge_model','assistant']
            judge.main()
            results=json.loads(destination.read_text())
            cases=[dict(sample_uid=x.get('sample_uid'),query=x['query'],gold=x['response'],
                        extracted_answer=x['extracted_answer'],model_answer=x['model_answer'])
                   for x in results if x['judge']=='PENDING_ASSISTANT_REVIEW']
            (destination.parent/'semantic-review-queue.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2))
            summary.append(dict(tag=tag,benchmark=bench,total=len(rows),pending=len(cases)))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    main()
