"""Finalize the user-authorized assistant/rule grading with full coverage checks."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/ulvr_vopd_4b_seed42_20261018'
BENCHES={'vstar':191,'zoombench':845,'hrbench-4k':800,'hrbench-8k':800}
TAGS=('step_30','step_65')

def main():
    decisions=[]
    for name in ('explicit-option-decisions.json','numeric-decisions.json','assistant-semantic-decisions.json'):
        decisions+=json.loads((ROOT/'reports'/name).read_text())
    indexed={(x['tag'],x['benchmark'],x['row']):x for x in decisions}
    assert len(indexed)==len(decisions)==2310
    dest=RUN/'assistant-reviewed-eval'
    sources={}
    for tag in TAGS:
        sources[tag]={}
        for bench,n in BENCHES.items():
            original=RUN/'eval'/tag
            path=RUN/'assistant-review'/tag/'judge'/bench/f'{tag}_answer.jsonl'
            rows=json.loads(path.read_text())
            raw=[json.loads(line) for line in (original/'model_answer'/bench/f'{tag}_answer.jsonl').read_text().splitlines() if line.strip()]
            assert len(rows)==len(raw)==n
            assert len({x['sample_uid'] for x in rows})==n
            counts={}
            for i,row in enumerate(rows):
                assert (row['sample_uid'],row['model_answer'],row['query'],row['response'])==(raw[i]['sample_uid'],raw[i]['model_answer'],raw[i]['query'],raw[i]['response'])
                assert row['model_answer'].strip() and not row['model_answer'].startswith(('[API_ERROR]','[FUTURE_ERROR]'))
                if row['judge']=='PENDING_ASSISTANT_REVIEW':
                    d=indexed[(tag,bench,i)]
                    assert d['sample_uid']==row['sample_uid']
                    row.update(judge=d['judge'],judge_source=d['source'],judge_evidence=d.get('reason') or d.get('evidence'),judge_selected_option=d.get('selected_option'))
                    if d.get('ambiguous'):row['judge_ambiguous']=True
                else:
                    assert row['judge']=='Yes'
                    row['judge_source']='upstream_rule:'+row['judge_source']
                assert row['judge'] in ('Yes','No')
                counts[row['judge_source']]=counts.get(row['judge_source'],0)+1
            sources[tag][bench]=counts
            p=dest/tag/'judge'/bench/f'{tag}_answer.jsonl';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
            meta=json.loads((original/f'{bench}.manifest.json').read_text());meta.update(judge_model='assistant-with-explicit-option-and-integer-rules-v1',judge_api_base='local:assistant-review',judge_protocol='official positive rules retained; explicit selected option/integer comparison; assistant semantic fallback',note='User authorized replacing Qwen2.5-72B judge. Four-benchmark result; MME deliberately skipped. Not directly equivalent to Vision-OPD judge scores.')
            (dest/tag/f'{bench}.manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    report=ROOT/'reports/assistant-four-benchmark'
    subprocess.run([sys.executable,str(ROOT/'eval-kit/summarize.py'),'--eval-dir',str(dest),'--models',*TAGS,'--benchmarks',','.join(BENCHES),'--out',str(report),'--sample','3','--seed','42'],check=True)
    (report/'grading-sources.json').write_text(json.dumps(sources,indent=2)+'\n')
    s=json.loads((report/'summary.json').read_text());print(json.dumps({'benchmarks':s['benchmarks'],'macro':s['macro_accuracy'],'micro':s['micro_accuracy']},indent=2))

if __name__=='__main__':main()
