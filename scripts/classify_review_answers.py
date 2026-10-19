from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/ulvr_vopd_4b_seed42_20261018/assistant-review'
benches=('vstar','zoombench','hrbench-4k','hrbench-8k')
def explicit(text):
 t=text.replace('**','').replace('__','').strip()
 # Only an explicitly selected option, not a mention in an explanation/options list.
 if re.fullmatch(r'[\s\(\[]*[A-F][\s\)\].!]*',t):
  return re.search('[A-F]',t).group(),t
 short=re.match(r'^\(?([A-F])\)?(?:[.\):]|\s|$)',t)
 if short and len(t.splitlines())==1:return short.group(1),t
 matches=[]
 for pat in [r'(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|option|choice)\s*(?:is\s*|would be\s*)?[:：=\-]?\s*(?:option\s+)?[\(\[]?([A-F])(?:[\)\].:：\s]|$)',r'\\boxed\s*\{\s*\(?([A-F])\)?\s*\}']:
  matches.extend((m.start(),m.group(1),m.group()) for m in re.finditer(pat,t,re.I) if m.group(1).isupper())
 if matches:
  # Last explicit conclusion is decisive, but flag conflicting explicit conclusions for review.
  choices={x[1] for x in matches}
  if len(choices)==1:
   z=max(matches);return z[1],z[2]
 # A standalone selected option on the last non-empty line.
 lines=[l.strip().lstrip('✅✔️ ').strip() for l in t.splitlines() if l.strip()]
 if lines:
  m=re.match(r'^\(?([A-F])\)?(?:[.\):]|\s|$)',lines[-1])
  if m and not re.search(r'\b(?:incorrect|wrong|not correct)\b',lines[-1],re.I):return m.group(1),lines[-1]
 return None,None
remaining=[];resolved=[]
for tag in ('step_30','step_65'):
 for bench in benches:
  path=BASE/tag/'judge'/bench/f'{tag}_answer.jsonl';rows=json.loads(path.read_text())
  for i,row in enumerate(rows):
   if row['judge']!='PENDING_ASSISTANT_REVIEW':continue
   answer=row['extracted_answer'];choice,evidence=explicit(answer)
   gold=row['response'].strip()
   rec=dict(tag=tag,benchmark=bench,row=i,sample_uid=row.get('sample_uid'),gold=gold,answer=answer,query=row['query'])
   if choice and re.fullmatch('[A-F]',gold):
    rec.update(judge='Yes' if choice==gold else 'No',selected_option=choice,evidence=evidence,source='explicit_option_rule')
    resolved.append(rec)
   else:remaining.append(rec)
(ROOT/'reports/explicit-option-decisions.json').write_text(json.dumps(resolved,ensure_ascii=False,indent=2))
for i,x in enumerate(remaining):x['review_id']=i
(ROOT/'reports/semantic-review-remaining.json').write_text(json.dumps(remaining,ensure_ascii=False,indent=2))
print('explicit',len(resolved),'remaining',len(remaining))
from collections import Counter
print(Counter(x['benchmark'] for x in remaining))
