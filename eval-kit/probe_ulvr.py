"""Inspect current HF metadata and download tiny validation previews, not full shards."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

DATASET = 'RuoliuYang/ULVR_v2_clean'
CONFIGS = ['text_cot','bbox_highlight','bbox_crop','depth','edge','segmentation','helper_interleaved','scene_graph']

def get(url):
    with urlopen(url, timeout=60) as response:
        return json.load(response)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=Path('ulvr-preview'))
    p.add_argument('--download-first-image',action='store_true')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    meta=get('https://huggingface.co/api/datasets/'+DATASET)
    sizes=get('https://datasets-server.huggingface.co/size?'+urlencode({'dataset':DATASET}))
    def fetch(config):
        result=get('https://datasets-server.huggingface.co/rows?'+urlencode({
            'dataset':DATASET,'config':config,'split':'validation','offset':0,'length':1}))
        row=result['rows'][0]['row']
        wrapped=json.loads(row['messages_json'])
        # Only input_image + question can be exposed to the student at evaluation time.
        # Gold assistant messages and intermediate helpers remain separate.
        input_row={k:row[k] for k in ['sample_id','source_dataset','question']}
        input_row.update({'config':config,'category':row['category'], 'input_image':row['input_image']})
        gold={k:row[k] for k in ['answer','num_intermediate_steps']}
        overview={'config':config,'sample_id':row['sample_id'],'category':row['category'],
                  'source_dataset':row['source_dataset'],'num_intermediate_steps':row['num_intermediate_steps'],
                  'messages_wrapper':list(wrapped), 'fields':list(row)}
        return input_row,gold,overview
    records=list(ThreadPoolExecutor(max_workers=4).map(fetch,CONFIGS))
    inputs=[r[0] for r in records]
    if args.download_first_image:
        src=inputs[0]['input_image']['src']
        if meta['sha'] not in src:
            raise ValueError('Viewer image revision differs from dataset metadata; rerun when cache is current')
        with urlopen(src,timeout=60) as response:
            (args.out/'text_cot_input.jpg').write_bytes(response.read())
        inputs[0]['input_image_local']='text_cot_input.jpg'
    (args.out/'inputs.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2))
    (args.out/'gold.json').write_text(json.dumps([{'config':r[0]['config'],'sample_id':r[0]['sample_id'],**r[1]} for r in records],ensure_ascii=False,indent=2))
    (args.out/'format.json').write_text(json.dumps([r[2] for r in records],ensure_ascii=False,indent=2))
    (args.out/'metadata.json').write_text(json.dumps({'dataset':DATASET,'revision':meta['sha'],'size':sizes},indent=2))
    for r in records:print(r[2]['config'],r[2]['sample_id'],r[2]['category'],flush=True)

if __name__=='__main__':main()
