"""CPU-only pipeline test using real V* images and a fixed-response API fixture.
This is an integration test, never a model evaluation or a real accuracy result.
"""
import json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading
import argparse

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--benchmarks',default='vstar')
    p.add_argument('--split-grading',action='store_true')
    p.add_argument('--runs-dir',type=Path,required=True);args=p.parse_args()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if data['model']=='fixture-judge':
                prompt=data['messages'][0]['content']
                gold=prompt.split('The answer is: ',1)[1].split('\nThe response is: ',1)[0]
                response=prompt.split('The response is: ',1)[1].split('\nPlease check',1)[0]
                answer='Yes' if gold.strip()==response.strip() else 'No'
            else:
                content=data['messages'][0]['content']
                assert content[0]['image_url']['url'].startswith('data:image/')
                answer='A'
            payload=json.dumps({'id':'fixture','object':'chat.completion','created':0,'model':data['model'],
                'choices':[{'index':0,'message':{'role':'assistant','content':answer},'finish_reason':'stop'}]}).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        api=f'http://127.0.0.1:{server.server_port}/v1'
        common=[sys.executable,str(ROOT/'run.py'),'--tag','API_FIXTURE_NOT_MODEL',
            '--model','fixture-model','--api-base',api,
            '--limit','2','--max-tokens','32','--workers','1',
            '--benchmarks',args.benchmarks,
            '--data-dir',str(args.data_dir),'--runs-dir',str(args.runs_dir)]
        judge=['--judge-model','fixture-judge','--judge-api-base',api]
        if args.split_grading:
            subprocess.run(common+['--infer-only'],check=True)
            subprocess.run(common+judge+['--judge-only'],check=True)
            # A different judge after successful grading must fail before changing results.
            changed=subprocess.run(common+['--judge-model','changed-judge','--judge-api-base',api,'--judge-only'],capture_output=True,text=True)
            assert changed.returncode and 'settings changed' in changed.stderr
        else:
            subprocess.run(common+judge,check=True)
    finally:
        server.shutdown();server.server_close()
    print('PASS: real images -> official infer -> fixture judge API -> official cal_acc. Displayed percentage is a fixture result, not model accuracy.')

if __name__=='__main__':main()
