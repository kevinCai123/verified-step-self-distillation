import argparse
import json
import time
from urllib.request import urlopen

parser=argparse.ArgumentParser()
parser.add_argument('--model',default='Qwen/Qwen3.5-9B')
parser.add_argument('--timeout',type=int,default=900)
args=parser.parse_args()
deadline=time.monotonic()+args.timeout
while time.monotonic()<deadline:
    try:
        with urlopen('http://127.0.0.1:8093/v1/models',timeout=5) as handle: result=json.load(handle)
        if args.model in {item['id'] for item in result['data']}:
            print(json.dumps(result)); break
    except OSError:
        pass
    time.sleep(2)
else:
    raise TimeoutError('Model server did not become ready: '+args.model)
