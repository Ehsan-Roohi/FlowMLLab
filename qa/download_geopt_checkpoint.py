"""Download and verify the pinned, official GeoPT 8-layer checkpoint."""
import argparse,hashlib,urllib.request
from pathlib import Path
REV='45df8bd3d02f9ab36a25d86d0a048e9d72572fd3'
HASH='c0b1b9c4e5d533dbc249190d3d1fbe8e6b066b36d325cf4377cc0d613c02d1c2'
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'GeoPT_8layers.pt'
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=HASH:
        temp=path.with_suffix('.download')
        urllib.request.urlretrieve(f'https://huggingface.co/GeoPT/GeoPT_Pretrained_Models/resolve/{REV}/GeoPT_8layers.pt',temp)
        assert hashlib.sha256(temp.read_bytes()).hexdigest()==HASH,'Checkpoint SHA256 mismatch'
        temp.replace(path)
    print(path)
if __name__=='__main__':main()
