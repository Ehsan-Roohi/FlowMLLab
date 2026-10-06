"""Run fresh training explicitly; never invoked by the document builders."""
import argparse
from dataclasses import replace
from pathlib import Path
from flowmllab.transformer_course import Protocol, run

parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--quick',action='store_true',help='Smoke protocol, not release evidence')
args=parser.parse_args()
protocol=Protocol()
if args.quick:
    protocol=replace(protocol,seeds=(17,),max_steps=30,pretrain_steps=30,adaptation_steps=30,transfer_blocks=(4,))
run(Path(__file__).resolve().parents[1],args.output,protocol)
