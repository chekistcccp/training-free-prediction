from __future__ import annotations
import argparse
from .pipeline import ExperimentRunner
from .utils import load_yaml

def main():
    p=argparse.ArgumentParser(description='CausalCRC-AD runner'); p.add_argument('--config',default='configs/experiment.yaml'); p.add_argument('command',choices=['all','msd','stageii','slice','localize']); a=p.parse_args(); cfg=load_yaml(a.config); r=ExperimentRunner(cfg); scores=None
    if a.command in {'all','msd','slice','localize'}:
        if cfg['experiments'].get('run_msd_slice_screening',True) or a.command in {'slice','localize'}: scores=r.run_msd_slice_screening()
    if a.command in {'all','msd','localize'}:
        if scores is None: scores=r.run_msd_slice_screening()
        if cfg['experiments'].get('run_msd_oracle_localization',True): r.run_msd_localization(scores,'oracle')
        if cfg['experiments'].get('run_msd_end_to_end_localization',True): r.run_msd_localization(scores,'end_to_end')
    if a.command in {'all','stageii'} and cfg['experiments'].get('run_stageii_external',True): r.run_stageii_external()

if __name__=='__main__': main()
