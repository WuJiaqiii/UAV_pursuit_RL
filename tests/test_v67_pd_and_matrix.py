"""V6.7 regression tests: teacher parity, train-only behavior, config and launcher."""
import copy
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

from pursuit_rl.config import load_config
from pursuit_rl.control.guidance import pd_intercept
from pursuit_rl.control.torch_guidance import TorchGuidanceController
from pursuit_rl.simulation.environment import PursuitBatch
from pursuit_rl.simulation.torch_backend import TorchPursuitBatch
from pursuit_rl.simulation.pd_teacher import pd_teacher
from pursuit_rl.training.trainer import Trainer

ROOT=Path(__file__).resolve().parents[1]

def test_teacher_off_in_old_default_and_on_in_basic():
    assert load_config('configs/smoke.yaml')['training']['aux_pd_coef'] == 0
    a=load_config('configs/basic_v67/s0_standard_pd02.yaml')
    b=load_config('configs/basic_v67/s0_standard_pure.yaml')
    assert a['training']['aux_pd_coef'] == .2
    assert b['training']['aux_pd_coef'] == 0
    assert a['training']['challenge_eval_enabled'] is False
    assert a['physics']['dt']==.001 and a['physics']['policy_interval_steps']==1
    assert a['curriculum'][0]['time'] == 40.
    assert a['reward']['w_progress'] == 8.

def test_teacher_matches_legacy_numpy_pd():
    c=load_config('configs/basic_v67/smoke.yaml')
    a=PursuitBatch(c,n=4,seed=25)
    np.testing.assert_allclose(pd_teacher(a),pd_intercept(a),atol=2e-6,rtol=0)

def test_teacher_matches_torch_pd():
    c=load_config('configs/basic_v67/smoke.yaml')
    a=TorchPursuitBatch(c,n=4,seed=25,device='cpu')
    actual=pd_teacher(a)
    expected=TorchGuidanceController('PD').act(a)
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=0)
    assert actual.shape==(4,4)
    assert actual.abs().max() <= 1

def test_linear_teacher_weight():
    c=load_config('configs/basic_v67/smoke.yaml')
    c['training']['aux_pd_schedule']='linear'
    c['training']['aux_pd_decay_steps']=100
    # Test schedule without expensive environment construction.
    t=object.__new__(Trainer)
    t.training_config=c['training']
    for step,want in [(0,.2),(50,.1),(100,0.),(150,0.)]:
        t.steps=step
        assert t.auxiliary_coefficient()==pytest.approx(want)

def test_short_tensor_train_reports_aux_and_independent_flag(tmp_path):
    c=load_config('configs/basic_v67/smoke.yaml')
    c['device']='cpu'
    c['simulation_backend']='torch'
    t=Trainer(c,tmp_path/'pd_teacher')
    t.exp_logger.use_tensorboard=False
    t.run()
    rows=[json.loads(x) for x in (tmp_path/'pd_teacher'/'train.jsonl').read_text().splitlines()]
    val=[json.loads(x) for x in (tmp_path/'pd_teacher'/'validation.jsonl').read_text().splitlines()]
    assert rows[0]['aux_pd_coef']==pytest.approx(.2)
    assert rows[0]['aux_pd_loss']>=0
    assert 'progress' in rows[0]['reward_components']
    assert val[0]['challenge_independent'] is False
    assert (tmp_path/'pd_teacher'/'checkpoints'/'best_stage0.pt').exists()

def test_launch_script_dry_run(tmp_path):
    cmd=[sys.executable,str(ROOT/'scripts/basic_experiment_matrix.py'),
         '--group','easy','--only','easy_pure','easy_pd02',
         '--seeds','42','43','--out-root',str(tmp_path/'launch'),
         '--dry-run']
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    assert p.returncode==0,p.stderr
    assert 'DRY RUN: 4 separate train processes' in p.stdout
    assert not (tmp_path/'launch').exists()
