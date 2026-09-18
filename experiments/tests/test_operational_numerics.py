"""Small authored fixtures; these tests never open the retained diagnostic bundles."""
import copy
import json

import numpy as np
import pytest
from scipy.special import logsumexp

from experiment_runner import operational_architecture as oa
from experiment_runner import operational_transport as ot
from test_operational_architecture import training_frame, clouds


def small_config():
    config = json.loads(oa.CONFIG.read_text())
    config['ot']['n_ref'] = 8
    config['wda'].update(n_ref=8, steps=3)
    return config


def independent_residual(h, b, state):
    cost = np.sum((h[:, None, :]-b[None, :, :])**2, axis=2)
    plan = np.exp(((np.array(state['f'])[:, None]+state['g']-cost)/state['epsilon']).astype(np.longdouble))/(len(h)*len(b))
    return float(np.abs(plan.sum(axis=0)-1/np.longdouble(len(b))).sum()
                 + np.abs(plan.sum(axis=1)-1/np.longdouble(len(h))).sum())


def test_observed_slow_balancing_mechanism_and_newton_repair(tmp_path):
    h, b = np.array([[0.], [1.], [3.]]), np.array([[0.], [1.], [4.]])
    cost, epsilon = (h-b.T)**2, .7
    f, g = np.zeros(3), np.zeros(3)
    # Independent reproduction of the archived 10,000-update scheme on this
    # tiny fixture only. No retained multi-hour baseline is executed.
    updates = 0
    for level in (1., .7):
        for _ in range(25 if level == 1. else 10000-updates):
            f = -level*(logsumexp((g-cost)/level, axis=1)-np.log(3))
            g = -level*(logsumexp((f[:, None]-cost)/level, axis=0)-np.log(3))
            shift=f[0]; f,g=f-shift,g+shift
            updates += 1
    assert independent_residual(h,b,{'f':f,'g':g,'epsilon':epsilon}) > 1e-6
    state = ot.sinkhorn(h,b,epsilon,out=tmp_path/'solve')
    assert state['iterations'] < 10
    assert independent_residual(h,b,state) <= 1e-6
    saved=json.loads((tmp_path/'solve/last.json').read_text())
    assert saved['status']=='completed' and saved['f']==state['f']
    assert len((tmp_path/'solve/trace.jsonl').read_text().splitlines()) > 2


def test_rectangular_plan_agrees_with_independent_positive_scaling():
    h=np.array([[-1.], [.3], [2.]])
    b=np.array([[-2.], [0.], [1.], [3.]])
    epsilon=1.4; kernel=np.exp(-(h-b.T)**2/epsilon)
    left=np.ones(3);right=np.ones(4)
    for _ in range(2000):
        left=(1/3)/(kernel@right);right=(1/4)/(kernel.T@left)
    expected=left[:,None]*kernel*right
    state=ot.sinkhorn(h,b,epsilon,tol=1e-12)
    actual=np.exp((np.array(state['f'])[:,None]+state['g']-(h-b.T)**2)/epsilon)/12
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)


@pytest.mark.parametrize('interruption', [False, True])
def test_latest_state_survives_failure_and_interruption(tmp_path,monkeypatch,interruption):
    h,b=np.array([[0.],[1.],[3.]]),np.array([[0.],[1.],[4.]])
    if interruption:
        def stop(*args,**kwargs):raise KeyboardInterrupt('fixture signal')
        monkeypatch.setattr(ot,'cg',stop)
    with pytest.raises(KeyboardInterrupt if interruption else ValueError):
        ot.sinkhorn(h,b,.7,max_iters=1,out=tmp_path/'state')
    state=json.loads((tmp_path/'state/last.json').read_text())
    assert state['status']==('interrupted' if interruption else 'failed')
    assert independent_residual(h,b,state)==pytest.approx(state['marginal_l1'],abs=1e-14)
    assert state['problem_hash']==oa.sha256_json(json.loads((tmp_path/'state/problem.json').read_text()))
    with pytest.raises(FileExistsError):ot.sinkhorn(h,b,.7,out=tmp_path/'state')


def test_preparation_before_solve_and_verified_reuse(tmp_path,monkeypatch):
    config=small_config();frame=training_frame(12)
    choices={'wda_steps':3,**{k:{'kappa':.02 if k in ('CAND','-REP') else None,'rho':0}
                            for k in ('CAND','-OT','-REP','S1')}}
    directory=tmp_path/'preparation'
    original=oa.sinkhorn
    monkeypatch.setattr(oa,'sinkhorn',lambda *a,**kw:pytest.fail('preparation attempted a solve'))
    train,prepared,e1=oa.prepare_final(frame,config['module_configs']['simulator'],choices,config,directory)
    assert set(prepared['coordinates'])=={'CAND','-REP'}
    assert (directory/'training_e1_features.parquet').exists()
    monkeypatch.setattr(oa,'current_fit_features',lambda *a,**kw:pytest.fail('reuse refit E1'))
    monkeypatch.setattr(oa,'fit_wda',lambda *a,**kw:pytest.fail('reuse refit WDA'))
    loaded=oa.load_final_preparation(directory,frame,config['module_configs']['simulator'],choices,config)
    assert loaded[1:]==(prepared,e1)
    for k in train:np.testing.assert_array_equal(train[k],loaded[0][k])
    monkeypatch.setattr(oa,'sinkhorn',original)
    first=oa.refit_stages(train,choices,config,prepared=prepared)
    again=oa.refit_stages(loaded[0],choices,config,prepared=loaded[1])
    assert first==again
    monkeypatch.setattr(oa,'select_stages',lambda *a,**kw:pytest.fail('reuse repeated selection'))
    reused=tmp_path/'reused';reused.mkdir()
    fitted=oa.fit_architecture(frame,config['module_configs']['simulator'],config,reused,reuse_training=directory)
    assert fitted['states']==first['states'] and fitted['e1_full']==e1['e1_full']
    assert (reused/'training_reuse.json').exists()
    changed=frame.copy();changed['spectrum']=[np.asarray(v)+1e-6 for v in frame.spectrum]
    with pytest.raises(ValueError,match='incompatible'):
        oa.load_final_preparation(directory,changed,config['module_configs']['simulator'],choices,config)
    reshaped=frame.copy();reshaped['spectrum']=[np.asarray(v)[None,:] for v in frame.spectrum]
    with pytest.raises(ValueError,match='incompatible'):
        oa.load_final_preparation(directory,reshaped,config['module_configs']['simulator'],choices,config)
    other=copy.deepcopy(config);other['wda']['learning_rate']*=2
    with pytest.raises(ValueError,match='incompatible'):
        oa.load_final_preparation(directory,frame,config['module_configs']['simulator'],choices,other)
    state_path=directory/'preparation.json';state_path.write_text(state_path.read_text()+' ')
    with pytest.raises(ValueError,match='hash mismatch'):
        oa.load_final_preparation(directory,frame,config['module_configs']['simulator'],choices,config)


def test_selection_recovery_verifies_original_argmax_and_ties():
    config=small_config();x,y,ids=clouds();numeric={'v':x,'labels':y,'ids':ids,'c':np.zeros(len(x))}
    choices,saved=oa.select_stages(numeric,numeric,config)
    assert oa.choices_from_selection(saved,config)==choices
    for v in saved['history']['CAND']['kappa']:v['development_frame_auroc']=.5
    saved['selected_states']['CAND']['coordinate']['kappa']=config['ot']['kappa_grid'][0]
    assert oa.choices_from_selection(saved,config)['CAND']['kappa']==config['ot']['kappa_grid'][0]
    saved['selected_states']['CAND']['coordinate']['kappa']=config['ot']['kappa_grid'][-1]
    with pytest.raises(ValueError,match='argmax'):oa.choices_from_selection(saved,config)
