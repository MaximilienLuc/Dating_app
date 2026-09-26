import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from src.stability import session_sample, validate_contract, importance_vector, model_factory, auc_or_nan


def test_session_bootstrap_preserves_both_directions_and_multiplicity():
    class Draw:
        def choice(self, unique, size, replace):
            assert list(unique) == [1, 2, 3] and size == 3 and replace
            return [2, 2, 1]
    np.testing.assert_array_equal(session_sample([1, 1, 2, 2, 2, 3], Draw()), [2, 3, 4, 2, 3, 4, 0, 1])


def contract():
    data = pd.DataFrame({'iid':[1,2,3,4], 'pid':[2,1,4,3], 'wave':[1,1,2,2],
                         'dec':[0,1,0,1], 'x':[1.,2.,3.,4.]})
    return data, {'features':['x'], 'protected':['gender'], 'target':'dec'}, {'group':'wave', 'train_waves':[1], 'test_waves':[2]}


def test_rejects_candidate_only_overlap():
    data, c, s = contract()
    data.loc[2, 'pid'] = 1
    with pytest.raises(ValueError, match='Participant overlap'):
        validate_contract(data, c, s)


def test_rejects_outcome_as_feature():
    d,c,s = contract()
    c['features'].append('match')
    with pytest.raises(ValueError, match='Protected'):
        validate_contract(d,c,s)


def test_accepts_disjoint_complete_sessions():
    a,b = validate_contract(*contract())
    assert list(a.wave) == [1,1] and list(b.wave) == [2,2]


def test_missing_wave_is_not_silently_dropped():
    d,c,s = contract()
    d.loc[0,'wave'] = 3
    with pytest.raises(ValueError, match='cover'):
        validate_contract(d,c,s)


def test_coefficient_units_survive_a_change_of_scaler():
    rng=np.random.default_rng(9)
    X=pd.DataFrame({'x':rng.normal(0,10,80), 'z':rng.normal(0,2,80)})
    y=(X.x + rng.normal(0,10,80) > 0).astype(int)
    m=model_factory('logit').fit(X,y)
    beta=importance_vector(m,'logit',X.std(ddof=0).to_numpy())
    raw_beta=beta/X.std(ddof=0).to_numpy()
    expected=X.to_numpy()@raw_beta + m[-1].intercept_[0] - m[-2].mean_@raw_beta
    np.testing.assert_allclose(expected,m.decision_function(X),atol=1e-10)


def test_all_missing_bootstrap_column_keeps_feature_alignment():
    X=pd.DataFrame({'x':[1.,2.,3.,4.], 'empty':[np.nan]*4})
    m=model_factory('logit').fit(X,[0,1,0,1])
    assert importance_vector(m,'logit',np.array([1.,0.])).shape == (2,)


def test_single_class_auc_is_explicitly_missing():
    assert np.isnan(auc_or_nan([0,0],[.1,.2]))
    assert auc_or_nan([0,1],[.1,.2]) == roc_auc_score([0,1],[.1,.2])


def test_model_specific_features_keep_order_and_do_not_mutate_contract():
    from src.stability import model_features
    c={'features':['a','shar1_1_A','b','shar1_1_B'], 'features_logit':['b','a'],
       'target':'dec','protected':['race']}
    assert model_features('logit',c) == ['b','a']
    assert model_features('xgb',c) == ['a','shar1_1_A','b','shar1_1_B']
    selected=model_features('logit',c)
    selected.append('new')
    assert c['features_logit'] == ['b','a']


def test_legacy_contract_falls_back_to_shared_features():
    from src.stability import model_features
    _, c, _=contract()
    assert model_features('logit',c) == c['features']


def test_rejects_protected_attributes_in_logit_only_list():
    d,c,s=contract()
    c['features_logit']=['x','gender']
    with pytest.raises(ValueError, match='Protected'):
        validate_contract(d,c,s)


def test_rejects_missing_logit_only_column():
    d,c,s=contract()
    c['features_logit']=['not_in_data']
    with pytest.raises(ValueError, match='Missing columns'):
        validate_contract(d,c,s)


def test_run_uses_model_specific_columns_for_reference_and_every_refit(tmp_path, monkeypatch):
    from src import stability
    d,c,s=contract()
    d['extra']=[2.,3.,4.,5.]
    c['features']=['x','extra']
    c['features_logit']=['x']
    (tmp_path/'data').mkdir()
    (tmp_path/'src').mkdir()
    d.to_parquet(tmp_path/'data/clean.parquet',index=False)
    import json
    (tmp_path/'data/features.json').write_text(json.dumps(c))
    (tmp_path/'data/split.json').write_text(json.dumps(s))
    (tmp_path/'01_data_models_v0.py').write_text('# fixture')
    (tmp_path/'src/stability.py').write_text('# fixture')
    frozen=d[['iid','pid','wave']].copy()
    frozen['tabicl_proba']=[.2,.8,.3,.7]
    frozen.iloc[::-1].to_parquet(tmp_path/'data/tabicl_predictions.parquet',index=False)
    seen=[]
    class Estimator:
        def __init__(self,name): self.name=name
        def fit(self,X,y):
            expected=['x'] if self.name=='logit' else ['x','extra']
            assert list(X.columns)==expected
            self.columns=list(X.columns)
            seen.append(self.name)
            return self
        def predict_proba(self,X):
            assert list(X.columns)==self.columns
            return np.column_stack([np.full(len(X),.4),np.full(len(X),.6)])
    monkeypatch.setattr(stability,'model_factory',lambda name,seed:Estimator(name))
    monkeypatch.setattr(stability,'importance_vector',lambda model,name,sd:np.ones(len(model.columns)))
    report=stability.run(tmp_path,tmp_path/'results',n_refits=2,n_eval=2)
    assert seen.count('logit')==3 and seen.count('xgb')==3
    assert set(seen)=={'logit','xgb'}
    assert report['tabicl']['status']=='test_only'
    assert 'flip_rate' not in report['models']['tabicl']
    assert report['models']['tabicl']['reference_auc']==1.0
    assert 'logit_minus_tabicl' in report['paired_auc_differences']
    assert report['features_by_model']=={'logit':['x'],'xgb':['x','extra']}
    assert pd.read_csv(tmp_path/'results/logit_coefficients.csv').feature.tolist()==['x']


def test_tabicl_key_join_restores_test_order_and_allows_train_rows():
    from src.stability import frozen_tabicl_predictions
    d,_,_=contract()
    predictions=d[['iid','pid','wave']].copy()
    predictions['tabicl_proba']=[.1,.2,.3,.4]
    np.testing.assert_allclose(frozen_tabicl_predictions(d.iloc[2:],predictions.iloc[::-1]),[.3,.4])


@pytest.mark.parametrize('failure',['duplicate','missing','nan','range'])
def test_tabicl_rejects_bad_predictions(failure):
    from src.stability import frozen_tabicl_predictions
    d,_,_=contract()
    p=d[['iid','pid','wave']].copy()
    p['tabicl_proba']=.5
    if failure=='duplicate': p=pd.concat([p,p.iloc[:1]])
    if failure=='missing': p=p.iloc[1:]
    if failure=='nan': p.loc[0,'tabicl_proba']=np.nan
    if failure=='range': p.loc[0,'tabicl_proba']=1.1
    with pytest.raises(ValueError): frozen_tabicl_predictions(d,p)
