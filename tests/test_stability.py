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
