import numpy as np
from engine.research_features import enhance


def test_balance_and_link_diversity_are_finite_with_zero_counts():
    names=['followers_log1p','following_log1p','url_ratio','unique_domains']
    X=np.array([[0,0,0,0],[np.log1p(9),np.log1p(3),.5,2]])
    Y,new=enhance(X,names,np.array([20,20]))
    assert np.isfinite(Y).all()
    np.testing.assert_allclose(Y[1,-4:],[np.log(2.5),np.log(13),.3,.2])
    assert Y.shape==(2,len(new))
