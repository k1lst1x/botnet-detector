import numpy as np
from engine.graph_features import reference_features


def test_repeated_links_count_other_accounts_not_self():
    X=reference_features([{'a','b'},{'a','b'},{'z'}],[{'a','b'},{'a','b'},{'z'}],training=True)
    assert X[0,1]==np.log1p(1) and X[0,2]==2
    assert X[2,1]==0 and X[2,2]==0


def test_single_match_and_popular_tokens_do_not_form_repeated_edge():
    X=reference_features([{'a'},{'a'},{'a'}],[{'a'}],max_members=2)
    assert np.array_equal(X,np.zeros((1,4)))
    X=reference_features([{'a'}],[{'a','b'}])
    assert X[0,1]==0 and X[0,2]==1


def test_validation_queries_cannot_change_reference():
    ref=[{'a','b'},{'c'}]
    one=reference_features(ref,[{'a','b'}])
    batch=reference_features(ref,[{'a','b'},{'a','b','c'}])
    np.testing.assert_array_equal(one[0],batch[0])


def test_empty_queries_are_zero_and_token_order_does_not_matter():
    ref=[{'a','b'},{'a','b'}]
    X=reference_features(ref,[set(),{'b','a'}])
    np.testing.assert_array_equal(X[0],np.zeros(4))
    assert X[1,2]==2 and X[1,3]==1
