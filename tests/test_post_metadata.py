from engine.post_metadata import metadata


def test_expanded_link_and_stable_mentions_are_preserved():
    r=metadata({'entities':{'urls':[{'url':'https://t.co/a','expanded_url':'https://example.org/x'}],'user_mentions':[{'id_str':'123'}]},'in_reply_to_user_id':123})
    assert r=={'urls':['https://example.org/x'],'targets':['123']}


def test_missing_metadata_is_not_empty_and_names_are_not_ids():
    assert metadata({})=={'urls':None,'targets':None}
    assert metadata({'entities':{'urls':[],'user_mentions':[]}})=={'urls':[],'targets':[]}
    assert metadata({'entities':{'user_mentions':[{'screen_name':'person'}]}})['targets'] is None
