"""Extract observed URL and target metadata without guessing or network requests."""
from urllib.parse import urlsplit


def metadata(raw):
    entities=raw.get('entities')
    if not isinstance(entities,dict):return {'urls':None,'targets':None}
    urls=None;targets=None
    if isinstance(entities.get('urls'),list):
        urls=[]
        for entry in entities['urls']:
            if not isinstance(entry,dict):raise ValueError('Malformed URL entity')
            value=entry.get('expanded_url') or entry.get('url')
            if not isinstance(value,str):raise ValueError('Missing URL entity value')
            parts=urlsplit(value)
            if parts.scheme not in ('http','https') or not parts.hostname:raise ValueError('Invalid URL entity')
            urls.append(value)
    if isinstance(entities.get('user_mentions'),list):
        targets=[]
        for entry in entities['user_mentions']:
            if not isinstance(entry,dict):raise ValueError('Malformed mention')
            uid=entry.get('id_str') or entry.get('id')
            if uid is None:
                targets=None;break
            targets.append(str(uid))
        if targets is not None and raw.get('in_reply_to_user_id') is not None:
            targets.append(str(raw['in_reply_to_user_id']))
        if targets is not None:targets=sorted(set(targets))
    return {'urls':urls,'targets':targets}
