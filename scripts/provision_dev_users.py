"""Run only via: docker compose exec -T web python manage.py shell < this_file.

Local development only. Existing passwords are never reset. Credentials remain
in the ignored media directory with mode 0600 and are not printed.
"""

import json
import os
from pathlib import Path
import secrets

from django.conf import settings
from django.contrib.auth import get_user_model

if not settings.DEBUG:
    raise RuntimeError(
        "Development users require DJANGO_DEBUG=1; use normal user management elsewhere"
    )

path = Path(settings.MEDIA_ROOT) / "dev-login.json"
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    data = {
        "primary": {"username": "botnet_admin", "password": secrets.token_urlsafe(24)},
        "secondary": {
            "username": "botnet_smoke_other",
            "password": secrets.token_urlsafe(24),
        },
    }
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(data, stream, indent=2)
data = json.loads(path.read_text())
User = get_user_model()
for role, details in data.items():
    user = User.objects.filter(username=details["username"]).first()
    if user is None:
        user = User.objects.create_user(
            username=details["username"],
            password=details["password"],
            is_staff=role == "primary",
            is_superuser=role == "primary",
        )
    elif not user.check_password(details["password"]):
        raise RuntimeError(
            "Existing account has a different password; no password was reset"
        )
    details["id"] = user.pk
path.write_text(json.dumps(data, indent=2) + "\n")
path.chmod(0o600)
print(
    "Local users ready; credentials are in ignored media/dev-login.json (not printed)."
)
