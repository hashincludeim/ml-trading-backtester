#!/bin/sh
# Container entrypoint: serve the dashboard with gunicorn on $PORT.
set -e

# The app has no logins, so nothing signed with the key needs to survive a restart: a per-boot
# random key is fine when the host does not provide one.
: "${DJANGO_SECRET_KEY:=$(python -c 'import secrets; print(secrets.token_urlsafe(50))')}"
export DJANGO_SECRET_KEY

exec gunicorn config.wsgi \
    --chdir web \
    --bind "0.0.0.0:${PORT:-8000}" \
    --workers "${WEB_WORKERS:-2}" \
    --threads "${WEB_THREADS:-1}" \
    --timeout 120 \
    --access-logfile -
