"""Settings for the test suite: throwaway data dir, in-memory DB, no caching."""

import os
import tempfile

os.environ.setdefault("DJANGO_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp(prefix="stockml-test-"))

from config.settings.base import *  # noqa: F403

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}
CACHES = {"default": {"BACKEND": "django.core.cache.backends.dummy.DummyCache"}}
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.FileSystemStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
STATIC_ROOT = None  # skip WhiteNoise's missing-directory warning in tests
