# Image for Google Cloud Run, or any other container host.
# Prices and training results are produced beforehand (by .github/workflows/deploy.yml, or locally
# with fetch_prices + train_models) and copied in as data/, so the container never downloads
# data or trains models.
FROM python:3.12-slim

RUN useradd --create-home --uid 1000 app
USER app
ENV PATH=/home/app/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_DEBUG=false \
    DJANGO_ALLOWED_HOSTS=.run.app,localhost,127.0.0.1 \
    DJANGO_SECURE_COOKIES=true \
    PORT=8000
WORKDIR /home/app/stockml

COPY --chown=app pyproject.toml README.md ./
COPY --chown=app src ./src
RUN pip install --no-cache-dir --user ".[deploy]"

COPY --chown=app web ./web
COPY --chown=app deploy ./deploy
COPY --chown=app data ./data
RUN export DJANGO_SECRET_KEY=build-only \
    && python web/manage.py collectstatic --noinput \
    && python web/manage.py migrate --noinput

EXPOSE 8000
CMD ["sh", "deploy/start.sh"]
