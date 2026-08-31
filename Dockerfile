# Runtime image for the API, worker, scheduler, and migrate commands.
#
# Deliberate choices:
#   * `--no-dev` and no `--all-extras`: the previous image installed pytest and
#     the dev group and copied `tests/` into production. Test tooling is not
#     something a deployed container should be able to run.
#   * a non-root user: nothing in this image needs root at runtime, and the
#     artifact root is the only writable path it needs.
#   * `--frozen`: the image is built from `uv.lock`, never from a fresh resolve.
FROM python:3.14-slim

# Pinned so an image rebuild cannot silently change the installer.
ARG UV_VERSION=0.11.2

WORKDIR /app

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md alembic.ini ./
COPY src ./src
COPY migrations ./migrations
COPY scripts ./scripts
COPY draft_assistant/data ./draft_assistant/data

RUN uv sync --frozen --no-dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_SYNC=1 \
    # Fail closed: an image started without explicit configuration refuses to
    # boot rather than running with development defaults. `docker-compose.yml`
    # and any deployment supply the real values through the environment.
    APP_ENV=production \
    ARTIFACT_LOCAL_ROOT=/var/lib/fantasy-app/artifacts

# The artifact root is the only path the application writes to.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /var/lib/fantasy-app/artifacts \
    && chown -R appuser:appuser /var/lib/fantasy-app /app

USER appuser

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "src.app.cli", "api", "--host", "0.0.0.0", "--port", "8000"]
