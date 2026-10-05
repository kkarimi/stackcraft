FROM python:3.13-slim-bookworm@sha256:5024f48ba9441d4b13a95d3945abc6365538e3a31109833367a1923523c6efed AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy
WORKDIR /opt/build
COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-editable --python /usr/local/bin/python

FROM python:3.13-slim-bookworm@sha256:5024f48ba9441d4b13a95d3945abc6365538e3a31109833367a1923523c6efed AS runtime
RUN useradd --create-home --uid 1000 stackcraft
COPY --from=builder --chown=1000:1000 /opt/venv /opt/venv
COPY --chown=1000:1000 LICENSE NOTICE /home/stackcraft/
USER 1000
WORKDIR /home/stackcraft
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD /opt/venv/bin/python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '7860') + '/health', timeout=3)"
CMD ["/bin/sh", "-c", "exec /opt/venv/bin/stackcraft serve --host 0.0.0.0 --port \"${PORT:-7860}\""]
