FROM ghcr.io/astral-sh/uv:0.6-debian-slim
WORKDIR /app
COPY .python-version pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-dev --no-install-project --compile-bytecode
COPY . .
EXPOSE 3000
ENV UV_NO_SYNC=1 UV_FROZEN=1 UV_OFFLINE=1 UV_COMPILE_BYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT []
CMD ["python", "-m", "src.app.main"]
