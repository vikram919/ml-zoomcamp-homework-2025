FROM agrigorev/zoomcamp-model:2025


WORKDIR /app

COPY . /app

RUN python -m venv .venv

# Ensure we use the virtual environment for all subsequent commands
ENV PATH=".venv/bin:$PATH"

RUN pip install uv && uv sync --reinstall
EXPOSE 8080

RUN chmod +x docker-entrypoint.sh

ENTRYPOINT ["./docker-entrypoint.sh"]