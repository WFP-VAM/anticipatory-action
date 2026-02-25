# Inspired by https://pixi.sh/latest/deployment/container/#example-usage

# Build stage
FROM ghcr.io/prefix-dev/pixi:0.45.0 AS build

WORKDIR /app

# Copy project definition files
COPY pyproject.toml pixi.lock ./

# Install only production dependencies
# use `--locked` to ensure the lockfile is up to date with pyproject.toml
RUN pixi install --locked --environment production

# Create the shell-hook bash script to run the commands in the activated environment
RUN printf "#!/bin/bash\n\n%s\n\n%s" \
     "$(pixi shell-hook -s bash -e production)" \
     'exec "$@"' > /app/entrypoint.sh

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# TODO Figure out why libxml2 is not already pulled in by pix env via conda
RUN apt-get update && apt-get install -y \
    libxml2 libxml2-dev \
    && rm -rf /var/lib/apt/lists/*

    # Copy Python environment from build stage
COPY --from=build /app/.pixi/envs/production /app/.pixi/envs/production
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh

# Copy only necessary project files (excluding .git, data)
COPY AA ./AA

ENV PYTHONPATH="/app"
ENV PATH="/app/.pixi/envs/production/bin:$PATH"

ENTRYPOINT ["/app/entrypoint.sh"]

# Default to running an interactive shell when no command is provided
# Override this with specific commands when needed
CMD ["tail", "-f", "/dev/null"]
