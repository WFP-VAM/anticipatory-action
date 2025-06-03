# Inspired by https://pixi.sh/latest/deployment/container/#example-usage

# Build Pixi environment
FROM ghcr.io/prefix-dev/pixi:0.45.0 AS build

WORKDIR /app

# Copy project definition files
COPY pyproject.toml pixi.lock ./

# Install production dependencies
# use `--locked` to ensure the lockfile is up to date with pyproject.toml
RUN pixi install --locked --environment production

# Create the shell-hook bash script to activate the environment
RUN pixi shell-hook -e production -s bash > /shell-hook
RUN echo "#!/bin/bash" > /app/entrypoint.sh
RUN cat /shell-hook >> /app/entrypoint.sh
# Extend the shell-hook script to run the command passed to the container
RUN echo 'exec "$@"' >> /app/entrypoint.sh

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy Pixi environment and entrypoint from build stage
COPY --from=build /app/.pixi/envs/production /app/.pixi/envs/production
COPY --from=build --chmod=0755 /app/entrypoint.sh /app/entrypoint.sh

# Copy project code
COPY pyproject.toml pixi.lock ./
COPY AA ./AA
COPY config ./config

# Set runtime environment
ENV PYTHONPATH="/app"
ENV PATH="/app/.pixi/envs/production/bin:$PATH"

# Flexible defaults
ENV MODE=""
ENV COUNTRY=""
ENV INDEX=""
ENV VULNERABILITY=""
ENV ISSUE=""

# Activate Pixi environment and dispatch the right command
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash", "-c", "\
    if [ \"$MODE\" = 'triggers' ]; then \
        python -m AA.triggers $COUNTRY $INDEX $VULNERABILITY; \
    elif [ \"$MODE\" = 'analytical' ]; then \
        python -m AA.analytical $COUNTRY $INDEX; \
    elif [ \"$MODE\" = 'operational' ]; then \
        python -m AA.operational $COUNTRY $INDEX $ISSUE; \
    else \
        echo 'Unknown or missing MODE'; exit 1; \
    fi"]
