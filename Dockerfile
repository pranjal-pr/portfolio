FROM node:20-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

USER node

ENV HOME=/home/node \
    VIRTUAL_ENV=/home/node/.venv \
    PATH=/home/node/.venv/bin:$PATH \
    NODE_ENV=space \
    PORT=7860 \
    PYTHON_EXECUTABLE=python

WORKDIR $HOME/app

COPY --chown=node:node package.json package-lock.json requirements.txt ./

RUN python3 -m venv "$VIRTUAL_ENV" \
    && npm ci --omit=dev \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=node:node . .

EXPOSE 7860

CMD ["npm", "start"]
