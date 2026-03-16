FROM node:20-bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    NODE_ENV=space \
    PORT=7860 \
    PYTHON_EXECUTABLE=python3

WORKDIR $HOME/app

COPY --chown=user package.json package-lock.json requirements.txt ./

RUN npm ci --omit=dev \
    && python3 -m pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

EXPOSE 7860

CMD ["npm", "start"]
