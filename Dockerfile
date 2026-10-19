FROM ghcr.io/ggml-org/llama.cpp@sha256:395d01bb47eb23d5e967129c6ebd2b5969365401621e16dd3db33e50fa7462e3
# Build-time dependencies only: no pip/network installation on worker startup.
RUN python3 -m pip install --break-system-packages --disable-pip-version-check --no-cache-dir \
       runpod==1.12.0 huggingface_hub==2.2.0 hf_xet==1.7.0
COPY worker.py /opt/orca/worker.py
COPY chat_template.jinja /opt/orca/chat_template.jinja
ENV ORCA_CHAT_TEMPLATE=/opt/orca/chat_template.jinja \
    ORCA_REQUIRE_CACHE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "-u", "/opt/orca/worker.py"]
CMD []
