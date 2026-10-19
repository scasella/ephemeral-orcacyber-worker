# Ephemeral OrcaCyber worker

Minimal Runpod Serverless worker for OrcaCyber-Nano-1.0 with DFlash2 speculative decoding and streaming Chat Completions. The image contains the serving code, a chat template, and pinned dependencies. It does not contain model weights or credentials.

Configure Runpod's cached model as `orcarouter/OrcaCyber-Nano-1.0`, with authorized Hugging Face access. The worker uses the pinned snapshot `fe6ba411d0941f61643b6406614875b292982142`. Keep workersMin=0 to scale to zero.

Model licensing and access terms apply independently; obtain access from its publisher. Never bake tokens into this image.
