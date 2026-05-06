# llmr [WORK IN PROGRESS - NOT READY FOR USE]

**Drop a GGUF in. Get an API out.** Zero config, auto-optimized.

A tiny CLI that currently runs GGUF models through llama.cpp Docker servers with automatic hardware detection, model discovery, and tuned inference profiles, all in one command. vLLM and SGLang support are planned and the backend boundary is being kept explicit so those adapters can land without changing the CLI shape.

## Quick Start

```bash
# Install
cargo install --path .

# Go (auto-finds GGUF models, detects hardware, optimizes settings)
llmr serve
```

That's it. Your model is live at `http://localhost:8080`.

## Why llmr

- **No config** — Detects CPU, GPU, VRAM, and RAM automatically
- **No hunting** — Scans your drives for GGUF files on first run
- **No guesswork** — Benchmarks your hardware and caches optimal profiles
- **No lock-in** — Runs in Docker, works on Linux, macOS, and Windows
- **No manual daemon step** — If Docker is installed but stopped, `serve` and `tune` try to start it before running containers or tuning benchmarks

## Commands

| Command | Description |
|---------|-------------|
| `llmr serve` | Start a llama.cpp server (auto-discovers GGUF models) |
| `llmr serve -m model.gguf` | Serve a specific GGUF model with llama.cpp |
| `llmr serve --public` | Bind to 0.0.0.0 |
| `llmr serve --no-gpu` | CPU-only mode |
| `llmr status` | Show running containers |
| `llmr stop` | Stop all servers |
| `llmr stop -n <name>` | Stop specific container |
| `llmr profiles list` | List cached profiles |
| `llmr profiles show <key>` | Show profile details |
| `llmr profiles delete <key>` | Delete cached profile |
| `llmr profiles clear` | Clear all cached profiles |
| `llmr tune` | Auto-tune a llama.cpp profile for a GGUF model |
| `llmr bench` | Run benchmarks |
| `llmr bench --tasks gsm8k` | Quality evaluation |
| `llmr doctor` | Run diagnostics |
| `llmr update` | Update to latest version |
| `llmr version` | Show version |

Common options: `-p` port, `-t` threads, `-c` ctx_size, `-g` gpu_layers, `--parallel`, `--split-mode`, `--batch_size`, `--dry-run`, `--debug`.

Run `llmr serve --help` for the full list.

## Backend Support

Current runtime support is llama.cpp only. The code keeps backend selection, image choice, container args, health checks, and tuning profiles behind explicit backend-aware APIs; vLLM and SGLang are planned but intentionally rejected by serve/tune until their container adapters and tuning profiles are implemented.

### Bench Command

Quality evaluation using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
llmr bench --tasks gsm8k
```

Automatically installs `lm-eval[api]` if not present.

For performance benchmarking, use the standalone binary:

```bash
llmr-bench --config config.yaml --output report.json
```

## Prerequisites

- [Rust](https://rustup.rs/) 1.75+
- [Docker](https://docs.docker.com/get-docker/)
- Python 3.10+ (for quality evaluation)

`llmr serve` and `llmr tune` require Docker for real execution. Dry-run flows stay offline and only render the planned command/profile behavior.

## Profiles

Settings are auto-cached per model + hardware combo. On first run, `llmr serve` asks whether to run tuning. If accepted, Docker is started if needed, benchmarks must complete successfully, and only then is the tuned profile saved. Subsequent starts reuse the cached profile.

Config lives at `~/.config/llmr/` (Linux), `~/Library/Application Support/llmr/` (macOS), or `%APPDATA%\llmr\` (Windows).

## Building

```bash
cargo build --release
cargo test
```
