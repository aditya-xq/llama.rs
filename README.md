# llmr

**Drop a GGUF in. Get an API out.** Zero config, auto-optimized.

A tiny CLI that runs [llama.cpp](https://github.com/ggerganov/llama.cpp) in Docker with automatic hardware detection, model discovery, and tuned inference profiles, all in one command.

## Quick Start

```bash
# Install
cargo install --path .

# Go (auto-finds models, detects hardware, optimizes settings)
llmr serve
```

That's it. Your model is live at `http://localhost:8080`.

## Why llmr

- **No config** — Detects CPU, GPU, VRAM, and RAM automatically
- **No hunting** — Scans your drives for GGUF files on first run
- **No guesswork** — Benchmarks your hardware and caches optimal profiles
- **No lock-in** — Runs in Docker, works on Linux, macOS, and Windows

## Commands

| Command | Description |
|---------|-------------|
| `llmr serve` | Start server (auto-discovers models) |
| `llmr serve -m model.gguf` | Serve a specific model |
| `llmr serve --public` | Bind to 0.0.0.0 |
| `llmr serve --no-gpu` | CPU-only mode |
| `llmr status` | Show running containers |
| `llmr stop` | Stop all servers |
| `llmr doctor` | Run diagnostics |
| `llmr profiles list` | List cached profiles |

Common options: `-p` port, `-t` threads, `-c` context size, `-g` GPU layers, `-n` parallel slots, `--metrics`, `--dry-run`, `--debug`.

Run `llmr serve --help` for the full list.

## Prerequisites

- [Rust](https://rustup.rs/) 1.75+
- [Docker](https://docs.docker.com/get-docker/)

## Profiles

Settings are auto-cached per model + hardware combo. First run benchmarks your system; subsequent starts are instant.

Config lives at `~/.config/llama.rs/` (Linux), `~/Library/Application Support/llama.rs/` (macOS), or `%APPDATA%\llama.rs\` (Windows).

## Building

```bash
cargo build --release
cargo test
```
