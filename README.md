# llama.rs (aka llmr)

A tiny CLI for running [llama.cpp](https://github.com/ggerganov/llama.cpp) in Docker with automatic hardware detection, model discovery, and optimized inference profiles.

## Features

- Automatic hardware detection (CPU, RAM, GPU, VRAM)
- Smart model discovery - scans local drives for GGUF models
- Optimized model inference profiles auto-generated for your hardware
- Health check and metrics endpoints
- Inference profile management
- Cross-platform (Linux, macOS, Windows)

## Prerequisites

- [Rust](https://rustup.rs/) 1.75+
- [Docker](https://docs.docker.com/get-docker/) daemon
- GGUF model file

## Installation

```bash
cargo build --release
cargo install --path .
```

Or download pre-built binaries from the releases page.

## Quick Start

```bash
# Check environment
llmr doctor

# Serve a model (auto-discovers GGUF models on your system)
llmr serve

# Stop server(s)
llmr stop
```

## Commands

### serve

Start the llama.cpp server in Docker. When run without `--model`, automatically scans for GGUF models on your system.

```bash
llmr serve [options]
```

Options:
- `-m, --model <path>` - Path to GGUF model file (auto-discovers if omitted)
- `-p, --port <port>` - Server port (default: 8080)
- `--metrics` - Enable metrics endpoint
- `--benchmark` - Run optimization benchmarks before starting
- `--no-benchmark` - Skip optimization benchmarks, use cached/default profile
- `--dry-run` - Print Docker command without running
- `--public` - Bind to 0.0.0.0 instead of localhost
- `--skip-hardware` - Skip hardware detection
- `--no-gpu` - Run in CPU-only mode
- `-t, --threads <n>` - Number of threads
- `-c, --ctx-size <n>` - Context size
- `-g, --gpu-layers <n>` - Number of GPU layers
- `--split-mode <mode>` - Split mode: layer, row, none, auto
- `-b, --batch-size <n>` - Batch size
- `-u, --ubatch-size <n>` - Ubatch size
- `--cache-type-k <type>` - KV cache type for K (default: q4_0/f16 based on GPU)
- `--cache-type-v <type>` - KV cache type for V (default: q4_0/f16 based on GPU)
- `-n, --parallel <n>` - Parallel slots (default: 1)
- `--debug` - Enable debug logging

**First run**: When run without `--model`, the CLI scans your disks for GGUF files and caches the locations for faster startup on subsequent runs. On first serve, optimization benchmarks run automatically to find optimal configuration for your hardware.

### status

Show running containers.

```bash
llmr status
llmr status <name>
```

### stop

Stop running containers. Without a name, stops all llmr containers.

```bash
llmr stop
llmr stop <name>
llmr stop <name> --force
```

### profiles

Manage saved profiles.

```bash
llmr profiles list     # List all saved profiles
llmr profiles show <key>   # Show profile details
llmr profiles delete <key> # Delete a profile
llmr profiles clear       # Delete all profiles
llmr profiles --file <path>  # Load profile from file
```

### doctor

Run environment diagnostics.

```bash
llmr doctor
```

### version

Show version information.

```bash
llmr version
```

## Configuration

Config files are stored in:
- Linux: `~/.config/llama.rs/`
- macOS: `~/Library/Application Support/llama.rs/`
- Windows: `%APPDATA%\llama.rs\`

### Profiles

Profiles are cached TOML files with hardware-specific settings, keyed by:
- Model filename
- CPU core count
- GPU count
- Total detected VRAM (MB)

Profiles store launch parameters to skip recomputing on subsequent runs. Use `--retune` to regenerate.

## Troubleshooting

### Docker daemon not running

```bash
# Linux
sudo systemctl start docker
```

### Port already in use

```bash
llmr serve --model model.gguf --port 8081
```

## Building from Source

```bash
cargo build        # Debug
cargo build --release  # Optimized
cargo test         # Run tests
```

## Testing

The project includes comprehensive unit, integration, and E2E tests.

### Running Tests

```bash
# Run all tests
cargo test

# Run unit tests only (89 tests)
cargo test --lib

# Run integration tests
cargo test --test integration

# Run E2E CLI tests
cargo test --test e2e
```

### Adding Tests

- **Unit tests**: Add `#[cfg(test)] mod tests { ... }` in source files
- **Integration tests**: Add to `tests/integration/`
- **E2E tests**: Add to `tests/e2e/cli.rs` using `assert_cmd`
