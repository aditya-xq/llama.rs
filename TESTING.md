# Testing

This document covers testing automation, guidelines, and manual testing procedures for ensuring high quality.

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test suites
cargo test --lib          # Library tests
cargo test --test unit    # Unit tests
cargo test --test integration  # Integration tests
cargo test --test e2e    # End-to-end CLI tests
cargo test --test smoke   # Smoke tests
cargo test --test errors  # Error handling tests
cargo test --test security  # Security tests
cargo test --test perf    # Performance tests
```

## Test Suites

### Unit Tests (`tests/unit/`)
- **hardware.rs**: CPU, GPU, RAM detection and conversions
- **scanner.rs**: GGUF model scanning
- **diagnostics.rs**: Environment diagnostics
- **errors.rs**: Error types and handling
- **logger.rs**: Logging infrastructure
- **tuning.rs**: Auto-tuning logic
- **profile_integration.rs**: Profile management
- **public_api.rs**: Public API exports

### Integration Tests (`tests/integration/`)
- **docker.rs**: Docker client and container management
- **model_scanner.rs**: Model discovery and validation
- **profile_manager.rs**: Profile persistence
- **bench.rs**: Benchmark configuration

### End-to-End Tests (`tests/e2e/`)
- **cli.rs**: CLI command-line interface
- **smoke.rs**: Quick sanity checks
- **errors.rs**: Error message validation

### Security Tests (`tests/security/`)
- **cli_injection.rs**: CLI input sanitization

### Performance Tests (`tests/perf/`)
- **benchmark.rs**: Benchmark configuration parsing

## Test Patterns

### Unit Test Example
```rust
use llmr::hardware::{detect, HardwareInfo, CpuInfo};

#[test]
fn test_cpu_info_default() {
    let cpu = CpuInfo::default();
    assert!(cpu.cores > 0);
}

#[tokio::test]
async fn test_detect() {
    let info = detect().await.unwrap();
    assert!(info.cpu.cores > 0);
}
```

### Integration Test Example
```rust
use llmr::docker::DockerClient;
use llmr::errors::Error;

#[tokio::test]
async fn test_docker_client_new() {
    let client = DockerClient::new();
    assert!(client.is_ok() || matches!(client, Err(Error::DockerCliNotFound)));
}
```

### E2E CLI Test Example
```rust
use assert_cmd::Command;
use predicates::prelude::*;

#[test]
fn test_cli_help() {
    let mut cmd = Command::cargo_bin("llmr").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("llmr"));
}
```

## Manual Testing Guide

### Prerequisites
- Docker installed
- GGUF model files for testing
- Python 3.10+ (for quality evaluation)
- Network access for lm-evaluation-harness

### Functional Tests

1. **Serve Command**
```bash
# Auto-discover models
llmr serve

# Serve specific model
llmr serve -m model.gguf

# Dry run to see docker command
llmr serve -m model.gguf --dry-run

# CPU-only mode
llmr serve -m model.gguf --no-gpu

# Public binding
llmr serve -m model.gguf --public
```

2. **Status & Stop**
```bash
llmr status
llmr stop
llmr stop -n <container-name>
```

3. **Profiles**
```bash
llmr profiles list
llmr profiles show <key>
llmr profiles delete <key>
llmr profiles clear
```

4. **Tuning**
```bash
llmr tune -m model.gguf
llmr tune -m model.gguf --dry-run
```

Expected behavior: real tuning requires Docker. If Docker is installed but the daemon is stopped, `llmr tune` should print that it is attempting to start Docker, wait for readiness, then run benchmark containers. It must not save a tuned profile when every benchmark candidate fails.

6. **Backend Boundary**

Current serve/tune support is llama.cpp only. Tests should verify that planned backends such as vLLM and SGLang have explicit metadata but cannot accidentally reuse llama.cpp server arguments. A planned backend profile should fail before Docker execution with a clear "planned but not wired" style error.

5. **Diagnostics**
```bash
llmr doctor
```

### Benchmark Testing

```bash
# Quality evaluation
llmr bench --tasks gsm8k

# With custom server URL
llmr bench -u http://127.0.0.1:8080 --tasks gsm8k

# Performance benchmark
llmr-bench --config config.yaml --output report.json
llmr-bench --config config.yaml --dry-run
```

### Manual Hardware Verification

1. Run `llmr doctor` to verify detection
2. Check CPU cores, GPU detection, RAM reporting
3. Test with different GPU types (NVIDIA, AMD, CPU-only)

### Edge Case Testing

1. **Missing Model**: `llmr serve` without models
2. **Invalid Model Path**: `llmr serve -m nonexistent.gguf`
3. **Port Conflicts**: Run on occupied port
4. **Docker Not Running**: With Docker installed but stopped, `llmr serve` and `llmr tune` should attempt daemon startup before running tuning or serve containers. If startup fails, the command should fail clearly without saving a tuned profile.
5. **No GPU**: Verify CPU-only fallback

### Security Testing

1. Model path injection attempts
2. Shell metacharacter handling
3. Dry-run output sanitization

## Writing Tests

### When to Add Tests
- New CLI commands or options
- Error handling changes
- Hardware detection updates
- Profile management changes
- Docker integration changes
- Backend adapter or profile changes

### Test Conventions
- Use `#[tokio::test]` for async tests
- Use descriptive test names: `test_<feature>_<expected_behavior>`
- Test both success and failure paths
- Mock external dependencies where possible

### Running Specific Tests
```bash
# Single test
cargo test test_hardware_info_default

# Tests matching pattern
cargo test hardware

# With output
cargo test -- --nocapture
```

## CI/CD

Tests run automatically on:
- Pull requests
- Push to main branch
- Release tags

Required checks:
- `cargo fmt -- --check`
- `cargo clippy -- -D warnings`
- `cargo test`
- `cargo build --release`
