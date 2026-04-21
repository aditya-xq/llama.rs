use std::env;
use std::io::{stderr, stdout, IsTerminal};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub struct Logger;

impl Logger {
    pub fn setup(verbose: u8, quiet: bool) -> Result<(), String> {
        let filter = if quiet {
            "error"
        } else if verbose >= 2 {
            "llmr=debug"
        } else if verbose == 1 {
            "llmr=info"
        } else {
            "warn"
        };

        let subscriber = tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| filter.into()),
            )
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(false)
                    .with_ansi(Self::stream_supports_ansi(stderr().is_terminal()))
                    .compact(),
            );

        subscriber
            .try_init()
            .map_err(|e| format!("Logger already initialized: {}", e))
    }

    fn stream_supports_ansi(is_terminal: bool) -> bool {
        if env::var_os("NO_COLOR").is_some() {
            return false;
        }

        match env::var("TERM") {
            Ok(term) if term.eq_ignore_ascii_case("dumb") => false,
            _ => is_terminal,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Style {
    ansi_enabled: bool,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            ansi_enabled: Logger::stream_supports_ansi(stdout().is_terminal()),
        }
    }
}

impl Style {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn plain() -> Self {
        Self {
            ansi_enabled: false,
        }
    }

    fn esc(&self, code: &str) -> String {
        if self.ansi_enabled {
            format!("\x1b[{}m", code)
        } else {
            String::new()
        }
    }

    fn reset(&self) -> String {
        self.esc("0")
    }

    fn color(&self, fg: u8) -> String {
        self.esc(&format!("38;5;{}", fg))
    }

    fn wrap(&self, prefix: String, text: &str) -> String {
        if self.ansi_enabled {
            format!("{}{}{}", prefix, text, self.reset())
        } else {
            text.to_string()
        }
    }

    fn bold(&self) -> String {
        self.esc("1")
    }

    fn amber(&self) -> String {
        self.color(214)
    }

    fn terracotta(&self) -> String {
        self.color(166)
    }

    fn sage(&self) -> String {
        self.color(108)
    }

    fn slate(&self) -> String {
        self.color(245)
    }

    fn cream(&self) -> String {
        self.color(229)
    }

    fn burnt(&self) -> String {
        self.color(208)
    }

    pub fn header(&self, text: &str) -> String {
        self.wrap(format!("{}{}", self.bold(), self.amber()), text)
    }

    pub fn title(&self, text: &str) -> String {
        self.wrap(format!("{}{}", self.bold(), self.cream()), text)
    }

    pub fn success(&self, text: &str) -> String {
        self.wrap(self.sage(), text)
    }

    pub fn warning(&self, text: &str) -> String {
        self.wrap(format!("{}{}", self.bold(), self.burnt()), text)
    }

    pub fn error(&self, text: &str) -> String {
        self.wrap(format!("{}{}", self.bold(), self.terracotta()), text)
    }

    pub fn info(&self, text: &str) -> String {
        self.wrap(self.amber(), text)
    }

    pub fn muted<T: AsRef<str>>(&self, text: T) -> String {
        self.wrap(self.slate(), text.as_ref())
    }

    pub fn value<T: AsRef<str>>(&self, text: T) -> String {
        text.as_ref().to_string()
    }

    pub fn accent(&self, text: &str) -> String {
        self.wrap(self.cream(), text)
    }

    pub fn gpu_nvidia(&self, text: &str) -> String {
        self.wrap(self.sage(), text)
    }

    pub fn gpu_amd(&self, text: &str) -> String {
        self.wrap(self.terracotta(), text)
    }

    pub fn gpu_intel(&self, text: &str) -> String {
        self.wrap(self.amber(), text)
    }

    pub fn gpu_apple(&self, text: &str) -> String {
        self.wrap(self.cream(), text)
    }

    pub fn gpu_other(&self, text: &str) -> String {
        self.wrap(self.slate(), text)
    }

    pub fn vram(&self, mb: u64) -> String {
        let gb = mb as f64 / 1024.0;
        let (color, label) = if gb >= 8.0 {
            (self.sage(), format!("{:.0} GB VRAM", gb))
        } else if gb >= 4.0 {
            (self.burnt(), format!("{:.0} GB VRAM", gb))
        } else {
            (self.terracotta(), format!("{} MB VRAM", mb))
        };

        self.wrap(color, &label)
    }

    pub fn check(&self) -> String {
        self.wrap(self.sage(), "✓")
    }

    pub fn cross(&self) -> String {
        self.wrap(self.terracotta(), "✗")
    }

    pub fn dash(&self) -> String {
        self.wrap(self.slate(), "–")
    }
}

pub fn gpu_style(style: &Style, name: &str) -> String {
    let name_lower = name.to_lowercase();

    if name_lower.contains("nvidia") {
        style.gpu_nvidia(name)
    } else if name_lower.contains("amd") || name_lower.contains("radeon") {
        style.gpu_amd(name)
    } else if name_lower.contains("apple")
        || name_lower.contains(" m1")
        || name_lower.contains(" m2")
        || name_lower.contains(" m3")
        || name_lower.contains(" m4")
    {
        style.gpu_apple(name)
    } else if name_lower.contains("intel") {
        style.gpu_intel(name)
    } else {
        style.gpu_other(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_default() {
        let style = Style::plain();
        assert_eq!(style.success("ok"), "ok");
    }

    #[test]
    fn test_style_new() {
        let style = Style::new();
        assert!(style.title("hello").contains("hello"));
    }

    #[test]
    fn test_gpu_style_nvidia() {
        let style = Style {
            ansi_enabled: true,
        };
        let result = gpu_style(&style, "NVIDIA RTX 3080");
        assert!(result.contains("NVIDIA RTX 3080"));
        assert!(result.contains("\x1b[38;5;108m"));
    }

    #[test]
    fn test_gpu_style_amd() {
        let style = Style {
            ansi_enabled: true,
        };
        let result = gpu_style(&style, "AMD Radeon RX 6800");
        assert!(result.contains("AMD Radeon RX 6800"));
        assert!(result.contains("\x1b[38;5;166m"));
    }

    #[test]
    fn test_gpu_style_intel() {
        let style = Style {
            ansi_enabled: true,
        };
        let result = gpu_style(&style, "Intel Iris Xe");
        assert!(result.contains("Intel Iris Xe"));
        assert!(result.contains("\x1b[38;5;214m"));
    }

    #[test]
    fn test_gpu_style_unknown() {
        let style = Style {
            ansi_enabled: true,
        };
        let result = gpu_style(&style, "Vulkan GPU");
        assert!(result.contains("Vulkan GPU"));
        assert!(result.contains("\x1b[38;5;245m"));
    }

    #[test]
    fn test_plain_style_has_no_ansi() {
        let style = Style::plain();
        let output = style.warning("Warning");
        assert_eq!(output, "Warning");
        assert!(!output.contains("\x1b["));
    }
}
