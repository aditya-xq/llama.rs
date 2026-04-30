use super::args::*;
use crate::diagnostics::{print_diagnostic_results, DockerCheck, EnvCheck, HardwareCheck};
use crate::docker::DockerClient;
use crate::errors::{Error, Result};
use crate::hardware::HardwareInfo;
use crate::models::{ModelScanner, Profile, ProfileManager};
use crate::utils::Style;
use std::io::Write;
use tracing::{info, warn};

pub struct ServeCommand {
    args: ServeArgs,
    style: Style,
}

impl ServeCommand {
    pub fn new(args: ServeArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let style = &self.style;

        let model_path: String = if let Some(model) = &self.args.model {
            self.validate_model_path(model)?
        } else {
            match self.interactive_model_select().await? {
                Some(p) => p,
                None => {
                    println!(
                        "{}",
                        style.error("No model selected")
                    );
                    return Ok(());
                }
            }
        };

        info!("Starting llmr server");

        let hardware = if !self.args.skip_hardware && !self.args.no_gpu {
            crate::hardware::detect().await?
        } else {
            HardwareInfo::default()
        };

        let profile_manager = ProfileManager::new();
        let is_new_profile = !profile_manager
            .profile_exists(&model_path, &hardware)
            .await?;

        if is_new_profile {
            println!(
                "{}",
                style.info("First run for this model - applying optimizations...")
            );
            std::io::stdout().flush()?;
        }

        let base_profile = profile_manager
                .load_or_create(&model_path, &hardware)
                .await?;

        let enable_benchmark = self.args.benchmark && !self.args.no_benchmark && !self.args.dry_run;
        let auto_optimize = is_new_profile && !self.args.no_benchmark && !self.args.dry_run;

        let mut profile = if enable_benchmark || auto_optimize {
            println!(
                "  {} Running optimization benchmarks (this may take a few minutes)...",
                style.info("→")
            );
            std::io::stdout().flush()?;
            let optimized = profile_manager.benchmark(&model_path, &hardware).await?;
            let mut final_profile = optimized;
            final_profile.docker_image = base_profile.docker_image.clone();
            println!("{} Optimization complete", style.success("✓"));
            final_profile
        } else {
            base_profile
        };

        self.apply_serve_overrides(&mut profile);

        if self.args.dry_run {
            self.print_docker_command(&profile);
            return Ok(());
        }

        println!("  {} Connecting to Docker...", style.info("→"));
        std::io::stdout().flush()?;
        let docker_client = DockerClient::new()?;
        docker_client.get_info().await?;

        println!("  {} Starting container...", style.info("→"));
        std::io::stdout().flush()?;
        self.run_container(&docker_client, &model_path, &profile)
            .await?;

        println!();
        println!("  {} Server ready", style.success("✓"));
        println!(
            "    {}",
            style.accent(&format!(
                "http://localhost:{}/v1/chat/completions",
                self.args.port
            ))
        );
        if self.args.metrics {
            println!(
                "    {} Metrics at http://localhost:{}/metrics",
                style.info("→"),
                self.args.port
            );
        }
        println!(
            "    {} Health at http://localhost:{}/health",
            style.info("→"),
            self.args.port
        );
        println!(
            "  {} Stop with: {}",
            style.info("→"),
            style.muted(format!("llmr stop"))
        );

        Ok(())
    }

    fn apply_serve_overrides(&self, profile: &mut Profile) {
        if let Some(threads) = self.args.threads.filter(|threads| *threads > 0) {
            profile.threads = threads;
        }
        if let Some(ctx_size) = self.args.ctx_size.filter(|ctx_size| *ctx_size > 0) {
            profile.context_size = ctx_size;
        }
        if let Some(batch_size) = self.args.batch_size.filter(|batch_size| *batch_size > 0) {
            profile.batch_size = batch_size;
        }
        if let Some(ubatch_size) = self.args.ubatch_size.filter(|ubatch_size| *ubatch_size > 0) {
            profile.ubatch_size = ubatch_size;
        }
        if let Some(parallel) = self.args.parallel.filter(|parallel| *parallel > 0) {
            profile.parallel_slots = parallel;
        }
        if let Some(cache_type_k) = self
            .args
            .cache_type_k
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            profile.cache_type_k = cache_type_k.clone();
        }
        if let Some(cache_type_v) = self
            .args
            .cache_type_v
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            profile.cache_type_v = cache_type_v.clone();
        }

        if self.args.no_gpu {
            profile.gpu_layers = 0;
            profile.split_mode = "none".to_string();
            return;
        }

        if let Some(gpu_layers) = self.args.gpu_layers {
            profile.gpu_layers = i32::try_from(gpu_layers).unwrap_or(i32::MAX);
        }
        if let Some(split_mode) = self.args.split_mode {
            profile.split_mode = split_mode.as_str().to_string();
        }
    }

    fn validate_model_path(&self, model_path: &str) -> Result<String> {
        let path = std::path::Path::new(model_path);
        if !path.exists() {
            return Err(Error::ModelNotFound {
                path: model_path.to_string(),
            })?;
        }
        if !path.is_file() {
            return Err(Error::InvalidModelPath {
                path: model_path.to_string(),
            })?;
        }

        Ok(model_path.to_string())
    }

    async fn interactive_model_select(&self) -> Result<Option<String>> {
        let style = &self.style;
        let scanner = ModelScanner::new();
        let profile_manager = ProfileManager::new();

        println!();
        println!("  {}", style.title("Searching for GGUF models..."));
        println!();

        let cached_folders = profile_manager.get_cached_model_folders();
        let quick_scan_results = if !cached_folders.is_empty() {
            println!("  {} Checking cached locations...", style.info("→"));
            scanner.scan_paths(&cached_folders)
        } else {
            Vec::new()
        };

        let all = if !quick_scan_results.is_empty() {
            quick_scan_results
        } else {
            println!("  {} Scanning disks for GGUF files...", style.info("→"));
            scanner.scan_disks()
        };

        let model_paths: Vec<String> = all.iter().map(|m| m.path.to_string_lossy().to_string()).collect();
        if !model_paths.is_empty() {
            let _ = profile_manager.save_model_cache(&model_paths).await;
        }

        if all.is_empty() {
            println!(
                "  {} No GGUF models found on local disks",
                style.warning("!")
            );
            println!();
            println!("  {} Common model locations checked:", style.info("→"));
            for root in ModelScanner::find_root_paths() {
                let root_str = root.to_string_lossy();
                println!("    {}", style.muted(root_str.as_ref()));
            }
            println!();
            print!("  {} Enter model path manually: ", style.info("→"));
            std::io::stdout().flush()?;
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let path = input.trim().to_string();
            if path.is_empty() {
                return Ok(None);
            }
            return Ok(Some(path));
        }

        let all: Vec<_> = all;

        println!(
            "  {} Found {} model(s)",
            style.success("✓"),
            style.accent(&all.len().to_string())
        );
        println!();

        for (i, model) in all.iter().take(20).enumerate() {
            println!(
                "  {}.  {}  {}",
                i + 1,
                style.accent(&model.name),
                style.muted(&model.size_formatted)
            );
        }

        if all.len() > 20 {
            println!(
                "  {} ... and {} more",
                style.muted("→"),
                style.muted((all.len() - 20).to_string())
            );
        }

        println!();
        print!("  {} Select model (1-{}): ", style.info("→"), all.len());
        std::io::stdout().flush()?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;

        let choice: usize = input.trim().parse().unwrap_or(0);
        if choice == 0 || choice > all.len() {
            println!("{}", style.error("Invalid selection"));
            return Ok(None);
        }

        let selected = &all[choice - 1];
        Ok(Some(selected.path.to_string_lossy().to_string()))
    }

    fn print_docker_command(&self, profile: &crate::models::Profile) {
        let style = &self.style;
        println!();
        println!("  {}", style.title("Docker Command"));
        println!();
        println!("docker run \\");
        for arg in profile.to_docker_args(self.args.port, self.args.metrics, self.args.public) {
            if arg.contains(' ') {
                println!("  '{}' \\", arg);
            } else {
                println!("  {} \\", arg);
            }
        }
    }

    async fn run_container(
        &self,
        docker_client: &DockerClient,
        model_path: &str,
        profile: &crate::models::Profile,
    ) -> Result<()> {
        let container_name = profile.container_name();
        let style = &self.style;

        if let Some(_existing) = docker_client.get_container(&container_name).await? {
            println!("  {} Removing existing container", style.info("→"));
            docker_client.remove_container(&container_name).await?;
        }

        if !docker_client.image_exists(&profile.docker_image).await {
            println!(
                "  {} Pulling Docker image ({})...",
                style.info("→"),
                profile.docker_image
            );
            std::io::stdout().flush()?;
            docker_client.pull_image(&profile.docker_image).await?;
            println!("  {} Image pulled", style.success("✓"));
        }

        println!("  {} Starting llama.cpp server...", style.info("→"));
        std::io::stdout().flush()?;
        docker_client
            .run_container(
                &container_name,
                &profile.docker_image,
                model_path,
                self.args.port,
                profile,
                self.args.metrics,
                self.args.public,
                self.args.debug,
            )
            .await?;

        println!("  {} Waiting for server to be ready...", style.info("→"));
        std::io::stdout().flush()?;
        docker_client
            .wait_for_health(&container_name, self.args.port)
            .await?;

        Ok(())
    }
}

pub struct StatusCommand {
    args: StatusArgs,
    style: Style,
}

impl StatusCommand {
    pub fn new(args: StatusArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let docker_client = match DockerClient::new() {
            Ok(client) => client,
            Err(err) => {
                println!(
                    "{}",
                    self.style.warning(&format!("Docker unavailable: {err}"))
                );
                return Ok(());
            }
        };

        if let Some(name) = &self.args.name {
            self.show_container(&docker_client, name).await?;
        } else {
            self.show_all_containers(&docker_client).await?;
        }

        Ok(())
    }

    async fn show_container(&self, docker_client: &DockerClient, name: &str) -> Result<()> {
        let style = &self.style;

        println!();
        println!("  {}", style.title("Container Status"));
        println!();

        let container = match docker_client.get_container(name).await {
            Ok(container) => container,
            Err(Error::DockerError { message }) => {
                println!(
                    "{}",
                    style.warning(&format!("Docker unavailable: {message}"))
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };

        if let Some(container) = container {
            println!("  {} {}", style.info("→"), style.accent("Name"));
            println!("    {}", container.name);

            println!("  {} {}", style.info("→"), style.accent("Image"));
            println!("    {}", container.image);

            println!("  {} {}", style.info("→"), style.accent("Status"));
            let status_lower = container.status.to_lowercase();
            if status_lower.contains("running") {
                println!(
                    "    {} {}",
                    style.success("running"),
                    style.muted(&container.status)
                );
            } else {
                println!(
                    "    {} {}",
                    style.warning("!"),
                    style.muted(&container.status)
                );
            }

            if !container.ports.is_empty() {
                println!("  {} {}", style.info("→"), style.accent("Ports"));
                for (port, host_ip) in &container.ports {
                    println!(
                        "    {}:{}",
                        style.accent(&port.to_string()),
                        style.muted(host_ip)
                    );
                }
            }

            println!("  {} {}", style.info("→"), style.accent("Started"));
            println!("    {}", container.created_at);

            println!();
            println!(
                "  {} Use `llmr stop` to stop this container",
                style.success("✓")
            );
        } else {
            println!(
                "  {} Container '{}' not found (not running)",
                style.warning("!"),
                name
            );
            println!();
            println!(
                "  {} Use `llmr serve` to start a container",
                style.info("→")
            );
        }

        Ok(())
    }

    async fn show_all_containers(&self, docker_client: &DockerClient) -> Result<()> {
        let style = &self.style;

        println!();
        println!("  {}", style.title("Llama.cpp Containers"));
        println!();

        let containers = match docker_client.list_containers_by_prefix("llama_").await {
            Ok(containers) => containers,
            Err(Error::DockerError { message }) => {
                println!(
                    "{}",
                    style.warning(&format!("Docker unavailable: {message}"))
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };

        if containers.is_empty() {
            println!("  {} No llama.cpp containers are running at the moment", style.dash());
            println!();
            println!("  {}", style.muted("Run `llmr serve` to start a container"));
            return Ok(());
        }

        println!("  {} {}", style.info("→"), style.accent("Running"));
        for container in containers {
            let status_lower = container.status.to_lowercase();
            let status_icon = if status_lower.contains("running") {
                style.success("●")
            } else {
                style.warning("●")
            };
            println!("    {} {}", status_icon, container.name);
            println!(
                "      {} · {}",
                container.image,
                style.muted(&container.status)
            );
        }

        println!();
        println!(
            "  {} Run `llmr status -n <name>` for details",
            style.info("→")
        );

        Ok(())
    }
}

pub struct StopCommand {
    args: StopArgs,
    style: Style,
}

impl StopCommand {
    pub fn new(args: StopArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let docker_client = match DockerClient::new() {
            Ok(client) => client,
            Err(err) => {
                println!(
                    "{}",
                    self.style.warning(&format!("Docker unavailable: {err}"))
                );
                return Ok(());
            }
        };

        if let Some(name) = &self.args.name {
            self.stop_container(&docker_client, name).await?;
        } else {
            self.stop_all_containers(&docker_client).await?;
        }

        Ok(())
    }

    async fn stop_container(&self, docker_client: &DockerClient, name: &str) -> Result<()> {
        let style = &self.style;
        let container = match docker_client.get_container(name).await {
            Ok(container) => container,
            Err(Error::DockerError { message }) => {
                println!(
                    "{}",
                    style.warning(&format!("Docker unavailable: {message}"))
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };

        if container.is_some() {
            docker_client.remove_container(name).await?;
            println!(
                "{}",
                style.success(&format!("Container '{}' stopped and removed", name))
            );
        } else {
            warn!("Container '{}' not found", name);
        }

        Ok(())
    }

    async fn stop_all_containers(&self, docker_client: &DockerClient) -> Result<()> {
        let style = &self.style;
        let containers = match docker_client.list_containers_by_prefix("llama_").await {
            Ok(containers) => containers,
            Err(Error::DockerError { message }) => {
                println!(
                    "{}",
                    style.warning(&format!("Docker unavailable: {message}"))
                );
                return Ok(());
            }
            Err(err) => return Err(err),
        };

        if containers.is_empty() {
            println!("{}", style.muted("No running llama_* containers found."));
            return Ok(());
        }

        for container in containers {
            docker_client.remove_container(&container.name).await?;
            println!(
                "{}",
                style.success(&format!(
                    "Container '{}' stopped and removed",
                    container.name
                ))
            );
        }

        Ok(())
    }
}

pub struct ProfilesCommand {
    args: ProfilesArgs,
    style: Style,
}

impl ProfilesCommand {
    pub fn new(args: ProfilesArgs, style: Style) -> Self {
        Self { args, style }
    }

    pub async fn execute(&self) -> Result<()> {
        let profile_manager = ProfileManager::new();

        match &self.args.subcommand {
            Some(ProfilesSubcommand::List) => {
                self.list_profiles(&profile_manager).await?;
            }
            Some(ProfilesSubcommand::Delete { key }) => {
                self.delete_profile(&profile_manager, key).await?;
            }
            Some(ProfilesSubcommand::Clear) => {
                self.clear_profiles(&profile_manager).await?;
            }
            Some(ProfilesSubcommand::Show { key }) => {
                self.show_profile(&profile_manager, key).await?;
            }
            None => {
                self.list_profiles(&profile_manager).await?;
            }
        }

        Ok(())
    }

    async fn list_profiles(&self, profile_manager: &ProfileManager) -> Result<()> {
        let style = &self.style;
        let profiles = profile_manager.list_all().await?;

        if profiles.is_empty() {
            println!("{}", style.muted("No saved profiles."));
            return Ok(());
        }

        println!(
            "Saved profiles ({}):",
            style.accent(&profiles.len().to_string())
        );
        for (key, profile) in profiles {
            println!("\n  Profile: {}", style.accent(&key));
            println!("    Docker Image: {}", profile.docker_image);
            println!(
                "    Threads: {} | Batch: {} | GPU Layers: {}",
                profile.threads, profile.batch_size, profile.gpu_layers
            );
            println!(
                "    Split Mode: {} | Context: {} | Cache K/V: {}/{}",
                profile.split_mode,
                profile.context_size,
                profile.cache_type_k,
                profile.cache_type_v
            );
            println!(
                "    GPU Type: {} | GPU Count: {}",
                profile.gpu_type, profile.gpu_count
            );
            if let Some(tps) = profile.best_tps {
                println!("    Best TPS: {:.2}", tps);
            }
        }

        Ok(())
    }

    async fn delete_profile(&self, profile_manager: &ProfileManager, key: &str) -> Result<()> {
        let style = &self.style;
        profile_manager.delete(key).await?;
        println!("{}", style.success(&format!("Profile '{}' deleted", key)));
        Ok(())
    }

    async fn clear_profiles(&self, profile_manager: &ProfileManager) -> Result<()> {
        let style = &self.style;
        println!(
            "{}",
            style.warning("This will delete ALL saved profiles in the config directory.")
        );
        print!("Are you sure? [y/N] ");
        std::io::Write::flush(&mut std::io::stdout())?;

        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let confirm = input.trim().to_lowercase();

        if confirm == "y" || confirm == "yes" {
            profile_manager.clear_all().await?;
            println!("{}", style.success("All profiles cleared."));
        } else {
            println!("{}", style.muted("Aborted."));
        }

        Ok(())
    }

    async fn show_profile(&self, profile_manager: &ProfileManager, key: &str) -> Result<()> {
        let style = &self.style;
        if let Some(profile) = profile_manager.load(key).await? {
            println!("Profile: {}", style.accent(key));
            println!("Model: {}", profile.model_file);
            println!("Docker Image: {}", profile.docker_image);
            println!("Threads: {}", profile.threads);
            println!("GPU Layers: {}", profile.gpu_layers);
            println!("Context Size: {}", profile.context_size);
            println!("Batch Size: {}", profile.batch_size);
            println!("Ubatch Size: {}", profile.ubatch_size);
            println!("Split Mode: {}", profile.split_mode);
            println!("GPU Type: {}", profile.gpu_type);
            println!("Cache Type K: {}", profile.cache_type_k);
            println!("Cache Type V: {}", profile.cache_type_v);
            println!("Parallel Slots: {}", profile.parallel_slots);
            println!("Created: {}", profile.created_at);
            if let Some(tps) = profile.best_tps {
                println!("Best TPS: {:.2}", tps);
            }
        } else {
            println!("{}", style.warning(&format!("Profile '{}' not found", key)));
        }

        Ok(())
    }
}

pub struct DoctorCommand {
    style: Style,
}

impl DoctorCommand {
    pub fn new(style: Style) -> Self {
        Self { style }
    }

    pub async fn execute(&self) -> Result<()> {
        let style = &self.style;

        println!();
        println!("  {}", style.title("llmr Doctor"));
        println!();

        let env_check = EnvCheck::new();

        let (docker_result, hardware_result) = tokio::join!(
            async {
                let check = DockerCheck::new();
                check.check().await
            },
            async {
                let check = HardwareCheck::new();
                check.check().await
            }
        );

        env_check.run(style);

        print_diagnostic_results(style, &docker_result, &hardware_result);

        println!();
        println!("  {} Diagnostics complete", style.success("✓"));

        Ok(())
    }
}

pub struct VersionCommand;

impl VersionCommand {
    pub async fn execute() -> Result<()> {
        println!("llmr {}", env!("CARGO_PKG_VERSION"));
        println!("A tiny CLI for running optimised llama.cpp in Docker");
        Ok(())
    }
}
