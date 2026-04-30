use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(name = "llmr")]
#[command(about = "A tiny CLI for running optimised inference via llama.cpp in Docker", long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub command: Commands,

    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[arg(short, long, global = true)]
    pub quiet: bool,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    Serve(ServeArgs),
    Status(StatusArgs),
    Stop(StopArgs),
    Profiles(ProfilesArgs),
    Doctor,
    Version,
}

#[derive(Parser, Debug)]
pub struct ServeArgs {
    #[arg(short = 'm', long)]
    pub model: Option<String>,

    #[arg(short = 'p', long, default_value_t = 8080)]
    pub port: u16,

    #[arg(long)]
    pub metrics: bool,

    #[arg(long)]
    pub benchmark: bool,

    #[arg(long)]
    pub no_benchmark: bool,

    #[arg(long)]
    pub skip_hardware: bool,

    #[arg(long)]
    pub dry_run: bool,

    #[arg(long)]
    pub public: bool,

    #[arg(long)]
    pub no_gpu: bool,

    #[arg(short, long)]
    pub auto: bool,

    #[arg(short = 't', long)]
    pub threads: Option<u32>,

    #[arg(short = 'c', long)]
    pub ctx_size: Option<u32>,

    #[arg(short = 'g', long)]
    pub gpu_layers: Option<u32>,

    #[arg(long)]
    pub split_mode: Option<SplitMode>,

    #[arg(short = 'b', long)]
    pub batch_size: Option<u32>,

    #[arg(short = 'u', long)]
    pub ubatch_size: Option<u32>,

    #[arg(long)]
    pub cache_type_k: Option<String>,

    #[arg(long)]
    pub cache_type_v: Option<String>,

    #[arg(long)]
    pub parallel: Option<u32>,

    #[arg(short, long)]
    pub debug: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum SplitMode {
    Layer,
    Row,
    None,
    Auto,
}

impl SplitMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Layer => "layer",
            Self::Row => "row",
            Self::None => "none",
            Self::Auto => "auto",
        }
    }
}

#[derive(Parser, Debug)]
pub struct StatusArgs {
    #[arg(short, long)]
    pub name: Option<String>,
}

#[derive(Parser, Debug)]
pub struct StopArgs {
    #[arg(short, long)]
    pub name: Option<String>,

    #[arg(long)]
    pub force: bool,
}

#[derive(Parser, Debug)]
pub struct ProfilesArgs {
    #[command(subcommand)]
    pub subcommand: Option<ProfilesSubcommand>,

    #[arg(long)]
    pub file: Option<String>,
}

#[derive(Subcommand, Debug)]
pub enum ProfilesSubcommand {
    List,
    Delete { key: String },
    Clear,
    Show { key: String },
}
