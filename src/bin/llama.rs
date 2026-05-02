use clap::Parser;

#[tokio::main]
async fn main() -> llmr::Result<()> {
    let global_args = llmr::cli::Args::parse();

    if let Err(e) = llmr::utils::Logger::setup(global_args.verbose, global_args.quiet) {
        eprintln!("Warning: {}", e);
    }

    let style = llmr::utils::Style::default();

    match global_args.command {
        llmr::cli::Commands::Serve(args) => {
            let command = llmr::cli::ServeCommand::new(args, style);
            command.execute().await?;
        }
        llmr::cli::Commands::Status(args) => {
            let command = llmr::cli::StatusCommand::new(args, style);
            command.execute().await?;
        }
        llmr::cli::Commands::Stop(args) => {
            let command = llmr::cli::StopCommand::new(args, style);
            command.execute().await?;
        }
        llmr::cli::Commands::Profiles(args) => {
            let command = llmr::cli::ProfilesCommand::new(args, style);
            command.execute().await?;
        }
        llmr::cli::Commands::Tune(args) => {
            let command = llmr::cli::TuneCommand::new(args, style);
            command.execute().await?;
        }
        llmr::cli::Commands::Doctor => {
            let command = llmr::cli::DoctorCommand::new(style);
            command.execute().await?;
        }
        llmr::cli::Commands::Version => {
            llmr::cli::VersionCommand::execute().await?;
        }
    }

    Ok(())
}
