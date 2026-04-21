use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Docker error: {message}")]
    DockerError { message: String },

    #[error("Docker CLI not found")]
    DockerCliNotFound,

    #[error("Missing required argument: {arg}")]
    MissingArgument { arg: String },

    #[error("Model not found: {path}")]
    ModelNotFound { path: String },

    #[error("Invalid model path: {path}")]
    InvalidModelPath { path: String },

    #[error("Invalid profile: {reason}")]
    InvalidProfile { reason: String },

    #[error("Timeout: {message}")]
    Timeout { message: String },

    #[error("IO error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    #[error("TOML deserialization error: {source}")]
    TomlDeError {
        #[from]
        source: toml::de::Error,
    },

    #[error("TOML serialization error: {source}")]
    TomlSerError {
        #[from]
        source: toml::ser::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_user_facing_error_messages() {
        let missing_argument = Error::MissingArgument {
            arg: "--model".to_string(),
        };
        let timeout = Error::Timeout {
            message: "container health check timed out".to_string(),
        };
        let docker_error = Error::DockerError {
            message: "connection refused".to_string(),
        };

        assert_eq!(
            missing_argument.to_string(),
            "Missing required argument: --model"
        );
        assert_eq!(
            timeout.to_string(),
            "Timeout: container health check timed out"
        );
        assert_eq!(docker_error.to_string(), "Docker error: connection refused");
    }

    #[test]
    fn test_io_error_conversion_preserves_source_message() {
        let err: Error = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found").into();
        assert!(matches!(err, Error::IoError { .. }));
        assert!(err.to_string().contains("file not found"));
    }

    #[test]
    fn test_toml_deserialization_error_conversion() {
        let parse_err = toml::from_str::<toml::Value>("not = [valid").unwrap_err();
        let err: Error = parse_err.into();

        assert!(matches!(err, Error::TomlDeError { .. }));
    }

    #[test]
    fn test_toml_serialization_error_conversion() {
        let invalid_float = toml::Value::Float(f64::NAN);
        let err: Error = toml::to_string(&invalid_float).unwrap_err().into();

        assert!(matches!(err, Error::TomlSerError { .. }));
    }
}
