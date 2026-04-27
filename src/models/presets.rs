use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPresetConfig {
    pub threads: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub gpu_layers: i32,
    pub split_mode: String,
    pub context_size: u32,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub parallel_slots: u32,
    pub notes: String,
}

impl ModelPresetConfig {
    pub fn is_default(&self) -> bool {
        self.threads == 0 && self.batch_size == 0
    }

    pub fn apply_to_profile(&self, profile: &mut crate::models::Profile) {
        if self.threads > 0 {
            profile.threads = self.threads;
        }
        if self.batch_size > 0 {
            profile.batch_size = self.batch_size;
        }
        if self.ubatch_size > 0 {
            profile.ubatch_size = self.ubatch_size;
        }
        if self.gpu_layers > -1 {
            profile.gpu_layers = self.gpu_layers;
        }
        if !self.split_mode.is_empty() {
            profile.split_mode = self.split_mode.clone();
        }
        if self.context_size > 0 {
            profile.context_size = self.context_size;
        }
        if !self.cache_type_k.is_empty() {
            profile.cache_type_k = self.cache_type_k.clone();
        }
        if !self.cache_type_v.is_empty() {
            profile.cache_type_v = self.cache_type_v.clone();
        }
        if self.parallel_slots > 0 {
            profile.parallel_slots = self.parallel_slots;
        }
    }
}

pub struct ModelPresets;

impl ModelPresets {
    pub fn find_preset(model_name: &str) -> Option<ModelPresetConfig> {
        let lower = model_name.to_lowercase();

        if lower.contains("qwen3") || lower.contains("qwen-3") {
            return Some(ModelPresetConfig {
                threads: 6,
                batch_size: 2048,
                ubatch_size: 512,
                gpu_layers: 999,
                split_mode: "none".to_string(),
                context_size: 8192,
                cache_type_k: "q8_0".to_string(),
                cache_type_v: "q8_0".to_string(),
                parallel_slots: 2,
                notes: "Qwen3 models".to_string(),
            });
        }

        None
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;
    use crate::HardwareInfo;

    #[test]
    fn test_model_preset_config_default() {
        let preset = ModelPresetConfig {
            threads: 0,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            split_mode: String::new(),
            context_size: 0,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            parallel_slots: 0,
            notes: String::new(),
        };

        assert!(preset.is_default());
    }

    #[test]
    fn test_model_preset_config_not_default() {
        let preset = ModelPresetConfig {
            threads: 8,
            batch_size: 2048,
            ubatch_size: 512,
            gpu_layers: 35,
            split_mode: "layer".to_string(),
            context_size: 8192,
            cache_type_k: "q8_0".to_string(),
            cache_type_v: "q8_0".to_string(),
            parallel_slots: 1,
            notes: String::new(),
        };

        assert!(!preset.is_default());
    }

    #[test]
    fn test_model_preset_apply_threads() {
        let mut profile = crate::models::Profile::new(
            "test.gguf".to_string(),
            4_000_000_000,
            &HardwareInfo::default(),
        );

        let preset = ModelPresetConfig {
            threads: 8,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            split_mode: String::new(),
            context_size: 0,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            parallel_slots: 0,
            notes: String::new(),
        };

        preset.apply_to_profile(&mut profile);
        assert_eq!(profile.threads, 8);
    }

    #[test]
    fn test_model_preset_apply_batch_size() {
        let mut profile = crate::models::Profile::new(
            "test.gguf".to_string(),
            4_000_000_000,
            &HardwareInfo::default(),
        );

        let preset = ModelPresetConfig {
            threads: 0,
            batch_size: 2048,
            ubatch_size: 0,
            gpu_layers: 0,
            split_mode: String::new(),
            context_size: 0,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            parallel_slots: 0,
            notes: String::new(),
        };

        preset.apply_to_profile(&mut profile);
        assert_eq!(profile.batch_size, 2048);
    }

    #[test]
    fn test_model_preset_apply_gpu_layers() {
        let mut profile = crate::models::Profile::new(
            "test.gguf".to_string(),
            4_000_000_000,
            &HardwareInfo::default(),
        );

        let preset = ModelPresetConfig {
            threads: 0,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 35,
            split_mode: String::new(),
            context_size: 0,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            parallel_slots: 0,
            notes: String::new(),
        };

        preset.apply_to_profile(&mut profile);
        assert_eq!(profile.gpu_layers, 35);
    }

    #[test]
    fn test_model_preset_apply_gpu_layers_negative() {
        let mut profile = crate::models::Profile::new(
            "test.gguf".to_string(),
            4_000_000_000,
            &HardwareInfo::default(),
        );
        profile.gpu_layers = 999;

        let preset = ModelPresetConfig {
            threads: 0,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: -1,
            split_mode: String::new(),
            context_size: 0,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            parallel_slots: 0,
            notes: String::new(),
        };

        preset.apply_to_profile(&mut profile);
        // gpu_layers > -1 check fails, so original value stays
        assert_eq!(profile.gpu_layers, 999);
    }

    #[test]
    fn test_model_preset_apply_split_mode() {
        let mut profile = crate::models::Profile::new(
            "test.gguf".to_string(),
            4_000_000_000,
            &HardwareInfo::default(),
        );

        let preset = ModelPresetConfig {
            threads: 0,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            split_mode: "layer".to_string(),
            context_size: 0,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            parallel_slots: 0,
            notes: String::new(),
        };

        preset.apply_to_profile(&mut profile);
        assert_eq!(profile.split_mode, "layer");
    }

    #[test]
    fn test_model_preset_apply_context_size() {
        let mut profile = crate::models::Profile::new(
            "test.gguf".to_string(),
            4_000_000_000,
            &HardwareInfo::default(),
        );

        let preset = ModelPresetConfig {
            threads: 0,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            split_mode: String::new(),
            context_size: 16384,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            parallel_slots: 0,
            notes: String::new(),
        };

        preset.apply_to_profile(&mut profile);
        assert_eq!(profile.context_size, 16384);
    }

    #[test]
    fn test_model_preset_apply_cache_types() {
        let mut profile = crate::models::Profile::new(
            "test.gguf".to_string(),
            4_000_000_000,
            &HardwareInfo::default(),
        );

        let preset = ModelPresetConfig {
            threads: 0,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            split_mode: String::new(),
            context_size: 0,
            cache_type_k: "q4_kv".to_string(),
            cache_type_v: "q4_kv".to_string(),
            parallel_slots: 0,
            notes: String::new(),
        };

        preset.apply_to_profile(&mut profile);
        assert_eq!(profile.cache_type_k, "q4_kv");
        assert_eq!(profile.cache_type_v, "q4_kv");
    }

    #[test]
    fn test_model_preset_apply_parallel_slots() {
        let mut profile = crate::models::Profile::new(
            "test.gguf".to_string(),
            4_000_000_000,
            &HardwareInfo::default(),
        );

        let preset = ModelPresetConfig {
            threads: 0,
            batch_size: 0,
            ubatch_size: 0,
            gpu_layers: 0,
            split_mode: String::new(),
            context_size: 0,
            cache_type_k: String::new(),
            cache_type_v: String::new(),
            parallel_slots: 4,
            notes: String::new(),
        };

        preset.apply_to_profile(&mut profile);
        assert_eq!(profile.parallel_slots, 4);
    }

    #[test]
    fn test_find_preset_qwen_3() {
        let preset = ModelPresets::find_preset("Qwen3-8B.gguf");
        assert!(preset.is_some());
        let preset = preset.unwrap();
        assert_eq!(preset.threads, 6);
    }

    #[test]
    fn test_find_preset_unknown() {
        let preset = ModelPresets::find_preset("unknown-model.gguf");
        assert!(preset.is_none());
    }

    #[test]
    fn test_find_preset_case_insensitive() {
        assert!(ModelPresets::find_preset("QWEN3-8B.gguf").is_some());
        assert!(ModelPresets::find_preset("qwen3-8b.gguf").is_some());
    }

    #[test]
    fn test_find_preset_preserves_notes() {
        let preset = ModelPresets::find_preset("Qwen3-8B.gguf").unwrap();
        assert!(!preset.notes.is_empty());
    }
}
