use llmr::hardware::{CpuInfo, GpuInfo, HardwareInfo, RamInfo};

#[test]
fn test_public_hardware_info_fields() {
    let info = HardwareInfo {
        cpu: CpuInfo {
            cores: 8,
            threads: 16,
            name: "Test CPU".to_string(),
            architecture: "x86_64".to_string(),
            frequency: Some(3600),
        },
        gpu: Some(GpuInfo {
            names: vec!["NVIDIA RTX 3080".to_string()],
            vram_mb: vec![10240],
            vram_free_mb: vec![8000],
            type_: "nvidia".to_string(),
        }),
        ram: RamInfo {
            total: 32_000_000_000,
            total_gb: 32,
            free_gb: 16,
        },
        has_nvlink: false,
    };

    assert_eq!(info.cpu.cores, 8);
    assert!(info.gpu.is_some());
    assert_eq!(info.ram.total_gb, 32);
}
