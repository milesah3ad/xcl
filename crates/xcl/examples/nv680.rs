//! Local probe for #680 (not for merging): does NVIDIA's DXIL reader abort the process
//! ("Never resolved value found in function" / "Invalid record") on synthetic compute programs
//! in the heap-window form? Program generator adapted from QemmHD's `pso156.rs`
//! (https://github.com/QemmHD/xcl/blob/exp/156-nvidia-pso/crates/xcl/examples/pso156.rs, GPL-2.0).
//!
//! ```text
//! nv680 gen --out DIR --name NAME --rounds N --heap 0|1 --shape base|atomic|atomic64|loop|wave [--seeds A..B] [--dxc-arg X]
//! nv680 create FILE.dxil        one compute pipeline with the ABI root signature; prints OK / ERR
//! ```

use std::path::PathBuf;
use std::time::Instant;

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{dxc, translate, CsInterface, Mode, Options};

const V: u64 = 256;

fn e(name: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    encode(name, enc, f, lit).unwrap_or_else(|x| panic!("{name}: {x}"))
}

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 | 1
    }
}

/// v[6:7] = s0 + tid * 4 : s1 (the window is not known at compile time), then `rounds` rounds of
/// LCG steps, a load + store each round, a divergent if every 4th, plus the shape's extra.
fn program(rounds: u32, shape: &str, seed: u32) -> Vec<u8> {
    let mut v = Vec::new();
    let mut rng = Lcg(seed.wrapping_mul(2_654_435_761) ^ 0x9E37_79B9);
    v.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None));
    v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 0), ("VSRC1", 6)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 1)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 255)], Some(seed)));
    let glbl = |op: &str, f: &[(&str, u64)]| e(op, "ENC_FLAT_GLBL", f, None);
    for r in 0..rounds {
        let off = |k: u32| ((r * 4 + k * 4) % 2048) as u64;
        v.extend(e("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 2), ("SRC1", 255)], Some(rng.next())));
        v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 2)], Some(rng.next())));
        v.extend(glbl("GLOBAL_LOAD_DWORD", &[("VDST", 4), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", off(0))]));
        v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 4), ("VSRC1", 3)], None));
        v.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", off(1))]));
        match shape {
            "atomic" => {
                // returning atomic add: GLC=1 gives the pre-op value back
                v.extend(glbl("GLOBAL_ATOMIC_ADD", &[("VDST", 8), ("ADDR", 6), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", off(3)), ("GLC", 1)]));
                v.extend(e("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 8), ("VSRC1", 3)], None));
            }
            "atomic64" => {
                // returning 64-bit atomics on the window: umax (v[2:3]) and a compare-swap (v[12:15])
                v.extend(glbl("GLOBAL_ATOMIC_UMAX_X2", &[("VDST", 10), ("ADDR", 6), ("DATA", 2), ("SADDR", 0x7D), ("OFFSET", off(3) & !7), ("GLC", 1)]));
                v.extend(e("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 10), ("VSRC1", 3)], None));
                if r % 2 == 0 {
                    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 12), ("SRC0", V + 3)], None));
                    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 13), ("SRC0", V + 2)], None));
                    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 14), ("SRC0", V + 10)], None));
                    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 15), ("SRC0", V + 11)], None));
                    v.extend(glbl("GLOBAL_ATOMIC_CMPSWAP_X2", &[("VDST", 10), ("ADDR", 6), ("DATA", 12), ("SADDR", 0x7D), ("OFFSET", off(5) & !7), ("GLC", 1)]));
                    v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 11), ("VSRC1", 3)], None));
                }
            }
            "wave" => {
                // a cross-lane read: v_readlane into s30, used by the next VALU op
                v.extend(e("V_READLANE_B32", "ENC_VOP3", &[("VDST", 30), ("SRC0", V + 3), ("SRC1", 128 + (r % 16) as u64)], None));
                v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 30), ("VSRC1", 3)], None));
            }
            _ => {}
        }
        if r % 4 == 3 {
            let mut body = e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 3)], Some(rng.next()));
            body.extend(glbl("GLOBAL_LOAD_DWORD", &[("VDST", 8), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", off(2))]));
            body.extend(e("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 8), ("VSRC1", 3)], None));
            v.extend(e("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(rng.next())));
            v.extend(e("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)], None));
            v.extend(e("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", (body.len() / 4) as u64)], None));
            v.extend(body);
            v.extend(e("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)], None));
        }
        if shape == "loop" && r % 8 == 7 {
            // a uniform 4-trip loop around a load/add: s40 counts down, backward branch
            v.extend(e("S_MOV_B32", "ENC_SOP1", &[("SDST", 40), ("SSRC0", 128 + 4)], None));
            let body_start = v.len();
            v.extend(glbl("GLOBAL_LOAD_DWORD", &[("VDST", 9), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", off(4))]));
            v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 9), ("VSRC1", 3)], None));
            v.extend(e("S_SUB_U32", "ENC_SOP2", &[("SDST", 40), ("SSRC0", 40), ("SSRC1", 129)], None));
            v.extend(e("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 40), ("SSRC1", 128)], None));
            let back = -(((v.len() - body_start) / 4 + 1) as i64);
            v.extend(e("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", (back as i16 as u16) as u64)], None));
        }
    }
    v.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", 0)]));
    v.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}

fn arg<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn gen(args: &[String]) {
    let out = PathBuf::from(arg(args, "--out").expect("--out"));
    std::fs::create_dir_all(&out).unwrap();
    let name = arg(args, "--name").expect("--name");
    let rounds: u32 = arg(args, "--rounds").unwrap_or("500").parse().unwrap();
    let heap = arg(args, "--heap").unwrap_or("1") == "1";
    let shape = arg(args, "--shape").unwrap_or("base");
    let (a, b) = arg(args, "--seeds").unwrap_or("1..2").split_once("..").map(|(a, b)| (a.parse::<u32>().unwrap(), b.parse::<u32>().unwrap())).unwrap();
    let extra: Vec<&str> = args.windows(2).filter(|w| w[0] == "--dxc-arg").map(|w| w[1].as_str()).collect();
    for seed in a..b {
        let code = program(rounds, shape, seed);
        let cs = CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 };
        let opt = Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, mem_windows: 4, heap_windows: heap, ..Default::default() };
        let t = translate(&code, &opt).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let file = out.join(format!("{name}_s{seed}.dxil"));
        std::fs::write(file.with_extension("hlsl"), &t.hlsl).unwrap();
        let t0 = Instant::now();
        match dxc::compile(&t.hlsl, "main", "cs_6_6", &extra) {
            Ok(d) => {
                std::fs::write(&file, &d).unwrap();
                println!("{}\t{} KB hlsl\t{} KB dxil\t{:.1} s dxc", file.display(), t.hlsl.len() / 1024, d.len() / 1024, t0.elapsed().as_secs_f64());
            }
            Err(x) => eprintln!("{name} s{seed}: dxc: {}", &x[..x.len().min(800)]),
        }
    }
}

fn create(file: &str) {
    let dxil = std::fs::read(file).unwrap();
    unsafe {
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).expect("device");
        let dev = dev.unwrap();
        if std::env::var_os("NV680_FEATURES").is_some() {
            let mut o9 = D3D12_FEATURE_DATA_D3D12_OPTIONS9::default();
            let mut o11 = D3D12_FEATURE_DATA_D3D12_OPTIONS11::default();
            let _ = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &mut o9 as *mut _ as *mut _, std::mem::size_of_val(&o9) as u32);
            let _ = dev.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &mut o11 as *mut _ as *mut _, std::mem::size_of_val(&o11) as u32);
            println!("AtomicInt64OnTypedResource={} AtomicInt64OnGroupShared={} AtomicInt64OnDescriptorHeapResource={}",
                o9.AtomicInt64OnTypedResourceSupported.as_bool(), o9.AtomicInt64OnGroupSharedSupported.as_bool(), o11.AtomicInt64OnDescriptorHeapResourceSupported.as_bool());
        }
        let rs = xcl::gpu::recomp::pso::abi_root_signature(&dev, false).unwrap();
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::ManuallyDrop::new(Some(rs)),
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const _, BytecodeLength: dxil.len() },
            ..Default::default()
        };
        let t0 = Instant::now();
        let r: windows::core::Result<ID3D12PipelineState> = dev.CreateComputePipelineState(&desc);
        match r {
            Ok(_) => println!("OK\t{:.2} s\t{} KB", t0.elapsed().as_secs_f64(), dxil.len() / 1024),
            Err(x) => println!("ERR {:#x}\t{:.2} s\t{} KB", x.code().0, t0.elapsed().as_secs_f64(), dxil.len() / 1024),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("gen") => gen(&args[1..]),
        Some("create") => create(&args[1]),
        _ => eprintln!("nv680 gen ... | nv680 create FILE"),
    }
}
