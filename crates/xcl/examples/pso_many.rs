//! Local probe for #1333 (not for merging): how many live pipeline states does the driver take
//! before it fails, and what does each one cost? Distinct synthetic compute programs (XCL's
//! recompiler, runtime options: 4 memory windows, split wave64), compiled on `--jobs` DXC threads
//! and created on one thread; every pipeline is kept alive. Every `--every` pipelines it prints
//! the count, rate, GPU local / non-local memory in use (DXGI), process private bytes and commit.
//!
//! ```text
//! pso_many --count 100000 [--rounds 40] [--jobs 15] [--every 1000] [--seed0 N] [--stop-free-gb 8] [--release-at N] [--dupes N] [--debug]
//! ```

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{dxc, translate, CsInterface, Mode, Options};

const V: u64 = 256;

fn e(name: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    encode(name, enc, f, lit).unwrap_or_else(|x| panic!("{name}: {x}"))
}

/// The `base` shape of nv680: LCG rounds with fresh literals, a load + store each round and a
/// divergent if every 4th. The seed's literals make every program distinct.
fn program(rounds: u32, seed: u32) -> Vec<u8> {
    let mut v = Vec::new();
    let mut s = seed.wrapping_mul(2_654_435_761) ^ 0x9E37_79B9;
    let mut next = || {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        s | 1
    };
    let glbl = |op: &str, f: &[(&str, u64)]| e(op, "ENC_FLAT_GLBL", f, None);
    v.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None));
    v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 0), ("VSRC1", 6)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 1)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 255)], Some(seed)));
    for r in 0..rounds {
        let off = |k: u32| ((r * 4 + k * 4) % 2048) as u64;
        v.extend(e("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 2), ("SRC1", 255)], Some(next())));
        v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 2)], Some(next())));
        v.extend(glbl("GLOBAL_LOAD_DWORD", &[("VDST", 4), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", off(0))]));
        v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 4), ("VSRC1", 3)], None));
        v.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", off(1))]));
        if r % 4 == 3 {
            let mut body = e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 3)], Some(next()));
            body.extend(glbl("GLOBAL_LOAD_DWORD", &[("VDST", 8), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", off(2))]));
            body.extend(e("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 8), ("VSRC1", 3)], None));
            v.extend(e("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(next())));
            v.extend(e("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 20), ("SSRC0", 106)], None));
            v.extend(e("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", (body.len() / 4) as u64)], None));
            v.extend(body);
            v.extend(e("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 20)], None));
        }
    }
    v.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", 0)]));
    v.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}

fn arg(args: &[String], name: &str, d: u64) -> u64 {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map_or(d, |v| v.parse().unwrap())
}

fn mb(b: u64) -> u64 {
    b >> 20
}

/// (private bytes of this process, system commit in use, system commit limit)
fn process_mem3() -> (u64, u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    unsafe {
        let mut pmc: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
        pmc.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        GetProcessMemoryInfo(GetCurrentProcess(), &mut pmc as *mut _ as *mut _, pmc.cb);
        let mut ms: MEMORYSTATUSEX = std::mem::zeroed();
        ms.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        GlobalMemoryStatusEx(&mut ms);
        (pmc.PrivateUsage as u64, ms.ullTotalPageFile - ms.ullAvailPageFile, ms.ullTotalPageFile)
    }
}

fn process_mem() -> (u64, u64) {
    let (p, c, _) = process_mem3();
    (p, c)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let count = arg(&args, "--count", 100_000) as u32;
    let rounds = arg(&args, "--rounds", 40) as u32;
    let jobs = arg(&args, "--jobs", 15) as usize;
    let every = arg(&args, "--every", 1000) as u32;
    let seed0 = arg(&args, "--seed0", 1_000_000) as u32;
    let stop_free = arg(&args, "--stop-free-gb", 8);
    let release_at = arg(&args, "--release-at", u64::MAX) as u32;
    unsafe {
        if args.iter().any(|a| a == "--debug") {
            let mut dbg: Option<ID3D12Debug> = None;
            D3D12GetDebugInterface(&mut dbg).unwrap();
            dbg.unwrap().EnableDebugLayer();
        }
        let mut dev: Option<ID3D12Device> = None;
        D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut dev).expect("device");
        let dev = dev.unwrap();
        let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
        let adapter: IDXGIAdapter3 = factory.EnumAdapterByLuid(dev.GetAdapterLuid()).unwrap();
        let desc = adapter.GetDesc1().unwrap();
        let name = String::from_utf16_lossy(&desc.Description).trim_end_matches('\0').to_string();
        let vid = |g| {
            let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            adapter.QueryVideoMemoryInfo(0, g, &mut i).unwrap();
            i
        };
        let l0 = vid(DXGI_MEMORY_SEGMENT_GROUP_LOCAL);
        let n0 = vid(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL);
        let (p0, c0) = process_mem();
        println!("adapter: {name}, local budget {} MB, non-local budget {} MB", mb(l0.Budget), mb(n0.Budget));
        println!("start: local {} MB, non-local {} MB, private {} MB, system commit {} of {} MB", mb(l0.CurrentUsage), mb(n0.CurrentUsage), mb(p0), mb(c0), mb(process_mem3().2));
        println!("count\tseconds\tper_s\tlocal_MB\tnonlocal_MB\tprivate_MB\tcommit_MB\tdxil_KB_avg\tdxc_ms\tcreate_ms");
        let rs = xcl::gpu::recomp::pso::abi_root_signature(&dev, false).unwrap();
        // D3D12 devices, root signatures and pipeline states are free-threaded
        struct Shared {
            dev: ID3D12Device,
            rs: ID3D12RootSignature,
            live: Mutex<Vec<ID3D12PipelineState>>,
        }
        unsafe impl Send for Shared {}
        unsafe impl Sync for Shared {}
        let sh = Shared { dev: dev.clone(), rs, live: Mutex::new(Vec::with_capacity(count as usize)) };
        // --dupes N: N pipelines of one fixed program first, all alive (like XCL's stand-ins,
        // which share one dummy shader)
        let dupes = arg(&args, "--dupes", 0) as u32;
        let mut dupe_live: Vec<ID3D12PipelineState> = Vec::with_capacity(dupes as usize);
        if dupes > 0 {
            let cs = CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 };
            let opt = Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, mem_windows: 4, ..Default::default() };
            let d = dxc::compile(&translate(&program(4, 7), &opt).unwrap().hlsl, "main", "cs_6_6", &[]).unwrap();
            let td = Instant::now();
            for _ in 0..dupes {
                let pd = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(sh.rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: d.as_ptr() as *const _, BytecodeLength: d.len() },
                    ..Default::default()
                };
                dupe_live.push(dev.CreateComputePipelineState(&pd).expect("dupe"));
                drop(std::mem::ManuallyDrop::into_inner(pd.pRootSignature));
            }
            let (p, c) = process_mem();
            println!("DUPES: {dupes} identical pipelines alive in {:.1} s ({:.1} us each), private {} MB, commit {} MB", td.elapsed().as_secs_f64(), td.elapsed().as_secs_f64() * 1e6 / dupes as f64, mb(p), mb(c));
        }
        let next = AtomicU32::new(0);
        let made = AtomicU32::new(0);
        let bytes = AtomicU64::new(0);
        let dxc_ns = AtomicU64::new(0);
        let create_ns = AtomicU64::new(0);
        let last = Mutex::new((0u32, 0u64, 0u64));
        let print = Mutex::new(());
        let t0 = Instant::now();
        let report = |n: u32| {
            let _g = print.lock().unwrap();
            let (l, nl, (p, c)) = (vid(DXGI_MEMORY_SEGMENT_GROUP_LOCAL), vid(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL), process_mem());
            let s = t0.elapsed().as_secs_f64();
            let (d, cr) = (dxc_ns.load(Ordering::Relaxed), create_ns.load(Ordering::Relaxed));
            let mut lst = last.lock().unwrap();
            let k = (n.saturating_sub(lst.0)).max(1) as u64;
            let (dm, cm) = ((d - lst.1) as f64 / k as f64 / 1e6, (cr - lst.2) as f64 / k as f64 / 1e6);
            *lst = (n, d, cr);
            println!("{n}\t{s:.1}\t{:.0}\t{}\t{}\t{}\t{}\t{}\t{dm:.0}\t{cm:.0}", n as f64 / s, mb(l.CurrentUsage), mb(nl.CurrentUsage), mb(p), mb(c), bytes.load(Ordering::Relaxed) / 1024 / n.max(1) as u64);
        };
        std::thread::scope(|sc| {
            for _ in 0..jobs {
                let (sh, next, made, bytes, report, dxc_ns, create_ns) = (&sh, &next, &made, &bytes, &report, &dxc_ns, &create_ns);
                sc.spawn(move || loop {
                    let k = next.fetch_add(1, Ordering::Relaxed);
                    if k >= count {
                        return;
                    }
                    let cs = CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 };
                    let opt = Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, mem_windows: 4, ..Default::default() };
                    let t = translate(&program(rounds, seed0 + k), &opt).unwrap();
                    let td = Instant::now();
                    let d = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
                    dxc_ns.fetch_add(td.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    bytes.fetch_add(d.len() as u64, Ordering::Relaxed);
                    let pd = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::ManuallyDrop::new(Some(sh.rs.clone())),
                        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: d.as_ptr() as *const _, BytecodeLength: d.len() },
                        ..Default::default()
                    };
                    let tc = Instant::now();
                    let r: windows::core::Result<ID3D12PipelineState> = sh.dev.CreateComputePipelineState(&pd);
                    create_ns.fetch_add(tc.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    drop(std::mem::ManuallyDrop::into_inner(pd.pRootSignature));
                    match r {
                        Ok(p) => sh.live.lock().unwrap().push(p),
                        Err(x) => {
                            let n = made.load(Ordering::Relaxed);
                            report(n);
                            let why = sh.dev.GetDeviceRemovedReason();
                            println!("FAILED after {n} live pipelines: {:#x}; device removed reason: {:#x}", x.code().0, why.err().map_or(0, |e| e.code().0));
                            std::process::exit(2);
                        }
                    }
                    let n = made.fetch_add(1, Ordering::Relaxed) + 1;
                    if n == release_at {
                        let dropped = std::mem::take(&mut *sh.live.lock().unwrap()).len();
                        report(n);
                        println!("RELEASED {dropped} pipelines at {n}");
                    }
                    if n % every == 0 {
                        report(n);
                        let (_, c, lim) = process_mem3();
                        if lim.saturating_sub(c) < stop_free << 30 {
                            println!("STOPPED after {n}: less than {stop_free} GB of system commit left ({} of {} MB in use)", mb(c), mb(lim));
                            std::process::exit(4);
                        }
                        if let Err(w) = sh.dev.GetDeviceRemovedReason() {
                            println!("DEVICE REMOVED after {n}: {:#x}", w.code().0);
                            std::process::exit(3);
                        }
                    }
                });
            }
        });
        println!("DONE: {} live pipelines, device ok: {}", sh.live.lock().unwrap().len(), dev.GetDeviceRemovedReason().is_ok());
    }
}
