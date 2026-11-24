//! Local probe for #1448 (not for merging): a synthetic compute program for the compact
//! backend whose blocks repeat with the same code but different branch targets.
//!
//! Every round is the same code (same literals): LCG step, a load + store, then an `if`
//! header (`v_cmp` / `s_and_saveexec` / `s_cbranch_execz`). The header jumps over a body that
//! is one block in even rounds and three blocks (a nested `if`) in odd rounds, so identical
//! headers jump by different numbers of blocks. `--vary 0` makes every body one block (same
//! targets), as a control.
//!
//! ```text
//! cmp1448 --out DIR --name NAME --rounds N [--vary 1] [--heap 0|1] [--salt N] [--dispatch N]   (prints cases, sizes, DXC time)
//! ```

use std::path::PathBuf;
use std::time::Instant;

use xcl::gpu::rdna2::encode;
use xcl::gpu::recomp::{compact, dxc, CsInterface, Mode, Options};

const V: u64 = 256;

fn e(name: &str, enc: &str, f: &[(&str, u64)], lit: Option<u32>) -> Vec<u8> {
    encode(name, enc, f, lit).unwrap_or_else(|x| panic!("{name}: {x}"))
}

fn glbl(op: &str, f: &[(&str, u64)]) -> Vec<u8> {
    e(op, "ENC_FLAT_GLBL", f, None)
}

/// `if (v2 > lit) { body }` with the exec save / restore the console uses.
fn if_block(lit: u32, body: Vec<u8>, sdst: u64) -> Vec<u8> {
    let mut v = e("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", 255), ("VSRC1", 2)], Some(lit));
    v.extend(e("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", sdst), ("SSRC0", 106)], None));
    v.extend(e("S_CBRANCH_EXECZ", "ENC_SOPP", &[("SIMM16", (body.len() / 4) as u64)], None));
    v.extend(body);
    v.extend(e("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", sdst)], None));
    v
}

fn program(rounds: u32, vary: bool, salt: u32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend(e("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None));
    v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 0), ("VSRC1", 6)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 1)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 2), ("SRC0", V)], None));
    v.extend(e("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 255)], Some(0x1234_5677 ^ salt)));
    for r in 0..rounds {
        v.extend(e("V_MUL_LO_U32", "ENC_VOP3", &[("VDST", 2), ("SRC0", V + 2), ("SRC1", 255)], Some(1_664_525)));
        v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 2), ("SRC0", 255), ("VSRC1", 2)], Some(1_013_904_223)));
        v.extend(glbl("GLOBAL_LOAD_DWORD", &[("VDST", 4), ("ADDR", 6), ("SADDR", 0x7D), ("OFFSET", 16)]));
        v.extend(e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 4), ("VSRC1", 3)], None));
        v.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", 32)]));
        let mut body = e("V_ADD_NC_U32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 255), ("VSRC1", 3)], Some(0x9E37_79B9));
        body.extend(e("V_XOR_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", V + 2), ("VSRC1", 3)], None));
        if vary && r % 2 == 1 {
            // a nested if: the outer body is three blocks instead of one
            body.extend(if_block(0x4000_0001, e("V_LSHRREV_B32", "ENC_VOP2", &[("VDST", 3), ("SRC0", 129), ("VSRC1", 3)], None), 22));
        }
        v.extend(if_block(0x8000_0001, body, 20));
    }
    v.extend(glbl("GLOBAL_STORE_DWORD", &[("ADDR", 6), ("DATA", 3), ("SADDR", 0x7D), ("OFFSET", 0)]));
    v.extend(e("S_ENDPGM", "ENC_SOPP", &[], None));
    v
}

fn arg<'a>(a: &'a [String], n: &str, d: &'a str) -> &'a str {
    a.iter().position(|x| x == n).and_then(|i| a.get(i + 1)).map_or(d, |s| s.as_str())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let out = PathBuf::from(arg(&a, "--out", "."));
    std::fs::create_dir_all(&out).unwrap();
    let name = arg(&a, "--name", "c");
    let rounds: u32 = arg(&a, "--rounds", "1500").parse().unwrap();
    let vary = arg(&a, "--vary", "1") == "1";
    let heap = arg(&a, "--heap", "0") == "1";
    let salt: u32 = arg(&a, "--salt", "0").parse().unwrap();
    let code = program(rounds, vary, salt);
    let cs = CsInterface { threads: [128, 1, 1], user_sgprs: 2, tgid: [true, false, false], tg_size: false, tid_components: 1 };
    let opt = Options { mode: Mode::Compute(cs), wave64: true, split_wave64: true, mem_windows: 4, heap_windows: heap, ..Default::default() };
    let t = compact::translate_compact(&code, &opt).unwrap();
    assert!(t.complete(), "{:?}", t.unsupported_why);
    let file = out.join(format!("{name}.dxil"));
    std::fs::write(file.with_extension("hlsl"), &t.hlsl).unwrap();
    let t0 = Instant::now();
    let d = dxc::compile(&t.hlsl, "main", "cs_6_6", &[]).unwrap();
    std::fs::write(&file, &d).unwrap();
    println!(
        "{}\tinstructions {}\twanted {}\tcases {}\thlsl {} KB\tdxil {} KB\tdxc {:.1} s",
        file.display(),
        code.len() / 4,
        compact::compact_wanted(&code),
        t.blocks,
        t.hlsl.len() / 1024,
        d.len() / 1024,
        t0.elapsed().as_secs_f64()
    );
    // --dispatch N: run it N times on the GPU (64 groups, addresses in window 0: s0 = 0, s1 = 0)
    let reps: usize = arg(&a, "--dispatch", "0").parse().unwrap();
    if reps > 0 {
        let mut t = xcl::gpu::recomp::runner::dispatch_timed(&d, [64, 1, 1], &[0, 0], &vec![0u32; 1 << 14], reps).unwrap();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("{}\tdispatch median {:.3} ms (min {:.3}, max {:.3}, {} runs)", file.display(), t[t.len() / 2], t[0], t[t.len() - 1], t.len());
    }
}
