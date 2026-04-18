# sgl-on-bb4424a — rebase notes

This branch is a downstream fork of [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
carrying the sglang integration shims from
[sgl-project/DeepGEMM@sgl-release](https://github.com/sgl-project/DeepGEMM/tree/sgl-release)
rebased onto a newer upstream base.

## Goal

sglang (via [`sglang/sgl-kernel`](https://github.com/sgl-project/sglang/tree/main/sgl-kernel))
pins DeepGEMM in its `CMakeLists.txt` and compiles
`csrc/python_api.cpp` as the entry point for the `deep_gemm_cpp` extension.
The currently pinned SHA in `sgl-kernel` is
[`ffe2b6b`](https://github.com/sgl-project/DeepGEMM/commit/ffe2b6b97420a9f8c58268ca55755168e6e2f360),
the tip of the `sgl-release` branch. That branch is anchored on upstream
[`239112c`](https://github.com/deepseek-ai/DeepGEMM/commit/239112cb4cd4e52587c662624aee6beda8bd9518)
(Oct 1, 2025).

The goal of this rebase is to move the same 8 sgl shim commits onto a newer upstream
base while avoiding the large upstream refactor at
[`38f8ef7`](https://github.com/deepseek-ai/DeepGEMM/commit/38f8ef73a48a42b1a04e0fa839c2341540de26a6)
("Multiple updates and refactorings #231", Nov 21, 2025), which deletes and renames many
of the files the sgl patches touch.

## Base SHA

[`bb4424a` — Fix sum_k * shape_m overflow](https://github.com/deepseek-ai/DeepGEMM/commit/bb4424aad49927283349738dfa7b54f5baad6025)
(Nov 19, 2025), the direct parent of `38f8ef7`.

Rationale: `bb4424a` is the last upstream commit before the Nov 21 refactor,
so cherry-picking the sgl shims onto it touches files in the same shape the
shims were written against.

## Upstream timeline considered

| SHA | Date | Role |
|---|---|---|
| [`239112c`](https://github.com/deepseek-ai/DeepGEMM/commit/239112cb4cd4e52587c662624aee6beda8bd9518) | 2025-10-01 | `sgl-release` base (old) |
| [`bb4424a`](https://github.com/deepseek-ai/DeepGEMM/commit/bb4424aad49927283349738dfa7b54f5baad6025) | 2025-11-19 | **this branch's base** |
| [`38f8ef7`](https://github.com/deepseek-ai/DeepGEMM/commit/38f8ef73a48a42b1a04e0fa839c2341540de26a6) | 2025-11-21 | Multiple updates and refactorings #231 — skipped |
| [`9b680f4`](https://github.com/deepseek-ai/DeepGEMM/commit/9b680f428484625f4f35dc3617f134187c6bcd4a) | 2025-12-05 | Update install.sh — attempted first, blocked by `38f8ef7` |
| [`0f5f266`](https://github.com/deepseek-ai/DeepGEMM/commit/0f5f2662027f0db05d4e3f6a94e56e2d8fc45c51) | 2026-01-16 | Multiple updates and refactorings #280 — skipped |
| [`d30fc36`](https://github.com/deepseek-ai/DeepGEMM/commit/d30fc36c8f229f4f873b90a492f6e19e6e610923) | 2026-03-22 | base of sgl `release` branch (TVM FFI, not used here) |
| [`7f2a703`](https://github.com/deepseek-ai/DeepGEMM/commit/7f2a703ed51ac1f7af07f5e1453b2d3267d37d50) | 2026-04-17 | Public release 26/04 #304 — explicitly skipped |

## Cherry-picked commits

All from [`sgl-project/DeepGEMM@sgl-release`](https://github.com/sgl-project/DeepGEMM/tree/sgl-release),
applied oldest-first:

| Source SHA (sgl fork) | Local SHA on this branch | Message |
|---|---|---|
| [`301cbc1`](https://github.com/sgl-project/DeepGEMM/commit/301cbc1) | `814578d` | feat: support libtorch |
| [`f4adba8`](https://github.com/sgl-project/DeepGEMM/commit/f4adba8) | `9d9fd46` | feat: support misc kernel launch |
| [`6635dd2`](https://github.com/sgl-project/DeepGEMM/commit/6635dd2) | `7e606c6` | feat: add signal for SBO in SM90 masked gemm |
| [`a01ab1a`](https://github.com/sgl-project/DeepGEMM/commit/a01ab1a) | `bf6bd55` | feat: add test for signal GEMM |
| [`5f8a71a`](https://github.com/sgl-project/DeepGEMM/commit/5f8a71a) | `1c78b90` | bugfix |
| [`3a29764`](https://github.com/sgl-project/DeepGEMM/commit/3a29764) | `2085ed5` | add max_block_n |
| [`f259a0e`](https://github.com/sgl-project/DeepGEMM/commit/f259a0e) | `81c3ce8` | bugfix |
| [`5f99d8d`](https://github.com/sgl-project/DeepGEMM/commit/5f99d8d) | `67a4d71` | rollback |

The source merge commit
[`ffe2b6b`](https://github.com/sgl-project/DeepGEMM/commit/ffe2b6b) (PR #14) is
omitted — merge commits carry no independent code changes.

## Conflicts and their resolution

Two files required manual resolution. Neither touches `csrc/python_api.cpp` or
any CUDA kernel; both live in Python packaging plumbing that
`sgl-kernel`'s `CMakeLists.txt` does not consume.

### 1. `deep_gemm/__init__.py` (during `301cbc1` — libtorch)

Upstream had added `__version__ = '2.1.1'` between `239112c` and `bb4424a`.
The sgl libtorch patch replaces the entire `deep_gemm_cpp.init(...)` block with
lazy `torch.ops.deep_gemm.*` wiring, so git flagged the context as conflicted.

**Resolution:** took the sgl patch verbatim (that is the purpose of the libtorch
shim) and re-appended `__version__ = '2.1.1'` at the end of the file to preserve
the upstream addition.

### 2. `setup.py` (during `f259a0e` and `5f99d8d`)

The libtorch commit [`301cbc1`](https://github.com/sgl-project/DeepGEMM/commit/301cbc1)
had already refactored the `ext_modules=[...]` list in `setup.py` into a
`get_ext_modules()` helper. The later `f259a0e` bugfix (rename
`deep_gemm_cpp` → `deep_gemm.deep_gemm_cpp`) and its rollback `5f99d8d` were
both written against the old inline block, so they didn't textually apply.

**Resolution:** ported the rename into the new `get_ext_modules()` helper for
`f259a0e`, then reverted it for `5f99d8d`. Net state: `name='deep_gemm_cpp'`
(the original upstream name), matching what `sgl-release` ends up at.

## Consuming this branch

In [`sglang/sgl-kernel/CMakeLists.txt`](https://github.com/sgl-project/sglang/blob/main/sgl-kernel/CMakeLists.txt)
at the `repo-deepgemm` block (lines 56–61 at time of writing):

```cmake
FetchContent_Declare(
    repo-deepgemm
    GIT_REPOSITORY https://github.com/bzhng-development/DeepGEMM
    GIT_TAG        67a4d717e7840e8852ae000ee0c80c6c77fcc6bc
    GIT_SHALLOW    OFF
)
```

Then rebuild `sgl-kernel`, clear the JIT cache (`rm -rf ~/.deep_gemm` or
`$DG_JIT_CACHE_DIR`), and run the relevant tests under
`sglang/sgl-kernel/tests/`:

- `test_fp8_blockwise_moe.py`
- `test_es_fp8_blockwise_moe.py`
- `test_es_mxfp8_blockscaled_moe.py`

## How to reproduce

```bash
git clone https://github.com/bzhng-development/DeepGEMM
cd DeepGEMM
git remote add upstream https://github.com/deepseek-ai/DeepGEMM
git remote add sgl https://github.com/sgl-project/DeepGEMM
git fetch upstream
git fetch sgl

git checkout -b sgl-on-bb4424a bb4424aad49927283349738dfa7b54f5baad6025
git cherry-pick 301cbc1 f4adba8 6635dd2 a01ab1a 5f8a71a 3a29764 f259a0e 5f99d8d
# resolve deep_gemm/__init__.py during 301cbc1 (take theirs + keep __version__)
# resolve setup.py during f259a0e and 5f99d8d (port rename into get_ext_modules helper)
```
