# サーバー向け LayerStacks NNUE ビルド手順

## 目的

大きめの CPU サーバーで LayerStacks 系 NNUE を検証するための build / 起動 / smoke test
手順をまとめる。Suisho11 や fuuppi-v2 のような個別モデル名に固定せず、`.bin` ヘッダーの
architecture descriptor から FT 種別、LayerStacks の形状、bucket mode を判断して読み込む。

## 前提

- Rust toolchain は repository の `rust-toolchain.toml` に従う。
- x86_64 サーバーでは `RUSTFLAGS="-C target-cpu=native"` を付け、実行 CPU 向けに最適化する。
- 192 logical CPU のように SMT を含む host では、まず実効 96 thread 程度に固定して測る。
- `go` を含む検証では stdout を読み続け、`bestmove` まで待つ。`isready` 後は必ず
  `readyok` を確認してから `position` / `go` に進む。

## NNUE ヘッダー確認

読み込み前に、少なくとも version、architecture string、LayerStack の bucket 表現を確認する。

```bash
python3 - <<'PY' /path/to/nn.bin
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = path.read_bytes()
version, hash_value, arch_len = struct.unpack_from("<III", data, 0)
arch = data[12:12 + arch_len].decode("utf-8", errors="replace")
print(f"path={path}")
print(f"version=0x{version:08X}")
print(f"hash=0x{hash_value:08X}")
print(f"arch_len={arch_len}")
print(arch)
PY
```

目安:

| ヘッダー | bucket mode | 追加ファイル |
|---|---|---|
| `{LayerStack=9}` を含む legacy LayerStack | `kingrank9` | `progress.bin` 不要 |
| progress8kpabs 系 LayerStacks | `progress8kpabs` | `LS_PROGRESS_COEFF` で `progress.bin` 必須 |
| `SqrClippedReLU` と `ClippedReLU` が混在 | sqr+crelu concat 系 | descriptor に従い dispatch |
| `ClippedReLU` のみで `{LayerStack=9}` | clipped-only / KingRank9 系 | descriptor に従い dispatch |

`version=0x7AF32F21` で `num_buckets` field が無い互換ファイルもあるため、rshogi は
ヘッダー直後の値が bucket 数として妥当でない場合に FT hash として扱い、既定の 9 buckets へ
fallback する。

## 推奨ビルド

広い LayerStacks 互換性を優先するサーバー検証では preset edition を使う。

```bash
RUSTFLAGS="-C target-cpu=native" \
cargo build --release -p rshogi-usi \
  --no-default-features \
  --features search-no-pass-rules,edition-layerstacks
```

生成物:

```text
target/release/rshogi-usi
```

より再現性を重視して `engines/` 配下に manifest 付きで残す場合:

```bash
RUSTFLAGS="-C target-cpu=native" \
cargo xtask build --edition layerstacks --profile production
```

## 検証用の絞り込みビルド

特定の FT / LayerStacks slot だけを含めたい場合は atomic feature を明示する。たとえば
`HalfKA_hm_merged 1536x16x32` 系と `HalfKA_merged 1024x8x64 clipped-only` 系を同じ
binary で検証する場合:

```bash
RUSTFLAGS="-C target-cpu=native" \
cargo build --release -p rshogi-usi \
  --no-default-features \
  --features search-no-pass-rules,layerstack-arch,ft-halfka_hm_merged,ft-halfka_merged,layerstacks-1536x16x32,layerstacks-1024x16x32
```

現在の compatibility slot:

| descriptor 上の形状 | activation | 必要 feature | 備考 |
|---|---|---|---|
| `1536x16x32` | sqr+crelu concat | `layerstacks-1536x16x32` | progress8kpabs 系の代表 |
| `1536x32x32` | sqr+crelu concat | `layerstacks-1536x32x32` | 32x32 variant |
| `1024x16x32` | sqr+crelu concat | `layerstacks-1024x16x32` | 通常の 1024 slot |
| `1024x8x64` | clipped-only | `layerstacks-1024x16x32` | KingRank9 互換を同 slot で dispatch |
| `768x16x32` | sqr+crelu concat | `layerstacks-768x16x32` | 768 slot |
| `768x8x32` | sqr+crelu concat | `layerstacks-768x8x32` | 768 small slot |
| `512x16x32` | sqr+crelu concat | `layerstacks-512x16x32` | 512 slot |

`1024x8x64` は descriptor で識別されるため、Suisho11 固定の special case ではない。
ただし user-facing feature 名は既存の `layerstacks-1024x16x32` を互換 slot として再利用している。

## 起動 smoke test

KingRank9 系:

```text
usi
setoption name EvalFile value /path/to/nn.bin
isready
```

`readyok` が返れば NNUE load は完了している。KingRank9 と判定された LayerStacks では
`progress.bin` は不要。

progress8kpabs 系:

```text
usi
setoption name EvalFile value /path/to/nn.bin
setoption name LS_PROGRESS_COEFF value /path/to/progress.bin
isready
```

ロード済み NNUE の bucket mode が `progress8kpabs` の場合だけ、`LS_PROGRESS_COEFF` 未指定を
`isready` でエラーにする。HalfKX や KingRank9 LayerStacks ではこの検査は不要。

## スレッド数検証

高スレッド検証では、pipe にコマンドを流すだけの簡易 driver は使わない。探索中の `info` 出力を
読み続けないと、stdout の backpressure で探索が止まったように見えることがある。

最小の手順:

1. process を起動する。Linux では必要に応じて `taskset -c 0-95` で CPU を固定する。
2. `usi` を送り、`usiok` を待つ。
3. `setoption name EvalFile value ...` と必要な `LS_PROGRESS_COEFF` を送る。
4. `setoption name Threads value <n>` を送る。
5. `isready` を送り、`readyok` を待つ。
6. `position startpos` を送る。
7. `go movetime <ms>` を送り、stdout を読み続けて `bestmove` まで待つ。

192 logical CPU host で実効 96 thread を使う例:

```bash
taskset -c 0-95 ./target/release/rshogi-usi
```

## サーバー向け USI option

server build では GUI に提示する hash 上限を大きめにしている。

```text
option name USI_Hash type spin default 256 min 1 max 262144
option name EvalHash type spin default 256 min 0 max 262144
```

実際に確保するメモリは host の空きメモリ、同時起動数、NUMA 構成に合わせて決める。
大きな値を使う場合も、`setoption` 後に `isready` / `readyok` を確認してから探索する。

## 最低限の確認コマンド

```bash
cargo check -p rshogi-usi \
  --no-default-features \
  --features search-no-pass-rules,edition-layerstacks

cargo test -p rshogi-core \
  --no-default-features \
  --features layerstack-arch,ft-halfka_hm_merged,ft-halfka_merged,layerstacks-1536x16x32,layerstacks-1024x16x32 \
  parse_layer_stacks_architecture -- --nocapture
```

release binary を作った後は、対象 NNUE ごとに `isready` で `readyok` を確認し、必要なら
`go movetime 1000` 程度の短い探索で `bestmove` まで待つ。
