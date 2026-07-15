# vast.ai 用 base image + onstart.sh

vast.ai のインスタンス準備時間（= 課金時間）を最小化するための構成。

- **Docker イメージ（課金前の loading 段階で pull される側）**: サードパーティの重量物を
  全て焼き込む。CUDA 12.9 devel + cuDNN 9、TensorRT 10.11、ONNX Runtime 1.24.2 GPU
  （`crates/tools/docs/rescore_psv.md` のバージョンと一致）、LLVM/clang 21（tatara 用）、
  Rust stable 1.95.0 + nightly-2026-04-03（rustfmt/clippy 込み）、huggingface_hub[hf_transfer]。
  **private コードは含まない**。
- **onstart.sh（課金開始後、全自動・冪等）**: repo clone、rescore_psv + パイロット用
  engine 2 種 + tatara + shogitest のビルド、教師データ shard の DL を tmux で並列実行。

## イメージの build（GitHub Actions が自動実行）

`.github/workflows/vast-image.yml` が本ディレクトリの Dockerfile を build し、
`ghcr.io/keinoda/shogi-lab:cuda129-trt1011` へ push する。trigger は
`infra/vast/Dockerfile` の変更 push、または Actions タブからの手動実行
（workflow_dispatch）。追加 secret は不要（GITHUB_TOKEN で完結）。

**初回 push 後に 1 回だけ必要な操作**: GitHub の Packages で `shogi-lab` を
**Public に変更**する（プロフィール → Packages → shogi-lab → Package settings →
Change visibility）。vast.ai が匿名 pull できるようにするため。イメージに private
コードは含まれないので公開して問題ない。

（ローカルで build する場合: `docker build -t ghcr.io/keinoda/shogi-lab:cuda129-trt1011 infra/vast/` — 空きディスク 30GB 以上推奨）

## vast.ai テンプレート設定

| 項目 | 値 |
|---|---|
| Docker Image | `ghcr.io/keinoda/shogi-lab:cuda129-trt1011` |
| Launch Mode | ssh |
| On-start Script | `onstart.sh` の中身を貼り付け |
| Disk | パイロット: 300GB 以上 / 全量蒸留まで見据えるなら 2.5TB+ |
| GPU フィルタ | Max CUDA 12.8 以上の host（TensorRT 10.11 は cuda-12.9 ビルド）。RTX 4090 / 5090 推奨 |

### Environment Variables（テンプレートの環境変数欄）

| 変数 | 意味 | 既定 |
|---|---|---|
| `GIT_TOKEN` | private repo clone 用 GitHub PAT。read-only の fine-grained PAT を対象 repo 限定で発行し、不要になったら revoke | なし |
| `RSHOGI_BRANCH` | rshogi のブランチ | `claude/busy-faraday-umwgl8` |
| `TATARA_BRANCH` | tatara のブランチ | `main` |
| `SHOGITEST_BRANCH` | shogitest のブランチ | `claude/nightly-toolchain-pin` |
| `DOWNLOAD_SHARDS` | DL する教師データ shard。`"000 001 002"`（パイロット 60GB）/ `all`（679GB）/ `none` | `000 001 002` |

## 起動後の確認

```bash
tail -f /workspace/onstart.log     # onstart の進行
tmux ls                            # hfdl / build_rshogi / build_tatara / build_shogitest
ls /workspace/.onstart/            # 完了 marker
```

ビルド群は 10〜20 分、shard 3 枚（60GB）は 1Gbps で 10〜15 分（並列実行）。

イメージにも onstart にも入れられないもの（scp で配置）:

1. rescore 用 ONNX モデル → `$SHOGI_DATA/nnue/`
2. progress.bin → `$SHOGI_DATA/progress/`
3. 開始局面集 → `/workspace/book/openings.epd`

## 注意

- TensorRT のエンジンコンパイルはモデル×GPU 固有のため事前化できない。初回の
  rescore_psv 実行時に数分かかる（`--onnx-tensorrt-cache` で 2 回目以降は即起動）
- vast の interruptible インスタンスでも、rescore の `.done` レジューム・tatara の
  `--resume`・onstart の冪等性で再開に耐える
- ORT / TensorRT のバージョンを変える場合は `crates/tools/docs/rescore_psv.md` と
  合わせて Dockerfile を更新する
