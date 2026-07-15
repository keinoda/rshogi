# vast.ai 用 base image + onstart.sh

vast.ai のインスタンス準備時間（= 課金時間）を最小化するための構成。

- **Docker イメージ（課金前の loading 段階で pull される側）**: サードパーティの重量物を
  全て焼き込む。CUDA 12.9 devel + cuDNN 9、TensorRT 10.11、ONNX Runtime 1.24.2 GPU
  （`crates/tools/docs/rescore_psv.md` のバージョンと一致）、LLVM/clang 21（tatara 用）、
  Rust stable 1.95.0 + nightly-2026-04-03（rustfmt/clippy 込み）、aria2/unzip、
  huggingface_hub[hf_transfer]。**private コードは含まない**。
- **onstart.sh（課金開始後、全自動・冪等）**: repo clone、教師データの HF ダウンロード
  （蒸留済み 352GB + run C 用元ラベル 59GB）、progress.bin 取得、rescore_psv +
  パイロット用 engine 2 種 + tatara + shogitest のビルドを tmux で並列実行。

## 教師データの取得経路

取得は全て **HuggingFace** から（public、トークン不要。8Gbps 回線なら 352GB ≈ 1 時間弱）:

- 蒸留済み（ponkotsu 再評価済み 88.0 億局面 = `dlsuisho_unique_001..019.bin`）→
  `$SHOGI_DATA/teachers/distilled/`
- 元ラベル（蒸留元 dataset）→ `$SHOGI_DATA/teachers/orig/`。既定はパイロット run C 用の
  3 ファイルのみ。**残り 40% の追い蒸留（rescore）をやる場合は `ORIG_INCLUDE` に
  019..030 を指定**すれば入力データが揃う（rescore_psv のビルドは従来どおり自動）
- add_aobazero プール（将来の増強用）→ `$SHOGI_DATA/teachers/pool/`。`HF_POOL=1` のときのみ

gigafile からの直接 DL は onstart から**廃止**した（海外 DC は 1 接続 ~200kB/s 制限で、
aria2 並列でも数 MB/s 止まりのため非現実的）。蒸留済みデータは国内 VM 上の
`relay_gigafile_to_hf.sh` で HF へ退避済み。スクリプト群（`gigafile_dl.sh` /
`extract_stored_zips.py` / `relay_gigafile_to_hf.sh`）は今後 gigafile で
データを受領したときの退避用に repo に残している。

## イメージの build（GitHub Actions が自動実行）

`.github/workflows/vast-image.yml` が本ディレクトリの Dockerfile を build し、
`ghcr.io/keinoda/shogi-lab:cuda129-trt1011` へ push する。trigger は
`infra/vast/Dockerfile` の変更 push、または Actions タブからの手動実行
（workflow_dispatch）。追加 secret は不要（GITHUB_TOKEN で完結）。

**初回 push 後に 1 回だけ必要な操作（実施済み）**: GitHub の Packages で `shogi-lab` を
**Public に変更**する（プロフィール → Packages → shogi-lab → Package settings →
Change visibility）。vast.ai が匿名 pull できるようにするため。

（ローカルで build する場合: `docker build -t ghcr.io/keinoda/shogi-lab:cuda129-trt1011 infra/vast/` — 空きディスク 30GB 以上推奨）

## vast.ai テンプレート設定

| 項目 | 値 |
|---|---|
| Docker Image | `ghcr.io/keinoda/shogi-lab:cuda129-trt1011`（**`ghcr.io/` から書く**。省略すると Docker Hub を探して manifest unknown になる） |
| Launch Mode | ssh |
| On-start Script | `onstart.sh` の中身を貼り付け |
| Disk | **1TB**（蒸留 352GB + 元ラベル 59GB + パイロット作業 ~150GB。019..030 の追い蒸留までやるなら 2TB） |
| GPU フィルタ | Max CUDA 12.8 以上の host（TensorRT 10.11 は cuda-12.9 ビルド）。RTX 4090 / 5090 推奨 |

### Environment Variables（全て省略可 — 既定値で動く）

| 変数 | 意味 | 既定 |
|---|---|---|
| `DISTILLED_DATASET` | 蒸留済み教師データの HF dataset（全量を自動 DL、`.bin` 19 本で完了判定） | `ngs436/dlsuisho-ponkotsu-distilled` |
| `SKIP_DISTILLED` | `1` で蒸留済みデータの DL をスキップ | なし |
| `ORIG_DATASET` | 元ラベル（蒸留元）の HF dataset — run C 対照 / 追い蒸留の入力 | `washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique` |
| `ORIG_INCLUDE` | 元ラベルから取得するファイル（空白/カンマ区切り、glob 可）。追い蒸留なら 019..030、全量なら `*.bin`（587GB）。空文字で DL しない | `dlsuisho_unique_001..003.bin`（3 ファイル 59GB） |
| `SKIP_ORIG` | `1` で元ラベルデータの DL をスキップ | なし |
| `HF_POOL` | `1` で add_aobazero プールを全量 DL（679GB、Disk 2TB 級が必要。将来の増強用・**既定オフ**） | なし |
| `HF_DATASET` | `HF_POOL=1` のときの対象 dataset | `washiun/Knowledge_distilled_by_DLSuisho15b_add_aobazero_unique` |
| `GIT_TOKEN` | GitHub PAT（**通常不要** — 対象 repo は全て public） | なし |
| `RSHOGI_BRANCH` | rshogi のブランチ | `claude/busy-faraday-umwgl8` |
| `TATARA_BRANCH` | tatara のブランチ | `main` |
| `SHOGITEST_BRANCH` | shogitest のブランチ | `claude/nightly-toolchain-pin` |

## 起動後の確認

```bash
tail -f /workspace/onstart.log     # onstart の進行
tmux ls                            # distdl / origdl / build_rshogi / build_tatara / build_shogitest
ls /workspace/.onstart/            # 完了 marker
tail -f /workspace/logs/distdl.log # 蒸留済みデータ DL の進行
grep -i warning /workspace/onstart.log /workspace/logs/*.log   # 異常の有無
```

ビルド群は 10〜20 分。蒸留済みデータ（352GB）は回線次第で 30 分〜数時間。
`distdl` は `.bin` が 19 本揃ったときだけ完了 marker を置く（HF 側のアップロードが
未完のうちに起動した場合は WARNING を出すので、揃ってから onstart を再実行すれば
差分 resume される）。

progress.bin は onstart が keinoda/yaneuraou の `sojo_tsec7` ブランチ
（`source/progress.bin`）から自動取得して `$SHOGI_DATA/progress/` に配置する。

イメージにも onstart にも入れられないもの（scp で配置）:

1. 開始局面集 → `/workspace/book/openings.epd`
2. rescore 用 ONNX モデル → `$SHOGI_DATA/nnue/`（**追い蒸留するときのみ必要**。
   パイロットでは不要 — 蒸留済みデータを直接学習に使う）

## 段階学習ハーネス (staged_train.py)

800 SB を一括予約せず「学習 → held-out 評価 → 延長判定 → resume」を自動で回す
スーパーバイザ。tatara の `--resume`（optimizer 状態 + LR horizon 込みの真の resume）
を使うため、**段階延長しても LR スケジュール（gamma^sb）は圧縮されず連続**する。

```bash
pip3 install --break-system-packages matplotlib   # レポート描画用 (無くても学習は可)
tmux new -d -s train "python3 $WORK/rshogi/infra/vast/staged_train.py \
  --data $SHOGI_DATA/teachers/<シャッフル済み学習PSV> \
  --test-data $SHOGI_DATA/teachers/floodgate.bin \
  --run-dir $WORK/runs/base2048 --gpu 0 \
  2>&1 | tee $WORK/logs/staged_train.log"
```

- `batches-per-superbatch` は `round(N_train / (20 × batch_size))` を自動計算
  （≈20 SB = 1 dataset pass。`--sbs-per-pass` で変更可）
- 既定: 2048x16x64 / 9 buckets / v17 系ハイパラ固定 / ladder 120→200→300→400
  （以後 +100）/ save-rate 10 / raw ckpt 8 個保持
- 判定: 5点MA best が直近 10SB 内 → 延長、±0.1% 横ばい → +40SB を 1 回、
  best から 20SB 以上 & 0.2% 悪化 → 停止（stage 途中でも abort）
- milestone（各 stage 目標）と held-out 最良の raw ckpt は `protected/` に
  hardlink され rolling 削除から保護。停止後は最良近傍の `.bin` を自己対局候補として列挙
- `run-dir/report.html` / `report.png` を毎 SB 更新（passes 軸の loss + MA3/5、
  LR、throughput、ETA、fp16 clamp、ckpt 位置。30 秒自動リロード）。閲覧は
  `ssh -L 8000:localhost:8000 <instance>` + `python3 -m http.server 8000 -d $WORK/runs/base2048`
- ハーネス自体も冪等（同一コマンド再実行で続きから）。tatara 異常終了は
  resume で自動リトライ（`--retries`）

**前提**: 学習 PSV は事前に全域シャッフルしておくこと（tatara の dataloader は
シャッフルなしの逐次読み）。88 億全量なら `shuffle_psv --chunk-size` の一時領域
込みで入力の 3 倍 ≈ 1.06TB を使うため、**シャッフルだけは 2TB ディスクで実施**する。

## 注意

- TensorRT のエンジンコンパイルはモデル×GPU 固有のため事前化できない。初回の
  rescore_psv 実行時に数分かかる（`--onnx-tensorrt-cache` で 2 回目以降は即起動）
- vast の interruptible インスタンスでも、hf download の差分 resume・rescore の
  `.done` レジューム・tatara の `--resume`・onstart の冪等性で再開に耐える
- ORT / TensorRT のバージョンを変える場合は `crates/tools/docs/rescore_psv.md` と
  合わせて Dockerfile を更新する
