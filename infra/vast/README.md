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
| Disk | 下の「ディスクサイズの目安」参照（蒸留済みデータのみなら 1TB、HF プール併用なら 1.5TB+） |
| GPU フィルタ | Max CUDA 12.8 以上の host（TensorRT 10.11 は cuda-12.9 ビルド）。RTX 4090 / 5090 推奨 |

### ディスクサイズの目安

- **蒸留済みデータ (gigafile) のみ**（`SKIP_HF=1`）: zip 4 本 352GB を DL →
  1 本ずつ展開・検証後に削除するためピーク **約 450GB**、定常 352GB。
  パイロット作業分を足して **1TB** で十分
- **HF プール（元ラベル 679GB）も併用**: ピーク 約 1.13TB / 定常 約 1.03TB →
  **1.5TB 以上**（学習 checkpoint・rescore 出力まで見込むなら 2TB）

### Environment Variables（テンプレートの環境変数欄）

| 変数 | 意味 | 既定 |
|---|---|---|
| `GIT_TOKEN` | GitHub PAT（**通常不要** — 対象 4 repo は全て public。private repo を使う構成に変えた場合のみ、read-only の fine-grained PAT を対象 repo 限定で発行して設定し、不要になったら revoke） | なし |
| `RSHOGI_BRANCH` | rshogi のブランチ | `claude/busy-faraday-umwgl8` |
| `TATARA_BRANCH` | tatara のブランチ | `main` |
| `SHOGITEST_BRANCH` | shogitest のブランチ | `claude/nightly-toolchain-pin` |
| `SKIP_HF` | `1` で HF プールの DL をスキップ（蒸留済みデータのみで作業する場合） | なし |
| `GIGAFILE_URLS` | gigafile.nu の URL。複数は**カンマ区切り**で指定（vast の環境変数欄はスペースを含む値を quote なしでは受け付けず "Invalid value" になる。スペース区切りを使う場合は値全体を `"..."` で囲む）。指定時は蒸留済み教師データを `$SHOGI_DATA/teachers/distilled/` へ DL → zip 展開・検証 → zip 削除まで自動実行（`gigafile_dl.sh` + `extract_stored_zips.py`、resume 対応）。**URL は repo に commit せず、テンプレートの環境変数でのみ渡す**（public repo のため） | なし |
| `GIGAFILE_DLKEY` | gigafile のダウンロードキー | なし |

HF の教師データ（元ラベルのプール）は**常に全量（34 shard / 679GB）を自動ダウンロード**する
（`SKIP_HF=1` で無効化可）。止める場合は `tmux kill-session -t hfdl`、
再開は onstart 再実行か同コマンドで resume される。gigafile のリンクには**保持期限がある**ため、
蒸留済みデータを受領したら速やかにダウンロードすること。

gigafile の zip（無圧縮 STORED）は DL 完了後に `extract_stored_zips.py` が
`$SHOGI_DATA/teachers/distilled/` 直下へ flatten 展開する。展開時に CRC・サイズ・
40B レコード境界を検証し、**検証に通った zip だけを削除**する（失敗した zip は
再ダウンロードに備えて残す）。展開済みの印は `<zip名>.extracted` スタンプで、
再実行時は DL も展開も skip される。

**gigafile の速度について**: gigafile.nu は国内向けサービスで、海外 DC からは
1 接続あたり **~200kB/s** に制限される（実測。接続を増やしても 1 本あたりは落ちない）。
このため `gigafile_dl.sh` は aria2c による分割並列（既定 8 接続/URL）+ URL 単位の
同時実行（既定 4、URL は別サーバに載っている）で計 ~32 接続まで束ねる。
それでも国内リージョンからのアクセスより桁で遅いことがあるので、
急ぐ場合は国内の VM 経由で HF 等へ退避するのが確実。

## 起動後の確認

```bash
tail -f /workspace/onstart.log     # onstart の進行
tmux ls                            # hfdl / gigafile / build_rshogi / build_tatara / build_shogitest
ls /workspace/.onstart/            # 完了 marker
tail -f /workspace/logs/gigafile.log   # 蒸留済みデータの DL/展開の進行
```

ビルド群は 10〜20 分、shard 3 枚（60GB）は 1Gbps で 10〜15 分（並列実行）。

progress.bin は onstart が keinoda/yaneuraou の `sojo_tsec7` ブランチ
（`source/progress.bin`）から自動取得して `$SHOGI_DATA/progress/` に配置する。

イメージにも onstart にも入れられないもの（scp で配置）:

1. rescore 用 ONNX モデル → `$SHOGI_DATA/nnue/`
2. 開始局面集 → `/workspace/book/openings.epd`

## 注意

- TensorRT のエンジンコンパイルはモデル×GPU 固有のため事前化できない。初回の
  rescore_psv 実行時に数分かかる（`--onnx-tensorrt-cache` で 2 回目以降は即起動）
- vast の interruptible インスタンスでも、rescore の `.done` レジューム・tatara の
  `--resume`・onstart の冪等性で再開に耐える
- ORT / TensorRT のバージョンを変える場合は `crates/tools/docs/rescore_psv.md` と
  合わせて Dockerfile を更新する
