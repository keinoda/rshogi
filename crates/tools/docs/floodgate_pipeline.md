# floodgate_pipeline

Floodgate（`wdoor.c.u-tokyo.ac.jp`）の公開棋譜を取得し、NNUE 学習向けに
CSA → SFEN へ変換するパイプライン。サブコマンドは `fetch-ratings` /
`fetch-index` / `download` / `extract` の 4 つ。

通信は `reqwest`（rustls）で行う。Floodgate は http アクセスを https へ 301
誘導するため、既定 root は https を直接指す（`--root` に http を渡しても
サーバ側リダイレクトで動作する）。

## サブコマンド

### `fetch-ratings`

レーティングページから高レートプレイヤー名を取得し、`download` の事前フィルタ
（`--player-file`）に使うリストを出力する。

```bash
cargo run -p tools --bin floodgate_pipeline -- fetch-ratings --min-rating 3900 --out high_rated.txt
```

- `--url <URL>`: レーティングページ URL。**未指定なら直近の日付のページを自動取得**する。
  ページは日次生成の日付スタンプ付き（`.../shogi/x/rating/players-floodgate-YYYYMMDD.html`）で、
  サーバ（JST）基準の当日から最大 7 日遡って最初に取得できたものを採用する。
- `--min-rating <N>`: この値以上のプレイヤーを出力（既定 3900）。
- `--out <PATH>`: 出力先。1 行 1 プレイヤーで `名前<TAB>レート` 形式。

### `fetch-index`

棋譜インデックス `00LIST.floodgate` をダウンロードする。

```bash
cargo run -p tools --bin floodgate_pipeline -- fetch-index --out 00LIST.floodgate
```

- `--root <URL>`: Floodgate root（既定は https）。
- `--out <PATH>`: 出力先。

### `download`

インデックスに記載された CSA ファイルを並列ダウンロードする。インデックス内の
絶対パスは末尾の `/x/` 以降を相対パスとして `--out-dir` 配下へ配置する。

```bash
cargo run -p tools --bin floodgate_pipeline -- download \
  --index 00LIST.floodgate --date-from 2026-03-10 --player-file high_rated.txt --concurrency 16
```

- `--index <PATH>` / `--root <URL>` / `--out-dir <DIR>`。
- `--limit <N>`: ダウンロード数の上限（テスト用）。
- `--date-from` / `--date-to <YYYY-MM-DD>`: 日付フィルタ。
- `--player-file <PATH>`: いずれかの対局者が含まれる対局のみ取得（`fetch-ratings` の出力を渡す）。
- `--concurrency <N>`: 並列数（既定 8、`0` で CPU コア数）。

### `extract`

ローカルの CSA から学習用 SFEN を抽出する。

```bash
cargo run -p tools --bin floodgate_pipeline -- extract --min-rating 3900 --max-ply 32
```

- `--root <DIR>` / `--out <PATH>`（`-` で標準出力、`.gz` 対応）。
- `--mode <initial|all|nth>`、`--nth <手数,...>`、`--min-ply` / `--max-ply`。
- `--mirror-dedup`（水平ミラー正規化で重複排除）/ `--emit-mirror`。
- `--per-game-cap <N>`（1 棋譜あたりの最大抽出数）/ `--min-rating <N>`。
- `--psv-out <PATH>`: SFEN テキストの代わりに **PSV（PackedSfenValue 40B）** で
  出力する（held-out 検証セット作成用）。`score=0` / `move16=0` で書くため、
  学習・検証に使う前に `rescore_psv` でスコアを付けること。`game_result` は CSA の
  終局手から**手番視点**で記録する（`%TORYO`/`%KACHI`/`%TIME_UP`/`%ILLEGAL_MOVE` は
  勝敗、`%SENNICHITE`/`%JISHOGI`/`%HIKIWAKE`/`%MAX_MOVES` は引き分け）。結果不明の
  棋譜（`%CHUDAN`・終局手なし）は捨てる。局面は SFEN 完全一致で dedup する。
  `--mirror-dedup` / `--emit-mirror` とは併用不可、`--out` は無視される。

## 典型的な流れ

```bash
# 1. 高レートプレイヤーリスト
cargo run -p tools --bin floodgate_pipeline -- fetch-ratings --min-rating 3900 --out high_rated.txt
# 2. インデックス取得
cargo run -p tools --bin floodgate_pipeline -- fetch-index --out 00LIST.floodgate
# 3. CSA ダウンロード（日付・プレイヤーでフィルタ）
cargo run -p tools --bin floodgate_pipeline -- download --date-from 2026-03-10 --player-file high_rated.txt --concurrency 16
# 4. SFEN 抽出
cargo run -p tools --bin floodgate_pipeline -- extract --min-rating 3900 --max-ply 32
```

## held-out 検証セット（floodgate.bin）の作り方

NNUE 学習の held-out（tatara の `--test-data`）用に、floodgate 実戦局面へ
教師モデルのスコアを付けた PSV を作る流れ:

```bash
# 1〜3 は上と同じ（レート・日付で対局を絞る）
# 4. PSV で抽出（中盤中心に間引く例: 16手目以降、1棋譜30局面まで）
cargo run --release -p tools --bin floodgate_pipeline -- extract \
  --min-rating 3900 --min-ply 16 --per-game-cap 30 --psv-out floodgate_raw.bin
# 5. 教師モデル（学習ラベルと同じモデル）でスコア付け
cargo run --release -p tools --features dlshogi-onnx --bin rescore_psv -- \
  --input floodgate_raw.bin --output-dir out \
  --dlshogi-onnx-model "$SHOGI_DATA/nnue/ponkotsu.onnx" \
  --onnx-gpu-id 0 --onnx-batch-size 1024 --onnx-tensorrt --onnx-tensorrt-cache trt
mv out/floodgate_raw.bin "$SHOGI_DATA/teachers/floodgate.bin"
```

held-out のラベルは**学習データと同じ教師モデル**で付けること（test_loss が
「未知局面での教師再現度」を測る量になる）。`game_result` は実対局の勝敗なので
test_accuracy（符号一致率）はそのまま解釈できる。
