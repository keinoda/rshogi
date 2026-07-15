#!/usr/bin/env bash
# vast.ai onstart — 課金開始直後に「イメージに焼けないもの」を全自動で準備する。
# 冪等: 再起動時は完了済みステップを skip する。進捗は /workspace/onstart.log と
# marker (/workspace/.onstart/) で確認。ビルド/DL は tmux セッションで並列に走る。
#
# 教師データは HF の公開 dataset から取得する (8Gbps 回線なら 352GB ≈ 1 時間弱):
#   - 蒸留済み (ponkotsu 再評価済み 88.0 億局面 / 352GB) -> teachers/distilled/
#   - 元ラベル (蒸留元 dataset。パイロット run C の対照用 3 ファイル / 59GB) -> teachers/orig/
#
# vast テンプレートの Environment Variables (全て省略可):
#   DISTILLED_DATASET : 蒸留済み教師データの HF dataset
#                       既定 ngs436/dlsuisho-ponkotsu-distilled
#   SKIP_DISTILLED    : 1 で蒸留済みデータの DL をスキップ
#   ORIG_DATASET      : 元ラベル (蒸留元) の HF dataset
#                       既定 washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique
#   ORIG_INCLUDE      : 元ラベルから取得するファイル (空白/カンマ区切り、glob 可)。
#                       既定はパイロット run C 用の dlsuisho_unique_001..003.bin。
#                       残り 40% の追い蒸留 (rescore) をやる場合は 019..030 を指定、
#                       全量なら '*.bin' (587GB)。空文字で DL しない
#   SKIP_ORIG         : 1 で元ラベルデータの DL をスキップ
#   HF_POOL           : 1 で add_aobazero プール (HF_DATASET) を全量 DL する
#                       (既定オフ。将来の教師データ増強用に経路だけ残置)
#   HF_DATASET        : HF_POOL=1 のときの対象 dataset
#                       既定 washiun/Knowledge_distilled_by_DLSuisho15b_add_aobazero_unique
#   GIT_TOKEN         : GitHub PAT (省略可。対象 repo は全て public)
#   RSHOGI_BRANCH     : 既定 claude/busy-faraday-umwgl8
#   TATARA_BRANCH     : 既定 main (net_to_yo 汎用化を使うなら claude/net-to-yo-dims-generic)
#   SHOGITEST_BRANCH  : 既定 claude/nightly-toolchain-pin (main へ merge 済みなら main)

set -uo pipefail
mkdir -p /workspace/.onstart
LOG=/workspace/onstart.log
exec >>"$LOG" 2>&1
echo "===== onstart $(date -u +%FT%TZ) ====="

export WORK=/workspace
export SHOGI_DATA=${SHOGI_DATA:-/workspace/shogi-data}
export RUSTUP_HOME=${RUSTUP_HOME:-/opt/rustup} CARGO_HOME=${CARGO_HOME:-/opt/cargo}
export PATH=/opt/cargo/bin:/usr/local/cuda/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=1

RSHOGI_BRANCH=${RSHOGI_BRANCH:-claude/busy-faraday-umwgl8}
TATARA_BRANCH=${TATARA_BRANCH:-main}
SHOGITEST_BRANCH=${SHOGITEST_BRANCH:-claude/nightly-toolchain-pin}
DISTILLED_DATASET=${DISTILLED_DATASET:-ngs436/dlsuisho-ponkotsu-distilled}
ORIG_DATASET=${ORIG_DATASET:-washiun/Knowledge_distilled_dataset_by_DLSuisho15b_unique}
ORIG_INCLUDE=${ORIG_INCLUDE:-dlsuisho_unique_001.bin dlsuisho_unique_002.bin dlsuisho_unique_003.bin}
HF_DATASET=${HF_DATASET:-washiun/Knowledge_distilled_by_DLSuisho15b_add_aobazero_unique}

# UI 経由で混入しがちな引用符の除去とカンマ→スペース正規化
ORIG_INCLUDE=$(printf '%s' "$ORIG_INCLUDE" | tr -d '"' | tr -d "'" | tr ',' ' ')

mkdir -p "$SHOGI_DATA"/{teachers/distilled,teachers/orig,teachers/pool,nnue,progress} \
         "$WORK"/{pilot,logs,book,trt_g0} "$WORK/pilot"/{rescored,logs}

service ssh start 2>/dev/null || true

# ---------- helpers ----------
marker() { echo "/workspace/.onstart/$1.done"; }
step_done() { [ -f "$(marker "$1")" ]; }
mark() { date -u +%FT%TZ > "$(marker "$1")"; echo "[onstart] step '$1' done"; }

clone_repo() { # name branch
    local name=$1 branch=$2 dir="$WORK/$1"
    if [ -d "$dir/.git" ]; then
        git -C "$dir" fetch origin "$branch" && git -C "$dir" checkout "$branch" \
            && git -C "$dir" pull --ff-only origin "$branch" || true
        return 0
    fi
    local url="https://github.com/keinoda/${name}"
    if [ -n "${GIT_TOKEN:-}" ]; then url="https://${GIT_TOKEN}@github.com/keinoda/${name}"; fi
    git clone --branch "$branch" "$url" "$dir" || git clone "$url" "$dir" || return 1
    # token を .git/config に残さない
    git -C "$dir" remote set-url origin "https://github.com/keinoda/${name}"
    git -C "$dir" checkout "$branch" || true
}

# ---------- 1. repos ----------
if ! step_done repos; then
    clone_repo rshogi "$RSHOGI_BRANCH" \
      && clone_repo tatara "$TATARA_BRANCH" \
      && clone_repo shogitest "$SHOGITEST_BRANCH" \
      && mark repos
fi

# ---------- 2.1 蒸留済み教師データ (HF、全量 352GB) ----------
# .bin 19 本 (dlsuisho_unique_001..019) が揃って完了。アップロード途中の dataset を
# 引いた場合は marker を置かず警告する (onstart 再実行で hf download が差分 resume する)。
if [ "${SKIP_DISTILLED:-0}" != "1" ] && ! step_done distdl && ! tmux has-session -t distdl 2>/dev/null; then
    tmux new-session -d -s distdl "( hf download '$DISTILLED_DATASET' --repo-type dataset \
        --local-dir '$SHOGI_DATA/teachers/distilled' && \
        N=\$(find '$SHOGI_DATA/teachers/distilled' -maxdepth 1 -name '*.bin' | wc -l) && \
        echo \"[distdl] .bin: \$N 本\" && \
        if [ \"\$N\" -ge 19 ]; then touch '$(marker distdl)'; else \
          echo '[distdl] WARNING: .bin が 19 本未満。HF 側のアップロード完了を確認して onstart を再実行すること'; fi \
        ) 2>&1 | tee '$WORK/logs/distdl.log'; sleep 5"
    echo "[onstart] distilled download started ($DISTILLED_DATASET)"
fi

# ---------- 2.2 元ラベル教師データ (HF、run C 対照用) ----------
if [ "${SKIP_ORIG:-0}" != "1" ] && [ -n "$ORIG_INCLUDE" ] && ! step_done origdl && ! tmux has-session -t origdl 2>/dev/null; then
    tmux new-session -d -s origdl "( hf download '$ORIG_DATASET' --repo-type dataset \
        --include $ORIG_INCLUDE \
        --local-dir '$SHOGI_DATA/teachers/orig' && \
        touch '$(marker origdl)' ) 2>&1 | tee '$WORK/logs/origdl.log'; sleep 5"
    echo "[onstart] orig-label download started ($ORIG_DATASET: $ORIG_INCLUDE)"
fi

# ---------- 2.4 add_aobazero プール全量 DL (HF_POOL=1 のときのみ、既定オフ) ----------
# 将来の教師データ増強用に経路だけ残置。全量 679GB のためディスクは 2TB 級が必要。
if [ "${HF_POOL:-0}" = "1" ] && ! step_done hfdl && ! tmux has-session -t hfdl 2>/dev/null; then
    tmux new-session -d -s hfdl "( hf download '$HF_DATASET' --repo-type dataset \
        --local-dir '$SHOGI_DATA/teachers/pool' && \
        touch '$(marker hfdl)' ) 2>&1 | tee '$WORK/logs/hfdl.log'; sleep 5"
    echo "[onstart] hf pool download started ($HF_DATASET)"
fi

# ---------- 2.5 progress.bin (keinoda/yaneuraou sojo_tsec7 branch の source/ から取得) ----------
# 学習 (tatara --progress-coeff) とエンジン (LS_PROGRESS_COEFF) の両方で同一ファイルを使う。
if [ ! -s "$SHOGI_DATA/progress/progress.bin" ]; then
    if wget -q -O "$SHOGI_DATA/progress/progress.bin.tmp" \
        "https://raw.githubusercontent.com/keinoda/yaneuraou/sojo_tsec7/source/progress.bin"; then
        mv "$SHOGI_DATA/progress/progress.bin.tmp" "$SHOGI_DATA/progress/progress.bin"
        SIZE=$(stat -c%s "$SHOGI_DATA/progress/progress.bin")
        # progress8kpabs 係数は f64 LE x 125388 = 1,003,104 bytes (tatara docs/progress-bin)
        if [ "$SIZE" != "1003104" ]; then
            echo "[onstart] WARNING: progress.bin size=$SIZE (expected 1003104) — 形式を確認すること"
        else
            echo "[onstart] progress.bin fetched ($SIZE bytes)"
        fi
    else
        rm -f "$SHOGI_DATA/progress/progress.bin.tmp"
        echo "[onstart] WARNING: progress.bin の取得に失敗 (手動で配置すること)"
    fi
fi

# ---------- 3. rshogi build (tmux): rescore/検証ツール + パイロット用 engine 2 種 ----------
if ! step_done build_rshogi && ! tmux has-session -t build_rshogi 2>/dev/null; then
    tmux new-session -d -s build_rshogi "( cd '$WORK/rshogi' && \
        cargo build --release -p tools --features dlshogi-onnx --bin rescore_psv && \
        cargo build --release -p tools --bin validate_psv --bin psv_to_jsonl --bin split_psv --bin shuffle_psv && \
        cargo build --profile production -p rshogi-usi --no-default-features \
          --features search-no-pass-rules,edition-layerstacks-halfka_hm_merged-3072x16x64-none && \
        cp target/production/rshogi-usi '$WORK/pilot/rshogi-3072' && \
        cargo build --profile production -p rshogi-usi --no-default-features \
          --features search-no-pass-rules,edition-layerstacks-halfka_hm_merged-1536x16x32-none && \
        cp target/production/rshogi-usi '$WORK/pilot/rshogi-1536' && \
        touch '$(marker build_rshogi)'; \
        echo exit=\$? ) 2>&1 | tee '$WORK/logs/build_rshogi.log'; sleep 5"
    echo "[onstart] rshogi build started"
fi

# ---------- 4. tatara build (tmux) ----------
# 正規手順は 3 段 (tatara docs/setup.ja.md):
#   1. setup-cuda-oxide.sh — kernel ビルドツール cargo-oxide を Cargo.lock の
#      pin rev で install し、codegen backend cache も揃える
#   2. build-kernels.sh    — GPU kernel (.ll) をビルド。GPU 世代は nvidia-smi で
#      自動判定 (Turing のみ CUDA_OXIDE_TARGET=sm_75 を自動設定)
#   3. cargo build         — host バイナリ
# 2 を飛ばすと nnue-train が起動直後 (SB0) に kernel 不在で落ちる。
if ! step_done build_tatara && ! tmux has-session -t build_tatara 2>/dev/null; then
    tmux new-session -d -s build_tatara "( cd '$WORK/tatara' && \
        bash scripts/setup-cuda-oxide.sh && \
        bash scripts/build-kernels.sh && \
        cargo build --release && \
        touch '$(marker build_tatara)'; echo exit=\$? ) 2>&1 | tee '$WORK/logs/build_tatara.log'; sleep 5"
    echo "[onstart] tatara build started"
fi

# ---------- 5. shogitest build (tmux) ----------
if ! step_done build_shogitest && ! tmux has-session -t build_shogitest 2>/dev/null; then
    tmux new-session -d -s build_shogitest "( cd '$WORK/shogitest' && \
        cargo install --path . --root /opt/cargo && \
        touch '$(marker build_shogitest)'; echo exit=\$? ) 2>&1 | tee '$WORK/logs/build_shogitest.log'; sleep 5"
    echo "[onstart] shogitest build started"
fi

# ---------- 6. サマリ ----------
cat <<EOF
[onstart] kicked. 状態確認:
  tail -f /workspace/onstart.log
  tmux ls                      # distdl / origdl / build_rshogi / build_tatara / build_shogitest
  ls /workspace/.onstart/      # 完了 marker
  tail -f /workspace/logs/distdl.log   # 蒸留済みデータ DL の進行
残る手動ステップ (イメージ/onstart に入れられないもの):
  1. 開始局面集 -> $WORK/book/openings.epd (scp)
  2. ponkotsu.onnx -> $SHOGI_DATA/nnue/ (scp。追い蒸留するときのみ必要)
EOF
echo "===== onstart end $(date -u +%FT%TZ) ====="
