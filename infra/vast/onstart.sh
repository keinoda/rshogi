#!/usr/bin/env bash
# vast.ai onstart — 課金開始直後に「イメージに焼けないもの」を全自動で準備する。
# 冪等: 再起動時は完了済みステップを skip する。進捗は /workspace/onstart.log と
# marker (/workspace/.onstart/) で確認。ビルド/DL は tmux セッションで並列に走る。
#
# vast テンプレートの Environment Variables で挙動を制御する:
#   GIT_TOKEN        : GitHub PAT (省略可)。対象 4 repo (rshogi / tatara / shogitest /
#                      yaneuraou) は全て public のため通常は不要。private repo を使う
#                      構成に変えた場合のみ read-only fine-grained PAT を設定する
#   RSHOGI_BRANCH    : 既定 claude/busy-faraday-umwgl8
#   TATARA_BRANCH    : 既定 main (net_to_yo 汎用化を使うなら claude/net-to-yo-dims-generic)
#   SHOGITEST_BRANCH : 既定 claude/nightly-toolchain-pin (main へ merge 済みなら main)
#   HF_DATASET       : 既定 washiun/Knowledge_distilled_by_DLSuisho15b_add_aobazero_unique
#   SKIP_HF          : 1 で HF プールの DL をスキップ (蒸留済みデータのみで作業する場合)
#   GIGAFILE_URLS    : gigafile.nu の URL (空白区切りで複数可)。指定時は蒸留済み教師データを
#                      $SHOGI_DATA/teachers/distilled/ へダウンロードする
#   GIGAFILE_DLKEY   : gigafile のダウンロードキー (設定されている場合のみ)
#
# 教師データは常に全量 (34 shard / 679GB) をダウンロードする。実績あるデータセットで
# 全量使うことが確定しているため、shard 小出しで後から待つ時間を作らない。
# 止めたい場合は `tmux kill-session -t hfdl`。中断後の再実行は resume される。

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
HF_DATASET=${HF_DATASET:-washiun/Knowledge_distilled_by_DLSuisho15b_add_aobazero_unique}

mkdir -p "$SHOGI_DATA"/{teachers/pool,nnue,progress} \
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

# ---------- 2. 教師データ DL (全量、tmux で並列) ----------
if [ "${SKIP_HF:-0}" != "1" ] && ! step_done hfdl && ! tmux has-session -t hfdl 2>/dev/null; then
    tmux new-session -d -s hfdl "hf download '$HF_DATASET' --repo-type dataset \
        --local-dir '$SHOGI_DATA/teachers/pool' \
        2>&1 | tee '$WORK/logs/hfdl.log'; \
        touch '$(marker hfdl)'"
    echo "[onstart] hf download started (full dataset)"
fi

# ---------- 2.3 蒸留済み教師データ (gigafile.nu、GIGAFILE_URLS 指定時) ----------
# gigafile のリンクには保持期限があるため、受領したら速やかに落とすこと。
if [ -n "${GIGAFILE_URLS:-}" ] && ! step_done gigafile && ! tmux has-session -t gigafile 2>/dev/null; then
    mkdir -p "$SHOGI_DATA/teachers/distilled"
    tmux new-session -d -s gigafile "( bash '$WORK/rshogi/infra/vast/gigafile_dl.sh' \
        -o '$SHOGI_DATA/teachers/distilled' ${GIGAFILE_DLKEY:+-k \"\$GIGAFILE_DLKEY\"} \
        \$GIGAFILE_URLS && touch '$(marker gigafile)' ) 2>&1 | tee '$WORK/logs/gigafile.log'; sleep 5"
    echo "[onstart] gigafile download started"
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
if ! step_done build_tatara && ! tmux has-session -t build_tatara 2>/dev/null; then
    # Turing (sm_75) のみ CUDA_OXIDE_TARGET 指定が必要 (tatara docs/setup.ja.md)
    CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    TARGET_ENV=""
    if [ -n "$CC" ] && [ "$CC" -lt 80 ]; then TARGET_ENV="CUDA_OXIDE_TARGET=sm_75"; fi
    tmux new-session -d -s build_tatara "( cd '$WORK/tatara' && \
        env $TARGET_ENV cargo build --release && \
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
  tmux ls                      # hfdl / build_rshogi / build_tatara / build_shogitest
  ls /workspace/.onstart/      # 完了 marker
残る手動ステップ (イメージ/onstart に入れられないもの):
  1. ponkotsu.onnx  -> $SHOGI_DATA/nnue/       (scp)
  2. 開始局面集      -> $WORK/book/openings.epd (scp)
EOF
echo "===== onstart end $(date -u +%FT%TZ) ====="
