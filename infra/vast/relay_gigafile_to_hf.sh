#!/usr/bin/env bash
# gigafile → HuggingFace リレー転送 (国内 VM 用)。
#
# gigafile.nu は海外 DC から 1 接続 ~200kB/s に制限されるため、国内の VM
# (OCI 東京の Always Free 等) で受けて HF dataset へ退避するためのスクリプト。
# ディスクを節約するため、zip 全展開はせず「zip 1 本 DL → メンバーを 1 ファイル
# ずつ展開 → HF へ upload → 削除」で回す。ピークディスク ≈ zip 1 本 (98GB) +
# メンバー 1 本 (20GB) ≈ 120GB なので、無料枠の 200GB boot volume に収まる。
#
# 前提:
#   sudo apt-get install -y aria2 unzip git python3 tmux
#   pip3 install -U 'huggingface_hub[hf_transfer]' --break-system-packages
#   hf auth login           # Write 権限のトークン
#   hf repo create <repo> --repo-type dataset [--private]
#
# 使い方:
#   relay_gigafile_to_hf.sh -r <user/dataset> [-w workdir] [-k dlkey] [-x conns] <URL>[,<URL>...]
# 例:
#   tmux new -s relay
#   bash rshogi/infra/vast/relay_gigafile_to_hf.sh -r foo/dlsuisho-ponkotsu \
#     'https://115.gigafile.nu/xxx,https://121.gigafile.nu/yyy'
#
# 冪等性 / resume:
#   - アップロード済みメンバーは ledger (workdir/uploaded.txt) に記録して skip
#   - 処理し終えた zip は <zip>.extracted スタンプを残して削除し、再実行時は
#     gigafile_dl.sh がスタンプを見て再ダウンロードしない
#   - 中断後は同じコマンドの再実行で続きから進む

set -uo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=""
WORKDIR="$HOME/relay-work"
DLKEY=""
CONNS=8
PSV_RECORD=40

usage() { echo "usage: $0 -r <user/dataset> [-w workdir] [-k dlkey] [-x conns] <url>[,<url>...]" >&2; }

while getopts "r:w:k:x:" opt; do
    case "$opt" in
        r) REPO="$OPTARG" ;;
        w) WORKDIR="$OPTARG" ;;
        k) DLKEY="$OPTARG" ;;
        x) CONNS="$OPTARG" ;;
        *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [ -z "$REPO" ] || [ $# -eq 0 ]; then
    usage
    exit 2
fi

for cmd in unzip hf python3; do
    command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: $cmd が必要です (ヘッダの前提を参照)" >&2; exit 1; }
done

# URL はカンマ区切り / 空白区切りの両方を受ける
URLS=$(printf '%s ' "$@" | tr -d '"' | tr -d "'" | tr ',' ' ')

mkdir -p "$WORKDIR/x"
LEDGER="$WORKDIR/uploaded.txt"
touch "$LEDGER"

upload_one() { # local_path remote_name
    local path=$1 name=$2 attempt
    for attempt in 1 2 3; do
        if hf upload "$REPO" "$path" "$name" --repo-type dataset --quiet; then
            return 0
        fi
        echo "WARNING: hf upload 失敗 ($name, attempt $attempt) — 30 秒後にリトライ" >&2
        sleep 30
    done
    return 1
}

process_zip() { # zip_path
    local zip=$1
    local stamp="${zip}.extracted"
    local members m base out size
    members=$(unzip -Z1 "$zip" | grep -v '/$') || { echo "ERROR: メンバー一覧の取得に失敗: $zip" >&2; return 1; }
    while IFS= read -r m; do
        base=$(basename "$m")
        [ -z "$base" ] && continue
        if grep -qxF "$base" "$LEDGER"; then
            echo "[relay] アップロード済みを skip: $base"
            continue
        fi
        echo "[relay] 展開: $base"
        rm -f "$WORKDIR/x/$base"
        # unzip は展開時に CRC を検証する (不一致なら非ゼロ exit)
        if ! unzip -q -o -j "$zip" "$m" -d "$WORKDIR/x"; then
            echo "ERROR: 展開に失敗 (CRC?): $m" >&2
            return 1
        fi
        out="$WORKDIR/x/$base"
        size=$(stat -c%s "$out")
        if [[ "$base" == *.bin ]] && [ $((size % PSV_RECORD)) -ne 0 ]; then
            echo "ERROR: $base が ${PSV_RECORD}B レコード境界でない ($size bytes)" >&2
            return 1
        fi
        echo "[relay] upload: $base ($size bytes) -> $REPO"
        if ! upload_one "$out" "$base"; then
            echo "ERROR: アップロードに失敗: $base (再実行で resume 可)" >&2
            return 1
        fi
        echo "$base" >> "$LEDGER"
        rm -f "$out"
    done <<< "$members"
    unzip -Z1 "$zip" | grep -v '/$' | sed 's|.*/||' > "$stamp"
    rm -f "$zip"
    echo "[relay] zip 完了・削除: $(basename "$zip")"
    return 0
}

FAILED=0
for url in $URLS; do
    echo ""
    echo "===== $url ====="
    # 1 本ずつ DL (ディスク節約のため URL 並列にはしない。国内なら十分速い)
    if ! bash "$SCRIPT_DIR/gigafile_dl.sh" -o "$WORKDIR" ${DLKEY:+-k "$DLKEY"} -x "$CONNS" "$url"; then
        echo "WARNING: ダウンロード失敗: $url" >&2
        FAILED=1
        continue
    fi
    # このターンで増えた未処理 zip を処理 (スタンプ済みは対象外)
    found=0
    for zip in "$WORKDIR"/*.zip; do
        [ -e "$zip" ] || continue
        [ -f "${zip}.extracted" ] && continue
        found=1
        process_zip "$zip" || FAILED=1
    done
    [ $found -eq 0 ] && echo "[relay] 未処理 zip なし (スタンプ済み or DL skip)"
done

echo ""
echo "[relay] 完了。アップロード済み: $(wc -l < "$LEDGER") ファイル ($LEDGER)"
echo "[relay] 確認: hf download $REPO --repo-type dataset --include '*.done' --local-dir /tmp/chk"
exit $FAILED
