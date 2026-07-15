#!/usr/bin/env bash
# gigafile.nu の共有リンクからダウンロードする (DownloadGigafile.ps1 の bash 移植)。
#
# 仕組み (ps1 と同一): ダウンロードページを GET して session cookie を取得し、
# 同セッションで https://<server>/download.php?file=<id>[&dlkey=...] を GET する。
# 大容量向けに resume (curl -C -) とリトライを追加している。
#
# 使い方:
#   gigafile_dl.sh -o <出力dir> [-k <ダウンロードキー>] <URL> [<URL>...]
# 例:
#   gigafile_dl.sh -o "$SHOGI_DATA/teachers/distilled" \
#     "https://115.gigafile.nu/1012-xxxxxxxxxxxxxxxx" "https://115.gigafile.nu/yyyy"
#
# 注意:
#   - 複数ファイルの「まとめページ」は非対応 (単一ファイルページの URL を個別に渡す)
#   - gigafile のリンクには保持期限がある。受領したら速やかに落とすこと

set -uo pipefail

UA="Mozilla/5.0 (X11; Linux x86_64) curl-gigafile-downloader"
OUTPUT_DIR="."
DLKEY=""

while getopts "o:k:" opt; do
    case "$opt" in
        o) OUTPUT_DIR="$OPTARG" ;;
        k) DLKEY="$OPTARG" ;;
        *) echo "usage: $0 [-o output_dir] [-k download_key] <url>..." >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
    echo "usage: $0 [-o output_dir] [-k download_key] <url>..." >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
FAILED=0

for url in "$@"; do
    echo ""
    echo "=== $url ==="

    if [[ ! "$url" =~ ^https?://([0-9]+\.gigafile\.nu)/([a-zA-Z0-9-]+)/?$ ]]; then
        echo "WARNING: URL の形式が正しくありません (例: https://115.gigafile.nu/1012-xxxx): $url" >&2
        FAILED=1
        continue
    fi
    server="${BASH_REMATCH[1]}"
    file_id="${BASH_REMATCH[2]}"

    jar=$(mktemp)
    page=$(mktemp)

    # 1. ページを GET して session cookie を取得
    if ! curl -fsSL -A "$UA" -c "$jar" -o "$page" "$url"; then
        echo "WARNING: ページの取得に失敗: $url" >&2
        rm -f "$jar" "$page"
        FAILED=1
        continue
    fi

    if grep -q 'id="contents_matomete"' "$page"; then
        echo "WARNING: 複数ファイルのまとめページのようです。単一ファイルページの URL を渡すこと" >&2
    fi

    # 2. ファイル名 (id="dl" 要素のテキスト)。実ページは要素内で改行するため、
    #    1 行に潰してから抽出する (grep は行単位のため)。取れなければ file_id を使う
    file_name=$(tr -d '\n\r' < "$page" | grep -oP 'id="dl"[^>]*>\s*\K[^<]+' | head -1 | sed 's/[[:space:]]*$//;s/^[[:space:]]*//')
    if [ -z "$file_name" ]; then
        echo "WARNING: ファイル名を取得できませんでした。ファイル ID を名前として使用します" >&2
        file_name="$file_id"
    fi
    safe_name=$(printf '%s' "$file_name" | tr '\\/:*?"<>|' '_')
    out_path="$OUTPUT_DIR/$safe_name"

    # 展開・検証済み (extract_stored_zips.py のスタンプ) なら再ダウンロードしない
    if [ -f "${out_path}.extracted" ]; then
        echo "展開済みスタンプあり、ダウンロードを skip: $safe_name"
        rm -f "$jar" "$page"
        continue
    fi

    size_text=$(tr -d '\n\r' < "$page" | grep -oP 'dl_size[^"]*"[^>]*>\s*\K[^<]+' | head -1 | sed 's/[[:space:]]*$//;s/^[[:space:]]*//' || true)
    echo "ファイル名: $file_name"
    [ -n "$size_text" ] && echo "サイズ: $size_text"
    echo "保存先: $out_path"

    dl_url="https://${server}/download.php?file=${file_id}"
    if [ -n "$DLKEY" ]; then dl_url="${dl_url}&dlkey=${DLKEY}"; fi

    # 3. resume 付きダウンロード (最大 10 回再開を試みる)
    ok=0
    for attempt in $(seq 1 10); do
        echo "ダウンロード中... (attempt $attempt, resume 有効)"
        curl -L -A "$UA" -b "$jar" -C - --retry 3 --retry-delay 10 -o "$out_path" "$dl_url"
        rc=$?
        # 0 = 完了。33 / 416 系 = 既に全量取得済みで resume 不要 → 完了扱い
        if [ $rc -eq 0 ] || [ $rc -eq 33 ]; then
            ok=1
            break
        fi
        echo "WARNING: curl exit=$rc — 20 秒後に resume でリトライ" >&2
        sleep 20
    done

    rm -f "$jar" "$page"

    if [ $ok -eq 1 ] && [ -s "$out_path" ]; then
        echo "完了: $out_path ($(stat -c%s "$out_path") bytes)"
    else
        echo "WARNING: 失敗しました: $url" >&2
        FAILED=1
    fi
done

exit $FAILED
