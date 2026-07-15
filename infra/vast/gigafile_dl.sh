#!/usr/bin/env bash
# gigafile.nu の共有リンクからダウンロードする (DownloadGigafile.ps1 の bash 移植 + 並列化)。
#
# 仕組み (ps1 と同一): ダウンロードページを GET して session cookie を取得し、
# 同セッションで https://<server>/download.php?file=<id>[&dlkey=...] を GET する。
#
# 高速化: gigafile は国内向けサービスで、海外 DC からは 1 接続あたり ~200kB/s に
# 制限される (実測。並列接続してもコネクション毎の帯域は落ちない = 並列が線形に効く)。
# そのため aria2c があれば分割並列 (-x N)、無ければ curl 単一接続に fallback する。
# 複数 URL は別サーバに載っているため同時にダウンロードする (既定 4 並列)。
#
# 使い方:
#   gigafile_dl.sh [-o 出力dir] [-k ダウンロードキー] [-x 接続数/URL] [-j URL並列数] <URL>...
# 例:
#   gigafile_dl.sh -o "$SHOGI_DATA/teachers/distilled" \
#     "https://115.gigafile.nu/1012-xxxxxxxxxxxxxxxx" "https://121.gigafile.nu/yyyy"
#
# 注意:
#   - 複数ファイルの「まとめページ」は非対応 (単一ファイルページの URL を個別に渡す)
#   - gigafile のリンクには保持期限がある。受領したら速やかに落とすこと
#   - 接続数の既定は 8/URL × 4 URL = 計 32。無料サービスなのでこれ以上は上げないこと

set -uo pipefail

UA="Mozilla/5.0 (X11; Linux x86_64) curl-gigafile-downloader"
OUTPUT_DIR="."
DLKEY=""
CONNS=8
PAR=0 # 0 = 自動 (URL 数と 4 の小さい方)

usage() { echo "usage: $0 [-o output_dir] [-k download_key] [-x conns_per_url] [-j parallel_urls] <url>..." >&2; }

while getopts "o:k:x:j:" opt; do
    case "$opt" in
        o) OUTPUT_DIR="$OPTARG" ;;
        k) DLKEY="$OPTARG" ;;
        x) CONNS="$OPTARG" ;;
        j) PAR="$OPTARG" ;;
        *) usage; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
    usage
    exit 2
fi

if [ "$PAR" -le 0 ]; then
    PAR=$#
    [ "$PAR" -gt 4 ] && PAR=4
fi

HAVE_ARIA2=0
command -v aria2c >/dev/null 2>&1 && HAVE_ARIA2=1
if [ "$HAVE_ARIA2" -eq 0 ]; then
    echo "NOTE: aria2c が見つからないため curl 単一接続で実行する。" >&2
    echo "      海外リージョンでは著しく遅いので 'apt-get install -y aria2' を推奨。" >&2
fi

mkdir -p "$OUTPUT_DIR"

# 単一接続 + resume の curl fallback。全量取得済み (exit 33 / HTTP 416) は完了扱い
curl_download() { # jar dl_url out_path
    local jar=$1 dl_url=$2 out_path=$3 rc attempt
    for attempt in $(seq 1 10); do
        echo "curl ダウンロード中... (attempt $attempt, resume 有効)"
        curl -L -A "$UA" -b "$jar" -C - --retry 3 --retry-delay 10 -o "$out_path" "$dl_url"
        rc=$?
        if [ $rc -eq 0 ] || [ $rc -eq 33 ]; then return 0; fi
        echo "WARNING: curl exit=$rc — 20 秒後に resume でリトライ" >&2
        sleep 20
    done
    return 1
}

download_one() { # url
    local url=$1
    echo "=== $url ==="

    if [[ ! "$url" =~ ^https?://([0-9]+\.gigafile\.nu)/([a-zA-Z0-9-]+)/?$ ]]; then
        echo "WARNING: URL の形式が正しくありません (例: https://115.gigafile.nu/1012-xxxx): $url" >&2
        return 1
    fi
    local server="${BASH_REMATCH[1]}" file_id="${BASH_REMATCH[2]}"

    local jar page
    jar=$(mktemp)
    page=$(mktemp)

    # 1. ページを GET して session cookie を取得
    if ! curl -fsSL -A "$UA" -c "$jar" -o "$page" "$url"; then
        echo "WARNING: ページの取得に失敗: $url" >&2
        rm -f "$jar" "$page"
        return 1
    fi

    if grep -q 'id="contents_matomete"' "$page"; then
        echo "WARNING: 複数ファイルのまとめページのようです。単一ファイルページの URL を渡すこと" >&2
    fi

    # 2. ファイル名 (id="dl" 要素のテキスト)。実ページは要素内で改行するため、
    #    1 行に潰してから抽出する (grep は行単位のため)。取れなければ file_id を使う
    local file_name safe_name out_path size_text
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
        return 0
    fi

    size_text=$(tr -d '\n\r' < "$page" | grep -oP 'dl_size[^"]*"[^>]*>\s*\K[^<]+' | head -1 | sed 's/[[:space:]]*$//;s/^[[:space:]]*//' || true)
    echo "ファイル名: $file_name"
    [ -n "$size_text" ] && echo "サイズ: $size_text"
    echo "保存先: $out_path"

    local dl_url="https://${server}/download.php?file=${file_id}"
    if [ -n "$DLKEY" ]; then dl_url="${dl_url}&dlkey=${DLKEY}"; fi

    # 3. ダウンロード: aria2c 分割並列 → 失敗時は curl (resume) に fallback
    local ok=0
    if [ "$HAVE_ARIA2" -eq 1 ]; then
        echo "aria2c 分割並列ダウンロード (-x $CONNS)"
        if aria2c --load-cookies "$jar" -c -x "$CONNS" -s "$CONNS" -k 1M \
            --file-allocation=none --auto-file-renaming=false --user-agent "$UA" \
            --max-tries=10 --retry-wait=20 \
            --show-console-readout=false --summary-interval=30 --console-log-level=notice \
            -d "$OUTPUT_DIR" -o "$safe_name" "$dl_url"; then
            ok=1
        else
            # 既に全量取得済みのファイルへの -c は Range 416 でエラーになるため、
            # 416/33 を完了扱いにできる curl 側で確定させる
            echo "WARNING: aria2c 失敗 — curl (単一接続, resume) で継続を試みる" >&2
        fi
    fi
    if [ $ok -eq 0 ]; then
        curl_download "$jar" "$dl_url" "$out_path" && ok=1
    fi

    rm -f "$jar" "$page"

    if [ $ok -eq 1 ] && [ -s "$out_path" ]; then
        echo "完了: $out_path ($(stat -c%s "$out_path") bytes)"
        return 0
    fi
    echo "WARNING: 失敗しました: $url" >&2
    return 1
}

# URL を PAR 本ずつ並列にダウンロードする (それぞれ別サーバのため合算で速くなる)
FAILED=0
urls=("$@")
idx=0
total=${#urls[@]}
while [ "$idx" -lt "$total" ]; do
    pids=()
    batch=()
    while [ "$idx" -lt "$total" ] && [ ${#pids[@]} -lt "$PAR" ]; do
        url="${urls[$idx]}"
        idx=$((idx + 1))
        label=$(printf '%s' "$url" | sed -E 's|^https?://([0-9]+)\..*|gf\1|')
        ( download_one "$url" 2>&1 | sed -u "s|^|[$label] |"; exit "${PIPESTATUS[0]}" ) &
        pids+=($!)
        batch+=("$url")
    done
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "WARNING: 失敗しました: ${batch[$i]}" >&2
            FAILED=1
        fi
    done
done

exit $FAILED
