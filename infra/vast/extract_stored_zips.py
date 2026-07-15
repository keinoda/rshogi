#!/usr/bin/env python3
"""gigafile で受領した教師データ zip (STORED) を展開・検証し、zip を削除する。

<dir> 直下の *.zip を名前順に処理する:

1. 各メンバーを <dir> 直下へ flatten して streaming 展開する
   (`ZipExtFile` が読み取り時に CRC を検証するため、読み切れた時点で
   payload の整合性が保証される。zip64 対応も zipfile 任せで安全)
2. 展開後サイズが central directory の宣言サイズと一致することを確認する
3. `*.bin` は PSV レコード境界 (40 bytes) に載っていることを確認する
4. 全メンバー OK なら `<zip>.extracted` スタンプ (メンバー名とサイズの一覧) を
   書いてから zip を削除する (`--remove-zips` 指定時)

冪等性 / resume:
- スタンプがある zip は展開せず skip (gigafile_dl.sh もスタンプを見て再 DL しない)
- 展開先に同サイズのファイルが既にあればそのメンバーは skip
- 書き込みは `.part` → rename なので、中断で壊れた中間ファイルを誤認しない
- 検証に失敗した zip は削除しない (gigafile リンクには保持期限があり、
  再ダウンロードできなくなる恐れがあるため)

使い方:
    extract_stored_zips.py <dir> [--remove-zips]
"""

import argparse
import os
import shutil
import sys
import zipfile

PSV_RECORD = 40
CHUNK = 8 * 1024 * 1024


def _extract_member(z: zipfile.ZipFile, info: zipfile.ZipInfo, dst: str) -> None:
    tmp = dst + ".part"
    with z.open(info) as src, open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, CHUNK)
    os.replace(tmp, dst)


def extract_zip(zip_path: str, out_dir: str, remove: bool) -> bool:
    """1 つの zip を展開・検証する。成功時 True。"""
    name = os.path.basename(zip_path)
    stamp = zip_path + ".extracted"
    if os.path.exists(stamp):
        print(f"[extract] スタンプあり、展開済みとして skip: {name}")
        if remove and os.path.exists(zip_path):
            os.remove(zip_path)
        return True

    lines = []
    try:
        with zipfile.ZipFile(zip_path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                base = os.path.basename(info.filename)
                if not base:
                    continue
                dst = os.path.join(out_dir, base)
                if os.path.exists(dst) and os.path.getsize(dst) == info.file_size:
                    print(f"[extract] 既存 (サイズ一致) を skip: {base}")
                else:
                    if info.compress_type != zipfile.ZIP_STORED:
                        print(
                            f"[extract] NOTE: {base} は STORED でない "
                            f"(compress_type={info.compress_type}) — そのまま展開する"
                        )
                    print(f"[extract] 展開中: {base} ({info.file_size:,} bytes)")
                    _extract_member(z, info, dst)
                actual = os.path.getsize(dst)
                if actual != info.file_size:
                    print(
                        f"[extract] ERROR: {base} サイズ不一致 "
                        f"actual={actual} expected={info.file_size}",
                        file=sys.stderr,
                    )
                    return False
                if base.endswith(".bin") and actual % PSV_RECORD != 0:
                    print(
                        f"[extract] ERROR: {base} が {PSV_RECORD}B レコード境界でない "
                        f"({actual} bytes)",
                        file=sys.stderr,
                    )
                    return False
                lines.append(f"{base}\t{actual}")
    except (zipfile.BadZipFile, OSError) as e:
        print(f"[extract] ERROR: {name}: {e}", file=sys.stderr)
        return False

    with open(stamp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if remove:
        os.remove(zip_path)
        print(f"[extract] 検証 OK、zip 削除: {name}")
    else:
        print(f"[extract] 検証 OK: {name}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="STORED zip を flatten 展開・検証して削除する (gigafile 教師データ用)"
    )
    ap.add_argument("dir", help="zip の置き場所 = 展開先ディレクトリ")
    ap.add_argument(
        "--remove-zips",
        action="store_true",
        help="検証成功した zip を削除する (ピークディスク削減)",
    )
    args = ap.parse_args()

    zips = sorted(
        os.path.join(args.dir, n)
        for n in os.listdir(args.dir)
        if n.lower().endswith(".zip")
    )
    ok = True
    for zp in zips:
        ok = extract_zip(zp, args.dir, args.remove_zips) and ok

    total_bytes = 0
    n_bins = 0
    for n in sorted(os.listdir(args.dir)):
        if n.endswith(".bin"):
            total_bytes += os.path.getsize(os.path.join(args.dir, n))
            n_bins += 1
    print(
        f"[extract] dir 内 *.bin: {n_bins} 本 / {total_bytes:,} bytes / "
        f"{total_bytes // PSV_RECORD:,} レコード"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
