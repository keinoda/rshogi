#!/usr/bin/env python3
"""tatara 段階学習ハーネス — 学習・評価・resume・停止判定・レポートの自動化。

800 SB を一度に予約せず、段階的に「学習 → held-out 評価 → 延長判定 → resume」を
回すためのスーパーバイザ。tatara の nnue-train を段階 (stage) ごとに子プロセスで
起動し、experiment.json (毎 superbatch 更新) を監視して以下を行う:

- batches-per-superbatch を round(N_train / (sbs_per_pass * batch_size)) で自動決定
  (既定 sbs_per_pass=20 → 約 20 SB で 1 dataset pass)
- 段階 ladder (既定 120→200→300→400、以後 +extend_step) を継続判定ルールで延長:
    * held-out 5点移動平均の最良が直近 improve_window SB 内 → 次の段階へ延長
    * 横ばい (±plateau_band) → plateau_extend SB だけ 1 回追加、再発なら停止
    * 最良が stop_age SB 以上前かつ stop_band 以上悪化 → 停止 (stage 途中でも abort)
- raw checkpoint の保護: milestone (各 stage 目標 SB) と held-out 最良を
  protected/ へ hardlink し、tatara の --keep-checkpoints rolling 削除から守る
- レポート: 毎 SB ごとに report.png / report.html を再生成
  (x軸 dataset passes、train/held-out loss + 3/5点移動平均、best からの相対悪化、
   LR、throughput、ETA、fp16 clamp ratio、checkpoint/milestone 位置)
- 停止後: held-out 最良付近の .bin を自己対局候補として state に列挙

冪等性: harness_state.json と run ディレクトリの実ファイル (experiments/*.json,
*.ckpt) から状態を再構成するため、ハーネス自体を Ctrl-C / 再起動しても同じ
コマンドの再実行で続きから進む。tatara の異常終了は同一 stage を --resume で
最大 --retries 回リトライする。

依存: python3 + matplotlib (レポート描画。無ければ PNG/HTML を skip して学習は継続)。
    pip3 install --break-system-packages matplotlib

使用例 (蒸留 88 億全量・2048x16x64 baseline):
    python3 staged_train.py \
        --data "$SHOGI_DATA/teachers/distilled_88b_shuffled.bin" \
        --test-data "$SHOGI_DATA/teachers/floodgate.bin" \
        --run-dir "$WORK/runs/base2048" --gpu 0
"""

import argparse
import base64
import html
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PSV_RECORD = 40

# dataviz 準拠の categorical palette (固定順で使用)
C_TRAIN = "#2a78d6"  # blue
C_TEST = "#008300"  # green (held-out 系列は同一 hue で太さ/実線度を変える)
C_LR = "#eb6834"  # orange
C_TPS = "#1baf7a"  # aqua
C_CLAMP = "#4a3aa7"  # violet
C_GRID = "#e5e5e5"
C_MUTED = "#767676"


def records_of(path: Path) -> int:
    size = path.stat().st_size
    if size % PSV_RECORD != 0:
        raise SystemExit(f"ERROR: {path} が {PSV_RECORD}B レコード境界でない ({size} bytes)")
    return size // PSV_RECORD


def moving_average(values, window):
    """trailing 移動平均 (先頭は在る分だけで平均)。"""
    out = []
    acc = 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


class Harness:
    def __init__(self, args):
        self.a = args
        self.run_dir = Path(args.run_dir)
        self.protected = self.run_dir / "protected"
        self.logs = self.run_dir / "logs"
        for d in (self.run_dir, self.protected, self.logs):
            d.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "harness_state.json"
        self.state = self.load_state()

        self.data = Path(args.data)
        self.test_data = Path(args.test_data)
        self.n_train = records_of(self.data)
        self.n_test = records_of(self.test_data)
        self.bps = round(self.n_train / (args.sbs_per_pass * args.batch_size))
        if self.bps < 1:
            raise SystemExit("ERROR: batches-per-superbatch が 0 (データが小さすぎる)")
        self.ladder = [int(x) for x in args.stages.split(",") if x.strip()]
        self.tatara_bin = Path(
            args.tatara_bin or (Path(args.tatara_dir) / "target/release/nnue-train")
        )
        self.stop_reason = None

    # ---------- 状態 ----------
    def load_state(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {
            "targets_done": [],
            "current_target": None,
            "plateau_after_best": None,
            "arrival": {},  # sb(str) -> epoch秒 (throughput/ETA 用)
            "clamp": {},  # sb(str) -> fp16 clamp ratio
            "stopped": False,
            "stop_reason": None,
            "candidates": [],
        }

    def save_state(self):
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1))
        tmp.replace(self.state_path)

    # ---------- experiment.json の統合 ----------
    def merged_history(self):
        """run 内の全 experiment.json を開始時刻順に統合し sb -> entry を返す。"""
        docs = []
        exp_dir = self.run_dir / "experiments"
        if exp_dir.is_dir():
            for p in sorted(exp_dir.glob("*.json")):
                try:
                    doc = json.loads(p.read_text())
                except (json.JSONDecodeError, OSError):
                    continue  # 書き込み途中
                docs.append((doc.get("date", ""), p.name, doc))
        docs.sort()
        merged = {}
        for _, _, doc in docs:
            for e in doc.get("history", []):
                merged[int(e["superbatch"])] = e
        return merged

    def loss_series(self, merged):
        sbs = sorted(merged)
        train = [(sb, merged[sb]["loss"]) for sb in sbs]
        test = [(sb, merged[sb]["test_loss"]) for sb in sbs if merged[sb].get("test_loss") is not None]
        return train, test

    # ---------- 継続判定 ----------
    def judge(self, merged, at_sb):
        """継続判定。('stop'|'extend', next_target|None, reason) を返す。"""
        a = self.a
        _, test = self.loss_series(merged)
        test = [(sb, v) for sb, v in test if sb <= at_sb]
        if len(test) < a.ma_window:
            nxt = self.next_rung(at_sb)
            if nxt is None:
                return ("stop", None, "履歴不足のまま上限に到達")
            return ("extend", nxt, "履歴不足 (判定材料が揃うまで延長)")
        sbs = [sb for sb, _ in test]
        ma = moving_average([v for _, v in test], a.ma_window)
        best_i = min(range(len(ma)), key=lambda i: ma[i])
        best_sb, best_ma = sbs[best_i], ma[best_i]
        last_sb, last_ma = sbs[-1], ma[-1]
        rel = (last_ma - best_ma) / best_ma if best_ma > 0 else 0.0

        if last_sb - best_sb >= a.stop_age and rel >= a.stop_band:
            return (
                "stop",
                None,
                f"5点MA best (sb{best_sb}) から {a.stop_age}SB 以上経過し "
                f"{rel * 100:.2f}% 悪化 (閾値 {a.stop_band * 100:.1f}%)",
            )
        if best_sb > last_sb - a.improve_window:
            nxt = self.next_rung(at_sb)
            if nxt is None:
                return ("stop", None, f"改善中だが上限 {a.max_superbatches}SB に到達")
            return ("extend", nxt, f"直近 {a.improve_window}SB 内で 5点MA best 更新 (sb{best_sb})")
        # 改善なし・停止条件未満 = 横ばい帯
        if abs(rel) <= a.plateau_band:
            if self.state.get("plateau_after_best") == best_sb:
                return ("stop", None, "横ばいが plateau 延長後も継続")
            self.state["plateau_after_best"] = best_sb
            nxt = min(at_sb + a.plateau_extend, a.max_superbatches)
            if nxt <= at_sb:
                return ("stop", None, f"横ばいのまま上限 {a.max_superbatches}SB に到達")
            return ("extend", nxt, f"±{a.plateau_band * 100:.1f}% の横ばい → {a.plateau_extend}SB だけ追加")
        # 悪化しているが stop_age 未達 → plateau 扱いで 1 回だけ様子見
        if self.state.get("plateau_after_best") == best_sb:
            return ("stop", None, f"best (sb{best_sb}) から {rel * 100:.2f}% 悪化が継続")
        self.state["plateau_after_best"] = best_sb
        nxt = min(at_sb + a.plateau_extend, a.max_superbatches)
        if nxt <= at_sb:
            return ("stop", None, f"上限 {a.max_superbatches}SB に到達")
        return ("extend", nxt, f"{rel * 100:.2f}% 悪化 (停止条件未満) → {a.plateau_extend}SB だけ追加")

    def next_rung(self, at_sb):
        for t in self.ladder:
            if t > at_sb:
                return min(t, self.a.max_superbatches)
        nxt = at_sb + self.a.extend_step
        return nxt if nxt <= self.a.max_superbatches else None

    def early_stop_check(self, merged, at_sb):
        """stage 途中の abort 判定 (停止ルールのみ)。"""
        if self.a.no_early_abort:
            return None
        a = self.a
        _, test = self.loss_series(merged)
        if len(test) < a.ma_window:
            return None
        sbs = [sb for sb, _ in test]
        ma = moving_average([v for _, v in test], a.ma_window)
        best_i = min(range(len(ma)), key=lambda i: ma[i])
        rel = (ma[-1] - ma[best_i]) / ma[best_i] if ma[best_i] > 0 else 0.0
        if sbs[-1] - sbs[best_i] >= a.stop_age and rel >= a.stop_band:
            return (
                f"early-abort: 5点MA best (sb{sbs[best_i]}) から {sbs[-1] - sbs[best_i]}SB 経過し "
                f"{rel * 100:.2f}% 悪化"
            )
        return None

    # ---------- checkpoint 保護 ----------
    def ckpt_path(self, sb):
        return self.run_dir / f"{self.a.net_id}-{sb}.ckpt"

    def saved_ckpt_sbs(self):
        pat = re.compile(re.escape(self.a.net_id) + r"-(\d+)\.ckpt$")
        out = []
        for p in self.run_dir.glob(f"{self.a.net_id}-*.ckpt"):
            m = pat.search(p.name)
            if m:
                out.append(int(m.group(1)))
        return sorted(out)

    def protect_checkpoints(self, merged):
        milestones = set(self.ladder) | set(self.state["targets_done"])
        for sb in self.saved_ckpt_sbs():
            if sb in milestones:
                dst = self.protected / f"milestone-{sb}.ckpt"
                if not dst.exists():
                    os.link(self.ckpt_path(sb), dst)
                    print(f"[harness] milestone 保護: sb{sb}")
        # held-out 最良 (raw test_loss) の raw ckpt を保護
        cand = [
            (merged[sb]["test_loss"], sb)
            for sb in self.saved_ckpt_sbs()
            if sb in merged and merged[sb].get("test_loss") is not None
        ]
        # protected 内の既存 best も比較対象に含める (rolling 削除後も維持)
        for p in self.protected.glob("best-*.ckpt"):
            sb = int(re.search(r"best-(\d+)\.ckpt$", p.name).group(1))
            if sb in merged and merged[sb].get("test_loss") is not None:
                cand.append((merged[sb]["test_loss"], sb))
        if not cand:
            return
        _, best_sb = min(cand)
        dst = self.protected / f"best-{best_sb}.ckpt"
        if dst.exists():
            return
        src = self.ckpt_path(best_sb)
        if not src.exists():
            return  # 既に rolling 削除済み (通常は起きない)
        for p in self.protected.glob("best-*.ckpt"):
            p.unlink()
        os.link(src, dst)
        print(f"[harness] held-out best 保護: sb{best_sb}")

    def latest_resume_ckpt(self):
        sbs = self.saved_ckpt_sbs()
        if sbs:
            return self.ckpt_path(sbs[-1])
        prot = sorted(
            self.protected.glob("*.ckpt"),
            key=lambda p: int(re.search(r"-(\d+)\.ckpt$", p.name).group(1)),
        )
        return prot[-1] if prot else None

    # ---------- tatara 起動 ----------
    def tatara_cmd(self, target, resume_ckpt):
        a = self.a
        cmd = [
            str(self.tatara_bin),
            "--data", str(self.data),
            "--test-data", str(self.test_data),
            "--test-positions", str(self.n_test),
            "--output", str(self.run_dir),
            "--net-id", a.net_id,
            "--superbatches", str(target),
            "--batches-per-superbatch", str(self.bps),
            "--batch-size", str(a.batch_size),
            "--lr", repr(a.lr),
            "--lr-gamma", repr(a.lr_gamma),
            "--lr-step", str(a.lr_step),
            "--weight-decay", "0.0",
            "--wdl", "0.0",
            "--scale", repr(a.scale),
            "--save-rate", str(a.save_rate),
            "--keep-checkpoints", str(a.keep_checkpoints),
            "--threads", str(a.threads),
            "--all-optim",
            "--win-rate-model",
        ]
        if resume_ckpt is not None:
            cmd += ["--resume", str(resume_ckpt)]
        cmd += shlex.split(a.tatara_extra)
        cmd += [
            "layerstack",
            "--ft-out", str(a.ft_out),
            "--l1", str(a.l1),
            "--l2", str(a.l2),
            "--bucket-mode", "progress8kpabs",
            "--progress-coeff", str(a.progress_coeff),
        ]
        if a.num_buckets is not None:
            cmd += ["--num-buckets", str(a.num_buckets)]
        return cmd

    def run_stage(self, target):
        """target SB まで学習する。'done' | 'aborted' | 'failed' を返す。"""
        a = self.a
        clamp_re = re.compile(r"\[fp16-clamp\] sb=(\d+) .*ratio=([0-9.eE+-]+)")
        attempt = 0
        while True:
            resume = self.latest_resume_ckpt()
            cmd = self.tatara_cmd(target, resume)
            log_path = self.logs / f"stage-{target}.log"
            print(f"[harness] stage → sb{target} (resume={resume.name if resume else 'なし'})")
            print(f"[harness]   {' '.join(cmd)}")
            if a.dry_run:
                return "done"
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
            log_fh = open(log_path, "ab")
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
            aborted = None
            log_off = 0
            # 再起動時に過去 SB へ現在時刻を刻むと throughput が壊れるため、
            # 起動前snapshot 以降に「新しく現れた」SB だけへ到着時刻を記録する
            pre = self.merged_history()
            last_reported = max(pre) if pre else 0
            try:
                while True:
                    rc = proc.poll()
                    # ログから fp16 clamp ratio を吸い上げる
                    try:
                        with open(log_path, "rb") as f:
                            f.seek(log_off)
                            chunk = f.read()
                            log_off += len(chunk)
                        for m in clamp_re.finditer(chunk.decode("utf-8", "replace")):
                            self.state["clamp"][m.group(1)] = float(m.group(2))
                    except OSError:
                        pass
                    merged = self.merged_history()
                    new_last = max(merged) if merged else 0
                    if new_last > last_reported:
                        for sb in range(last_reported + 1, new_last + 1):
                            if sb in merged:
                                self.state["arrival"].setdefault(str(sb), time.time())
                        last_reported = new_last
                        self.protect_checkpoints(merged)
                        self.save_state()
                        self.render_report(merged, target)
                        reason = self.early_stop_check(merged, new_last)
                        if reason is not None:
                            print(f"[harness] {reason} → tatara を停止")
                            aborted = reason
                            proc.terminate()
                            try:
                                proc.wait(timeout=60)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                            break
                    if rc is not None:
                        break
                    time.sleep(a.poll_secs)
            finally:
                log_fh.close()
            if aborted:
                self.stop_reason = aborted
                return "aborted"
            merged = self.merged_history()
            reached = max(merged) if merged else 0
            if proc.returncode == 0 and reached >= target:
                return "done"
            attempt += 1
            if attempt > a.retries:
                print(
                    f"[harness] ERROR: stage sb{target} が {a.retries} 回のリトライ後も"
                    f" 未達 (rc={proc.returncode}, 到達 sb{reached})"
                )
                return "failed"
            print(
                f"[harness] WARNING: tatara 異常終了 (rc={proc.returncode}, 到達 sb{reached})"
                f" — resume でリトライ {attempt}/{a.retries}"
            )
            time.sleep(10)

    # ---------- レポート ----------
    def lr_at(self, sb):
        return self.a.lr * (self.a.lr_gamma ** ((sb - 1) // self.a.lr_step))

    def passes_at(self, sb):
        return sb * self.bps * self.a.batch_size / self.n_train

    def throughput_series(self, sbs):
        raw = []
        for prev, cur in zip(sbs, sbs[1:]):
            t0 = self.state["arrival"].get(str(prev))
            t1 = self.state["arrival"].get(str(cur))
            if t0 and t1 and t1 > t0:
                raw.append((cur, t1 - t0, cur - prev))
        if not raw:
            return []
        # stage 境界 (プロセス再起動の待ち時間) をまたぐ外れ値 dt を除外する
        med = sorted(dt / n for _, dt, n in raw)[len(raw) // 2]
        return [
            (cur, n * self.bps * self.a.batch_size / dt)
            for cur, dt, n in raw
            if dt / n <= 5 * med
        ]

    def render_report(self, merged, target):
        try:
            self._render_report(merged, target)
        except Exception as e:  # レポート失敗で学習は止めない
            print(f"[harness] WARNING: レポート生成に失敗: {e}")

    def _render_report(self, merged, target):
        a = self.a
        train, test = self.loss_series(merged)
        if not train:
            return
        png_b64 = None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sbs_tr = [sb for sb, _ in train]
            sbs_te = [sb for sb, _ in test]
            ma3 = moving_average([v for _, v in test], 3) if test else []
            ma5 = moving_average([v for _, v in test], a.ma_window) if test else []
            tps = self.throughput_series(sbs_te or sbs_tr)
            clamp = sorted((int(k), v) for k, v in self.state["clamp"].items())

            n_rows = 3 + (1 if clamp else 0)
            fig, axes = plt.subplots(
                n_rows, 1, figsize=(12.5, 2.6 * n_rows + 1.2), sharex=True,
                gridspec_kw={"height_ratios": [2.2] + [1] * (n_rows - 1)},
            )
            fig.subplots_adjust(hspace=0.28, left=0.075, right=0.97, top=0.93, bottom=0.1)
            for ax in axes:
                ax.grid(True, color=C_GRID, linewidth=0.7)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)

            # (1) loss
            ax = axes[0]
            ax.plot([self.passes_at(s) for s in sbs_tr], [v for _, v in train],
                    color=C_TRAIN, linewidth=1.2, label="train loss")
            if test:
                xs = [self.passes_at(s) for s in sbs_te]
                ax.plot(xs, [v for _, v in test], color=C_TEST, linewidth=0.9, alpha=0.35,
                        label="held-out loss")
                ax.plot(xs, ma3, color=C_TEST, linewidth=1.3, linestyle="--", label="held-out MA3")
                ax.plot(xs, ma5, color=C_TEST, linewidth=2.2, label=f"held-out MA{a.ma_window}")
                bi = min(range(len(ma5)), key=lambda i: ma5[i])
                ax.plot(xs[bi], ma5[bi], "o", color=C_TEST, markersize=9,
                        markeredgecolor="white", markeredgewidth=1.5, zorder=5)
                ax.annotate(f"best MA{a.ma_window} @sb{sbs_te[bi]}",
                            (xs[bi], ma5[bi]), textcoords="offset points", xytext=(8, -12),
                            fontsize=9, color=C_MUTED)
            for t in sorted(set(self.ladder) | set(self.state["targets_done"])):
                if t <= max(sbs_tr):
                    ax.axvline(self.passes_at(t), color=C_MUTED, linewidth=0.8,
                               linestyle=":", alpha=0.6)
            for sb in self.saved_ckpt_sbs():
                ax.plot(self.passes_at(sb), ax.get_ylim()[0], marker="|", color=C_MUTED,
                        markersize=7, clip_on=False)
            ax.set_ylabel("loss")
            ax.legend(loc="upper right", frameon=False, fontsize=9)
            sec = ax.secondary_xaxis(
                "top",
                functions=(
                    lambda p: p * self.n_train / (self.bps * a.batch_size),
                    lambda s: s * self.bps * a.batch_size / self.n_train,
                ),
            )
            sec.set_xlabel("superbatch", fontsize=9, color=C_MUTED)
            sec.tick_params(labelsize=8, colors=C_MUTED)

            # (2) LR
            ax = axes[1]
            ax.plot([self.passes_at(s) for s in sbs_tr], [self.lr_at(s) for s in sbs_tr],
                    color=C_LR, linewidth=2.0)
            ax.set_ylabel("LR")
            ax.set_yscale("log")

            # (3) throughput
            ax = axes[2]
            if tps:
                ax.plot([self.passes_at(s) for s, _ in tps], [v / 1e6 for _, v in tps],
                        color=C_TPS, linewidth=2.0)
                mean_tps = sum(v for _, v in tps) / len(tps)
                remain = max(0, target - max(sbs_tr))
                eta_h = remain * self.bps * a.batch_size / mean_tps / 3600 if mean_tps else 0
                # 図中は CJK フォント非依存の ASCII 表記に統一 (日本語は HTML 側で出す)
                ax.set_title(
                    f"avg {mean_tps / 1e6:.2f}M pos/s | ETA to sb{target}: {eta_h:.1f}h",
                    fontsize=9, color=C_MUTED, loc="right", pad=2,
                )
            ax.set_ylabel("Mpos/s")

            # (4) clamp
            if clamp:
                ax = axes[3]
                ax.plot([self.passes_at(s) for s, _ in clamp], [v for _, v in clamp],
                        color=C_CLAMP, linewidth=2.0)
                ax.set_ylabel("fp16 clamp")
                ax.set_yscale("log")

            axes[-1].set_xlabel("dataset passes")
            fig.suptitle(
                f"{a.net_id}  ({a.ft_out}x{a.l1}x{a.l2}, {self.n_train:,} pos, "
                f"bps={self.bps}, 1 pass≈{self.n_train / (self.bps * a.batch_size):.1f} SB)",
                fontsize=11,
            )
            fig.savefig(self.run_dir / "report.png", dpi=110)
            plt.close(fig)
            png_b64 = base64.b64encode((self.run_dir / "report.png").read_bytes()).decode()
        except ImportError:
            pass  # matplotlib 無し → HTML のみ (表)

        self._render_html(merged, target, png_b64)

    def _render_html(self, merged, target, png_b64):
        a = self.a
        train, test = self.loss_series(merged)
        last_sb = max(merged)
        rows = []

        def row(k, v):
            rows.append(
                f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
            )

        row("superbatch", f"{last_sb} / 目標 {target}")
        row("dataset passes", f"{self.passes_at(last_sb):.2f}")
        row("train loss", f"{train[-1][1]:.8f}")
        if test:
            ma5 = moving_average([v for _, v in test], a.ma_window)
            bi = min(range(len(ma5)), key=lambda i: ma5[i])
            rel = (ma5[-1] - ma5[bi]) / ma5[bi] * 100 if ma5[bi] > 0 else 0.0
            row("held-out loss", f"{test[-1][1]:.8f}")
            row(f"held-out MA{a.ma_window}", f"{ma5[-1]:.8f}")
            row(f"best MA{a.ma_window}", f"{ma5[bi]:.8f} @sb{test[bi][0]}")
            row("best からの相対悪化", f"{rel:+.3f}%")
            if merged[last_sb].get("test_accuracy") is not None:
                row("test_accuracy", f"{merged[last_sb]['test_accuracy']:.6f}")
        row("LR", f"{self.lr_at(last_sb):.3e}")
        tps = self.throughput_series([sb for sb, _ in (test or train)])
        if tps:
            row("throughput (直近)", f"{tps[-1][1] / 1e6:.2f}M pos/s")
        clamp = self.state["clamp"].get(str(last_sb))
        if clamp is not None:
            row("fp16 clamp ratio", f"{clamp:.3e}")
        row("保存済み raw ckpt", ", ".join(f"sb{s}" for s in self.saved_ckpt_sbs()) or "なし")
        prot = sorted(p.name for p in self.protected.glob("*.ckpt"))
        row("保護済み (protected/)", ", ".join(prot) or "なし")
        if self.state.get("stop_reason"):
            row("停止理由", self.state["stop_reason"])
        if self.state.get("candidates"):
            row("自己対局候補", ", ".join(self.state["candidates"]))

        doc = f"""<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="30">
<title>{html.escape(a.net_id)} staged training</title>
<style>
body{{font-family:sans-serif;margin:1.2rem;color:#1a1a1a}}
table{{border-collapse:collapse;margin-top:1rem}}
th,td{{border:1px solid #ddd;padding:.35rem .6rem;text-align:left;font-size:.9rem}}
th{{background:#f7f7f7;font-weight:600}}
img{{max-width:100%;border:1px solid #eee}}
</style></head><body>
<h1 style="font-size:1.15rem">{html.escape(a.net_id)} — 段階学習レポート</h1>
<p style="color:#767676;font-size:.85rem">更新 {time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
 (30 秒ごとに自動リロード)</p>
{f'<img src="data:image/png;base64,{png_b64}" alt="loss / LR / throughput chart">' if png_b64 else "<p>matplotlib 未導入のため図は省略 (pip3 install --break-system-packages matplotlib)</p>"}
<table>{"".join(rows)}</table>
</body></html>"""
        (self.run_dir / "report.html").write_text(doc, encoding="utf-8")

    # ---------- 停止処理 ----------
    def finalize(self, merged):
        a = self.a
        _, test = self.loss_series(merged)
        by_loss = sorted((v, sb) for sb, v in test if (self.run_dir / f"{a.net_id}-{sb}.bin").exists())
        cand = []
        if by_loss:
            best_sb = by_loss[0][1]
            near = [sb for _, sb in by_loss if abs(sb - best_sb) <= 2 * a.save_rate]
            cand = [f"{a.net_id}-{sb}.bin" for sb in sorted(set(near + [sb for _, sb in by_loss[:3]]))]
        self.state["candidates"] = cand
        self.state["stopped"] = True
        self.state["stop_reason"] = self.stop_reason
        self.save_state()
        if merged:
            self.render_report(merged, max(merged))
        print("[harness] ===== 停止 =====")
        print(f"[harness] 理由: {self.stop_reason}")
        print(f"[harness] 自己対局候補 (held-out 最良近傍): {', '.join(cand) or 'なし'}")

    # ---------- メイン ----------
    def main(self):
        a = self.a
        print(
            f"[harness] N_train={self.n_train:,} N_test={self.n_test:,} "
            f"bps={self.bps} (1 pass ≈ {self.n_train / (self.bps * a.batch_size):.2f} SB) "
            f"ladder={self.ladder}"
        )
        while True:
            merged = self.merged_history()
            last_sb = max(merged) if merged else 0
            target = self.state.get("current_target")
            if target is not None and last_sb >= target:
                if target not in self.state["targets_done"]:
                    self.state["targets_done"].append(target)
                self.protect_checkpoints(merged)
                self.state["current_target"] = None
                self.save_state()
                target = None
            if target is None:
                if last_sb == 0:
                    target = self.ladder[0]
                    reason = "初回 stage"
                else:
                    verdict, target, reason = self.judge(merged, last_sb)
                    if verdict == "stop":
                        self.stop_reason = reason
                        self.finalize(merged)
                        return 0
                print(f"[harness] 判定: sb{last_sb} → 目標 sb{target} ({reason})")
                self.state["current_target"] = target
                self.save_state()
            result = self.run_stage(target)
            merged = self.merged_history()
            if result == "aborted":
                self.finalize(merged)
                return 0
            if result == "failed":
                self.state["stop_reason"] = "tatara の異常終了が継続 (要手動確認)"
                self.save_state()
                if merged:
                    self.render_report(merged, target)
                return 1
            # done → ループ先頭で targets_done 記録と次判定


def parse_args(argv=None):
    work = os.environ.get("WORK", "/workspace")
    shogi_data = os.environ.get("SHOGI_DATA", f"{work}/shogi-data")
    ap = argparse.ArgumentParser(
        description="tatara 段階学習ハーネス (学習→評価→resume→停止判定→レポート)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="学習 PSV (シャッフル済み単一ファイル)")
    ap.add_argument("--test-data", required=True, help="held-out PSV (毎 SB 全件評価)")
    ap.add_argument("--run-dir", required=True, help="checkpoint/レポートの出力 root")
    ap.add_argument("--tatara-dir", default=f"{work}/tatara")
    ap.add_argument("--tatara-bin", default=None, help="nnue-train のパス (既定: tatara-dir から)")
    ap.add_argument("--net-id", default="base2048x16x64")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES 値")
    ap.add_argument("--ft-out", type=int, default=2048)
    ap.add_argument("--l1", type=int, default=16)
    ap.add_argument("--l2", type=int, default=64)
    ap.add_argument("--num-buckets", type=int, default=None, help="未指定 = tatara 既定 (9)")
    ap.add_argument("--progress-coeff", default=f"{shogi_data}/progress/progress.bin")
    ap.add_argument("--batch-size", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=8.75e-4)
    ap.add_argument("--lr-gamma", type=float, default=0.995)
    ap.add_argument("--lr-step", type=int, default=1)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--scale", type=float, default=290.0)
    ap.add_argument("--sbs-per-pass", type=int, default=20, help="1 dataset pass あたりの SB 数")
    ap.add_argument("--save-rate", type=int, default=10)
    ap.add_argument("--keep-checkpoints", type=int, default=8)
    ap.add_argument("--stages", default="120,200,300,400")
    ap.add_argument("--extend-step", type=int, default=100, help="ladder 消化後の延長幅")
    ap.add_argument("--plateau-extend", type=int, default=40)
    ap.add_argument("--ma-window", type=int, default=5)
    ap.add_argument("--improve-window", type=int, default=10)
    ap.add_argument("--plateau-band", type=float, default=0.001)
    ap.add_argument("--stop-age", type=int, default=20)
    ap.add_argument("--stop-band", type=float, default=0.002)
    ap.add_argument("--max-superbatches", type=int, default=800)
    ap.add_argument("--no-early-abort", action="store_true",
                    help="stage 途中の停止判定 (abort) を無効化")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--poll-secs", type=float, default=10.0)
    ap.add_argument("--tatara-extra", default="", help="nnue-train へ渡す追加 global フラグ")
    ap.add_argument("--dry-run", action="store_true", help="コマンドを表示して実行しない")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(Harness(parse_args()).main())
