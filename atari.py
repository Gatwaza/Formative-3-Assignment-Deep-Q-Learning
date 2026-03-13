"""
atari_dqn_gui.py — ALE/DemonAttack-v5
──────────────────────────────────────
All-in-one dashboard with SIMULTANEOUS training + live play.

Left side  : Training stats, reward chart, progress bar
Right side : Live agent gameplay (auto-reloads model as training progresses)
Tab 2      : Hyperparameter experiments table

The play feed automatically reloads the model every episode,
so you watch the agent get better in real time as training runs.

Usage:
    python3 atari_dqn_gui.py
"""

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import csv
import os
import platform
from datetime import datetime

import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv, DummyVecEnv

gym.register_envs(ale_py)

ENV_ID       = "ALE/DemonAttack-v5"
CHECKPOINT   = "dqn_latest.zip"
ACTION_NAMES = ["NOOP","FIRE","RIGHT","LEFT","RIGHTFIRE","LEFTFIRE"]

# ── palette ───────────────────────────────────────────────────────────────────
BG    = "#06060e"
PANEL = "#0d0d22"
CARD  = "#111130"
ACC1  = "#00e5b0"   # teal
ACC2  = "#ff3860"   # red
ACC3  = "#8b5cf6"   # purple
ACC4  = "#fbbf24"   # gold
TEXT  = "#dde4ff"
MUTED = "#44447a"
DARK  = "#03030a"
FT    = ("Courier", 8)
FB    = ("Courier", 9, "bold")
FH    = ("Courier", 7, "bold")


# ──────────────────────────────────────────────────────────────────────────────
# Training callback
# ──────────────────────────────────────────────────────────────────────────────
class GUITrainCallback(BaseCallback):
    def __init__(self, app, total_steps: int, save_freq: int = 10_000):
        super().__init__()
        self.app         = app
        self.total_steps = total_steps
        self.save_freq   = save_freq
        self._last_save  = 0
        self._start      = time.time()

    def _on_step(self) -> bool:
        if not self.app.training:
            return False

        # progress
        pct = min(self.num_timesteps / self.total_steps * 100, 100)
        self.app.t_prog.set(pct)
        self.app.sv["t_steps"].set(f"{self.num_timesteps:,}")
        sps = self.num_timesteps / max(time.time()-self._start, 1)
        self.app.sv["t_sps"].set(f"{sps:.0f}")

        # periodic checkpoint save for live play
        if self.num_timesteps - self._last_save >= self.save_freq:
            self.model.save(CHECKPOINT)
            self._last_save = self.num_timesteps
            self.app.sv["t_ckpt"].set(
                datetime.now().strftime("%H:%M:%S"))

        # episode stats
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = float(info["episode"]["r"])
                l = int(info["episode"]["l"])
                self.app.t_ep += 1
                self.app.t_rewards.append(r)
                self.app.t_lengths.append(l)
                self.app.sv["t_ep"].set(str(self.app.t_ep))
                self.app.sv["t_rew"].set(f"{r:.0f}")
                if self.app.t_rewards:
                    self.app.sv["t_best"].set(
                        f"{max(self.app.t_rewards):.0f}")
                self.app.root.after(0, self.app._redraw_train_chart)
        return True


# ──────────────────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("DemonAttack-v5  |  DQN Dashboard  |  Train + Play Simultaneously")
        root.configure(bg=BG)
        root.geometry("1500x900")

        # frame queues for play canvas
        self._pq: queue.Queue = queue.Queue(maxsize=4)

        # training state
        self.t_env     = None
        self.training  = False
        self.t_ep      = 0
        self.t_rewards = []
        self.t_lengths = []
        self.t_prog    = tk.DoubleVar(value=0)

        # play state
        self.p_running   = False
        self.p_ep        = 0
        self.p_rewards   = []
        self.p_steps     = 0
        self._model_mtime = 0

        # StringVars
        self.sv = {}
        for k, v in [
            ("t_status","Idle"), ("t_ep","0"), ("t_steps","0"),
            ("t_rew","--"), ("t_best","--"), ("t_sps","0"),
            ("t_ckpt","--"),
            ("p_status","Idle"), ("p_ep","0"), ("p_steps","0"),
            ("p_rew","--"), ("p_best","--"), ("p_act","--"),
            ("p_reload","--"),
        ]:
            self.sv[k] = tk.StringVar(value=v)

        self._build()
        root.after(28, self._poll_play_frame)

    # ── root ──────────────────────────────────────────────────────────────────
    def _build(self):
        hdr = tk.Frame(self.root, bg=PANEL, height=50)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  DEMON ATTACK  |  DQN Dashboard",
                 font=("Courier", 15, "bold"), bg=PANEL, fg=ACC2
                 ).pack(side=tk.LEFT, padx=20, pady=8)
        tk.Label(hdr,
                 text="ALE/DemonAttack-v5  |  Discrete(6)  |  "
                      "NOOP · FIRE · RIGHT · LEFT · RIGHTFIRE · LEFTFIRE",
                 font=FT, bg=PANEL, fg=MUTED).pack(side=tk.LEFT, padx=10)
        tk.Label(hdr, text="Train + Play Simultaneously",
                 font=FT, bg=PANEL, fg=ACC1).pack(side=tk.RIGHT, padx=20)

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True)
        sty = ttk.Style()
        sty.theme_use("default")
        sty.configure("TNotebook", background=BG, borderwidth=0)
        sty.configure("TNotebook.Tab", background=PANEL, foreground=MUTED,
                      font=FT, padding=[16,6])
        sty.map("TNotebook.Tab",
                background=[("selected", CARD)],
                foreground=[("selected", ACC1)])

        t1 = tk.Frame(nb, bg=BG)
        t2 = tk.Frame(nb, bg=BG)
        nb.add(t1, text="   TRAIN + PLAY   ")
        nb.add(t2, text="   HYPERPARAMS    ")

        self._tab_main(t1)
        self._tab_hparam(t2)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1: SIMULTANEOUS TRAIN + PLAY
    # ─────────────────────────────────────────────────────────────────────────
    def _tab_main(self, parent):
        # ── top controls row ─────────────────────────────────────────────────
        ctrl = tk.Frame(parent, bg=PANEL, height=44)
        ctrl.pack(fill=tk.X, padx=0, pady=0)
        ctrl.pack_propagate(False)

        # left: training controls
        lc = tk.Frame(ctrl, bg=PANEL)
        lc.pack(side=tk.LEFT, padx=12, pady=6)

        self._sec_inline(lc, "POLICY")
        self.t_pol = tk.StringVar(value="CnnPolicy")
        self._mini_combo(lc, self.t_pol, ["CnnPolicy","MlpPolicy"])

        self._sep_v(lc)
        self._sec_inline(lc, "LR")
        self.t_lr = tk.StringVar(value="0.0001")
        self._mini_combo(lc, self.t_lr,
                         ["0.00001","0.0001","0.0005","0.001","0.01"])

        self._sep_v(lc)
        self._sec_inline(lc, "GAMMA")
        self.t_gm = tk.StringVar(value="0.99")
        self._mini_combo(lc, self.t_gm,
                         ["0.90","0.95","0.99","0.995","0.999"])

        self._sep_v(lc)
        self._sec_inline(lc, "BATCH")
        self.t_bs = tk.StringVar(value="32")
        self._mini_combo(lc, self.t_bs, ["16","32","64","128"])

        self._sep_v(lc)
        self._sec_inline(lc, "N_ENVS")
        self.t_ne = tk.StringVar(value="4")
        self._mini_combo(lc, self.t_ne, ["1","2","4"])

        self._sep_v(lc)
        self._sec_inline(lc, "EPS s/e/d")
        self.t_es = self._tiny(lc, "1.0")
        tk.Label(lc, text="/", bg=PANEL, fg=MUTED, font=FT).pack(side=tk.LEFT)
        self.t_ee = self._tiny(lc, "0.01")
        tk.Label(lc, text="/", bg=PANEL, fg=MUTED, font=FT).pack(side=tk.LEFT)
        self.t_ed = self._tiny(lc, "0.10")

        self._sep_v(lc)
        self._sec_inline(lc, "STEPS")
        self.t_ts_var = tk.StringVar(value="1000000")
        self._mini_combo(lc, self.t_ts_var,
                         ["200000","500000","1000000","2000000"])

        # right: buttons
        rc = tk.Frame(ctrl, bg=PANEL)
        rc.pack(side=tk.RIGHT, padx=12, pady=6)

        self.t_btn = tk.Button(rc, text=" START TRAINING ",
                               bg=ACC1, fg="#000", font=FB, relief=tk.FLAT,
                               padx=10, cursor="hand2",
                               activebackground=ACC1,
                               command=self._toggle_train)
        self.t_btn.pack(side=tk.LEFT, padx=4)

        self.p_btn = tk.Button(rc, text=" START PLAY ",
                               bg=ACC3, fg=TEXT, font=FB, relief=tk.FLAT,
                               padx=10, cursor="hand2",
                               activebackground=ACC3,
                               command=self._toggle_play)
        self.p_btn.pack(side=tk.LEFT, padx=4)

        tk.Button(rc, text=" RESET ",
                  bg=PANEL, fg=TEXT, font=FB, relief=tk.FLAT,
                  padx=8, cursor="hand2",
                  activebackground=PANEL,
                  command=self._reset).pack(side=tk.LEFT, padx=4)

        # ── split body ───────────────────────────────────────────────────────
        body = tk.Frame(parent, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4,6))

        # ── LEFT: training ────────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # training stat bar
        tsb = tk.Frame(left, bg=CARD, height=36)
        tsb.pack(fill=tk.X, pady=(0,4))
        tsb.pack_propagate(False)
        for lbl, key, col in [
            ("STATUS", "t_status", TEXT), ("EP",  "t_ep",   ACC1),
            ("SCORE",  "t_rew",    ACC2),  ("BEST","t_best", ACC4),
            ("STEPS",  "t_steps",  TEXT),  ("SPS", "t_sps",  MUTED),
        ]:
            self._stat_chip(tsb, lbl, key, col)

        # training chart
        chart_f = tk.Frame(left, bg=CARD)
        chart_f.pack(fill=tk.X, pady=(0,4))
        tk.Label(chart_f, text="TRAINING REWARD",
                 font=FH, bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=8, pady=(4,0))
        self.t_fig, self.t_ax = plt.subplots(figsize=(6, 1.6),
                                             facecolor=CARD)
        self.t_ax.set_facecolor(DARK)
        self._sax(self.t_ax)
        self.t_fig.tight_layout(pad=0.4)
        self.t_cv = FigureCanvasTkAgg(self.t_fig, master=chart_f)
        self.t_cv.get_tk_widget().pack(fill=tk.X, padx=6, pady=(0,6))

        # progress
        prog_f = tk.Frame(left, bg=BG)
        prog_f.pack(fill=tk.X, pady=(0,4))
        ttk.Progressbar(prog_f, variable=self.t_prog,
                        maximum=100, mode="determinate"
                        ).pack(fill=tk.X)

        # checkpoint info
        ck_f = tk.Frame(prog_f, bg=BG)
        ck_f.pack(fill=tk.X, pady=(2,0))
        tk.Label(ck_f, text="Last checkpoint:", font=FT, bg=BG, fg=MUTED
                 ).pack(side=tk.LEFT)
        tk.Label(ck_f, textvariable=self.sv["t_ckpt"],
                 font=FT, bg=BG, fg=ACC1).pack(side=tk.LEFT, padx=4)
        tk.Label(ck_f, text=f"  ({CHECKPOINT})",
                 font=FT, bg=BG, fg=MUTED).pack(side=tk.LEFT)

        # training log
        self.t_log = tk.StringVar(value="Configure settings above and press Start Training.")
        tk.Label(left, textvariable=self.t_log, font=FT, bg=BG,
                 fg=MUTED, anchor=tk.W, wraplength=650
                 ).pack(fill=tk.X, pady=(2,0))

        # hyperparameter summary inside left panel
        hp_f = tk.Frame(left, bg=CARD)
        hp_f.pack(fill=tk.X, pady=(6,0))
        tk.Label(hp_f, text="CURRENT HYPERPARAMETERS",
                 font=FH, bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=8, pady=(6,2))
        self.hp_display = tk.Label(hp_f,
                                   text="Set above and start training.",
                                   font=FT, bg=CARD, fg=MUTED,
                                   justify=tk.LEFT, anchor=tk.W)
        self.hp_display.pack(fill=tk.X, padx=8, pady=(0,6))

        # ── DIVIDER ───────────────────────────────────────────────────────────
        div = tk.Frame(body, bg="#1a1a40", width=2)
        div.pack(side=tk.LEFT, fill=tk.Y, padx=6)

        # ── RIGHT: live play ──────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # play stat bar
        psb = tk.Frame(right, bg=CARD, height=36)
        psb.pack(fill=tk.X, pady=(0,4))
        psb.pack_propagate(False)
        for lbl, key, col in [
            ("STATUS","p_status",TEXT), ("EP","p_ep",ACC1),
            ("SCORE", "p_rew",   ACC2), ("BEST","p_best",ACC4),
            ("ACTION","p_act",   ACC3), ("RELOAD","p_reload",MUTED),
        ]:
            self._stat_chip(psb, lbl, key, col)

        # play canvas
        cf = tk.Frame(right, bg=CARD)
        cf.pack(fill=tk.BOTH, expand=True, pady=(0,4))
        tk.Label(cf, text="LIVE AGENT FEED  |  GreedyQPolicy  (argmax Q)",
                 font=FH, bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=8, pady=(4,0))
        self.p_canvas = tk.Canvas(cf, bg=DARK, highlightthickness=0)
        self.p_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))
        self._ph()

        # play chart
        pc_f = tk.Frame(right, bg=CARD)
        pc_f.pack(fill=tk.X, pady=(0,4))
        tk.Label(pc_f, text="EVALUATION SCORES",
                 font=FH, bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=8, pady=(4,0))
        self.p_fig, self.p_ax = plt.subplots(figsize=(6, 1.6), facecolor=CARD)
        self.p_ax.set_facecolor(DARK)
        self._sax(self.p_ax)
        self.p_fig.tight_layout(pad=0.4)
        self.p_cv = FigureCanvasTkAgg(self.p_fig, master=pc_f)
        self.p_cv.get_tk_widget().pack(fill=tk.X, padx=6, pady=(0,6))

        self.p_log = tk.StringVar(value="Press Start Play (auto-reloads model each episode).")
        tk.Label(right, textvariable=self.p_log, font=FT, bg=BG,
                 fg=MUTED, anchor=tk.W, wraplength=650
                 ).pack(fill=tk.X)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2: HYPERPARAMS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    def _tab_hparam(self, parent):
        top = tk.Frame(parent, bg=BG)
        top.pack(fill=tk.X, padx=18, pady=(16,6))
        tk.Label(top, text="HYPERPARAMETER EXPERIMENTS  |  DemonAttack-v5",
                 font=("Courier", 12, "bold"), bg=BG, fg=ACC1
                 ).pack(side=tk.LEFT)
        tk.Button(top, text="  Reload  ", bg=CARD, fg=ACC1,
                  font=FT, relief=tk.FLAT, cursor="hand2",
                  command=self._reload_hp).pack(side=tk.RIGHT, padx=4)
        tk.Button(top, text="  + Add Row  ", bg=ACC3, fg=TEXT,
                  font=FT, relief=tk.FLAT, cursor="hand2",
                  command=self._add_hp_row).pack(side=tk.RIGHT, padx=4)

        tk.Label(parent,
                 text="Each member must run 10 experiments. "
                      "Rows auto-populate from hyperparameter_experiments.csv "
                      "when training completes or via run_experiments.py.",
                 font=FT, bg=BG, fg=MUTED, wraplength=1200, justify=tk.LEFT
                 ).pack(anchor=tk.W, padx=18, pady=(0,8))

        outer = tk.Frame(parent, bg=CARD)
        outer.pack(fill=tk.BOTH, expand=True, padx=18, pady=(0,18))
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb = ttk.Scrollbar(outer, orient=tk.HORIZONTAL)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        cols = ("#","Member","Policy","lr","gamma","batch",
                "eps_s","eps_e","eps_d","steps","n_ep",
                "mean","best","last20","Noted Behavior")

        sty = ttk.Style()
        sty.configure("HP.Treeview", background=CARD,
                      fieldbackground=CARD, foreground=TEXT,
                      rowheight=26, font=("Courier",9))
        sty.configure("HP.Treeview.Heading", background=PANEL,
                      foreground=ACC1, font=("Courier",9,"bold"))
        sty.map("HP.Treeview", background=[("selected", ACC3)])

        self.hp_tree = ttk.Treeview(outer, columns=cols, show="headings",
                                    style="HP.Treeview",
                                    yscrollcommand=vsb.set,
                                    xscrollcommand=hsb.set)
        vsb.config(command=self.hp_tree.yview)
        hsb.config(command=self.hp_tree.xview)

        widths = [26,100,85,65,55,50,55,55,55,80,50,60,60,65,310]
        for col, w in zip(cols, widths):
            self.hp_tree.heading(col, text=col)
            self.hp_tree.column(col, width=w, minwidth=w, anchor=tk.W)
        self.hp_tree.pack(fill=tk.BOTH, expand=True)
        self.hp_tree.tag_configure("odd",  background="#0c0c28")
        self.hp_tree.tag_configure("even", background=CARD)
        self._reload_hp()

    # ── hyperparams helpers ───────────────────────────────────────────────────
    def _reload_hp(self):
        for row in self.hp_tree.get_children():
            self.hp_tree.delete(row)
        fp    = "hyperparameter_experiments.csv"
        rows  = []
        if os.path.exists(fp):
            with open(fp, newline="") as fh:
                for i, r in enumerate(csv.DictReader(fh), 1):
                    rows.append((
                        str(i),
                        r.get("member","--"),
                        r.get("policy","--"),
                        r.get("lr","--"),
                        r.get("gamma","--"),
                        r.get("batch_size","--"),
                        r.get("eps_start","--"),
                        r.get("eps_end","--"),
                        r.get("eps_decay","--"),
                        r.get("timesteps","--"),
                        r.get("n_episodes","--"),
                        r.get("mean_score","--"),
                        r.get("best_score","--"),
                        r.get("mean_last20","--"),
                        r.get("noted_behavior","-- fill in --"),
                    ))
        if not rows:
            for i in range(1, 11):
                rows.append((str(i),"Member","CnnPolicy",
                             "0.0001","0.99","32","1.0","0.01","0.10",
                             "500000","--","--","--","--",
                             "-- run experiments to populate --"))
        for i, row in enumerate(rows):
            self.hp_tree.insert("", tk.END, values=row,
                                tags=("odd" if i%2 else "even",))

    def _add_hp_row(self):
        win = tk.Toplevel(self.root)
        win.title("Add Experiment Row")
        win.configure(bg=PANEL)
        win.geometry("500x440")
        win.grab_set()
        fields = [
            ("Member",""), ("Policy","CnnPolicy"),
            ("lr","0.0001"), ("gamma","0.99"), ("batch","32"),
            ("eps_start","1.0"), ("eps_end","0.01"), ("eps_decay","0.10"),
            ("timesteps","500000"), ("Noted Behavior","-- observations --"),
        ]
        entries = {}
        for label, default in fields:
            r = tk.Frame(win, bg=PANEL)
            r.pack(fill=tk.X, padx=18, pady=3)
            tk.Label(r, text=f"{label}:", width=14, anchor=tk.W,
                     font=FT, bg=PANEL, fg=MUTED).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            tk.Entry(r, textvariable=var, bg=DARK, fg=TEXT,
                     insertbackground=TEXT, font=FT,
                     relief=tk.FLAT, bd=4).pack(side=tk.LEFT, fill=tk.X, expand=True)
            entries[label] = var

        def save():
            fp     = "hyperparameter_experiments.csv"
            is_new = not os.path.exists(fp)
            with open(fp, "a", newline="") as fh:
                w = csv.writer(fh)
                if is_new:
                    w.writerow(["exp_id","label","member","env","policy",
                                "lr","gamma","batch_size","eps_start",
                                "eps_end","eps_decay","n_envs","timesteps",
                                "n_episodes","mean_score","best_score",
                                "mean_last20","elapsed_s","noted_behavior"])
                w.writerow(["manual", "--", entries["Member"].get(), ENV_ID,
                            entries["Policy"].get(), entries["lr"].get(),
                            entries["gamma"].get(), entries["batch"].get(),
                            entries["eps_start"].get(),entries["eps_end"].get(),
                            entries["eps_decay"].get(),"--",
                            entries["timesteps"].get(),"--","--","--","--","--",
                            entries["Noted Behavior"].get()])
            self._reload_hp()
            win.destroy()

        tk.Button(win, text="  Save  ", bg=ACC1, fg="#000",
                  font=FB, relief=tk.FLAT, pady=9, cursor="hand2",
                  command=save).pack(fill=tk.X, padx=18, pady=12)

    # ── shared widget helpers ─────────────────────────────────────────────────
    def _stat_chip(self, parent, label, key, color):
        f = tk.Frame(parent, bg=CARD, padx=8)
        f.pack(side=tk.LEFT, fill=tk.Y, padx=2, pady=4)
        tk.Label(f, text=label, font=("Courier",6,"bold"),
                 bg=CARD, fg=MUTED).pack(anchor=tk.W)
        tk.Label(f, textvariable=self.sv[key],
                 font=("Courier",9,"bold"), bg=CARD, fg=color).pack(anchor=tk.W)

    def _sec_inline(self, parent, text):
        tk.Label(parent, text=text, font=("Courier",6,"bold"),
                 bg=PANEL, fg=MUTED).pack(side=tk.LEFT, padx=(6,2))

    def _sep_v(self, parent):
        tk.Frame(parent, bg="#1a1a40", width=1).pack(side=tk.LEFT, fill=tk.Y,
                                                     padx=4, pady=2)

    def _mini_combo(self, parent, var, vals):
        sty = ttk.Style()
        sty.configure("M.TCombobox", fieldbackground=DARK, background=DARK,
                      foreground=TEXT, arrowcolor=ACC1,
                      selectbackground=DARK, selectforeground=TEXT)
        c = ttk.Combobox(parent, textvariable=var, values=vals,
                         state="readonly", style="M.TCombobox", width=9)
        c.pack(side=tk.LEFT, padx=2)
        return c

    def _tiny(self, parent, default):
        var = tk.StringVar(value=default)
        tk.Entry(parent, textvariable=var, width=5, bg=DARK, fg=TEXT,
                 insertbackground=TEXT, font=FT,
                 relief=tk.FLAT, bd=3).pack(side=tk.LEFT, padx=1)
        return var

    def _ph(self):
        self.p_canvas.delete("all")
        self.p_canvas.create_text(
            400, 200, fill=MUTED, font=("Courier",11), justify=tk.CENTER,
            text="Press Start Play to watch the agent.\n\n"
                 "Auto-reloads model as training progresses.\n"
                 "(Works even before training starts — waits for model.)")

    def _sax(self, ax):
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1a40")

    # ── frame pipeline ────────────────────────────────────────────────────────
    def _push_play(self, frame):
        if self._pq.full():
            try: self._pq.get_nowait()
            except queue.Empty: pass
        self._pq.put(frame)

    def _poll_play_frame(self):
        try:
            frame = self._pq.get_nowait()
            w = max(self.p_canvas.winfo_width(),  1)
            h = max(self.p_canvas.winfo_height(), 1)
            img   = Image.fromarray(frame).resize((w, h), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.p_canvas.delete("all")
            self.p_canvas.create_image(w//2, h//2, image=photo,
                                       anchor=tk.CENTER)
            self.p_canvas._ref = photo
        except queue.Empty:
            pass
        self.root.after(28, self._poll_play_frame)

    # ── charts ────────────────────────────────────────────────────────────────
    def _redraw_train_chart(self):
        self.t_ax.clear(); self.t_ax.set_facecolor(DARK); self._sax(self.t_ax)
        xs = list(range(1, len(self.t_rewards)+1))
        if xs:
            self.t_ax.fill_between(xs, self.t_rewards, alpha=.2, color=ACC2)
            self.t_ax.plot(xs, self.t_rewards, color=ACC2, lw=1.2)
            if len(self.t_rewards) >= 10:
                w  = min(10, len(self.t_rewards))
                ma = np.convolve(self.t_rewards, np.ones(w)/w, mode="valid")
                self.t_ax.plot(xs[w-1:], ma, color=ACC1, lw=2)
        self.t_fig.tight_layout(pad=0.4)
        self.t_cv.draw()

    def _redraw_play_chart(self):
        self.p_ax.clear(); self.p_ax.set_facecolor(DARK); self._sax(self.p_ax)
        xs = list(range(1, len(self.p_rewards)+1))
        if xs:
            self.p_ax.fill_between(xs, self.p_rewards, alpha=.25, color=ACC2)
            self.p_ax.plot(xs, self.p_rewards, color=ACC2, lw=1.3)
            if len(self.p_rewards) >= 5:
                w  = min(5, len(self.p_rewards))
                ma = np.convolve(self.p_rewards, np.ones(w)/w, mode="valid")
                self.p_ax.plot(xs[w-1:], ma, color=ACC1, lw=2)
        self.p_fig.tight_layout(pad=0.4)
        self.p_cv.draw()

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    def _toggle_train(self):
        if self.training:
            self.training = False
            self.t_btn.config(text=" START TRAINING ", bg=ACC1, fg="#000")
            self.sv["t_status"].set("Stopped")
            self.t_log.set("Training stopped.")
        else:
            self.training = True
            self.t_btn.config(text=" STOP TRAINING ", bg=ACC2, fg=TEXT)
            self.sv["t_status"].set("Training...")
            self.t_log.set("Starting...")
            threading.Thread(target=self._train_thread, daemon=True).start()

    def _train_thread(self):
        try:
            policy = self.t_pol.get()
            total  = int(self.t_ts_var.get())
            lr     = float(self.t_lr.get())
            gamma  = float(self.t_gm.get())
            batch  = int(self.t_bs.get())
            n_envs = int(self.t_ne.get())
            es     = float(self.t_es.get())
            ee     = float(self.t_ee.get())
            ed     = float(self.t_ed.get())

            # update hyperparameter display
            hp_str = (f"policy={policy}  lr={lr}  gamma={gamma}  "
                      f"batch={batch}  n_envs={n_envs}\n"
                      f"eps: {es} -> {ee}  decay={ed}  "
                      f"steps={total:,}")
            self.hp_display.config(text=hp_str, fg=TEXT)

            self.t_log.set(f"Building {n_envs}x {ENV_ID}...")
            vec_cls = DummyVecEnv
            if n_envs > 1 and platform.system() != "Darwin":
                vec_cls = SubprocVecEnv
            self.t_env = make_atari_env(ENV_ID, n_envs=n_envs,
                                        vec_env_cls=vec_cls, seed=42)
            self.t_env = VecFrameStack(self.t_env, n_stack=4)

            self.t_log.set(f"Building DQN ({policy})  "
                           f"obs=(84,84,4) n_envs={n_envs} ~{n_envs}x speed...")

            model = DQN(
                policy, self.t_env,
                learning_rate           = lr,
                gamma                   = gamma,
                batch_size              = batch,
                exploration_initial_eps = es,
                exploration_final_eps   = ee,
                exploration_fraction    = ed,
                buffer_size             = 50_000,
                learning_starts         = 5_000,
                target_update_interval  = 1_000,
                train_freq              = 4,
                optimize_memory_usage   = False,
                verbose                 = 0,
            )
            self.t_ep = 0
            self.t_rewards.clear()
            self.t_lengths.clear()

            cb = GUITrainCallback(self, total, save_freq=10_000)
            t0 = time.time()
            model.learn(total_timesteps=total, callback=cb, log_interval=None)
            elapsed = time.time() - t0

            if self.training:
                model.save("dqn_model")
                model.save(CHECKPOINT)
                last20 = float(np.mean(self.t_rewards[-20:])) \
                         if len(self.t_rewards) >= 20 else \
                         float(np.mean(self.t_rewards)) if self.t_rewards else 0
                self.t_log.set(
                    f"Done in {elapsed:.0f}s ({elapsed/60:.1f}m) | "
                    f"Saved dqn_model.zip | "
                    f"Mean (last 20): {last20:.1f}")
                self.sv["t_status"].set("Done")

                fp     = "hyperparameter_experiments.csv"
                is_new = not os.path.exists(fp)
                with open(fp, "a", newline="") as f:
                    w = csv.writer(f)
                    if is_new:
                        w.writerow(["exp_id","label","member","env","policy",
                                    "lr","gamma","batch_size","eps_start",
                                    "eps_end","eps_decay","n_envs","timesteps",
                                    "n_episodes","mean_score","best_score",
                                    "mean_last20","elapsed_s","noted_behavior"])
                    w.writerow(["gui", "--", "--", ENV_ID, policy,
                                lr, gamma, batch, es, ee, ed, n_envs, total,
                                self.t_ep,
                                round(float(np.mean(self.t_rewards)),1) if self.t_rewards else 0,
                                round(float(max(self.t_rewards)),1) if self.t_rewards else 0,
                                round(last20,1), round(elapsed,1),
                                "-- fill in manually --"])

        except Exception as exc:
            self.t_log.set(f"Error: {exc}")
            self.sv["t_status"].set("Error")

        self.training = False
        self.t_btn.config(text=" START TRAINING ", bg=ACC1, fg="#000")

    # ── PLAY ──────────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self.p_running:
            self.p_running = False
            self.p_btn.config(text=" START PLAY ", bg=ACC3, fg=TEXT)
            self.sv["p_status"].set("Stopped")
        else:
            self.p_running = True
            self.p_btn.config(text=" STOP PLAY ", bg=ACC2, fg=TEXT)
            self.sv["p_status"].set("Running...")
            threading.Thread(target=self._play_thread, daemon=True).start()

    def _try_reload(self) -> DQN | None:
        try:
            mtime = os.path.getmtime(CHECKPOINT)
            if mtime > self._model_mtime:
                m = DQN.load(CHECKPOINT)
                self._model_mtime = mtime
                self.sv["p_reload"].set(datetime.now().strftime("%H:%M:%S"))
                return m
        except Exception:
            pass
        return None

    def _play_thread(self):
        try:
            self.p_log.set(
                f"Waiting for {CHECKPOINT}...  "
                "Start training or run train.py in terminal.")

            # wait for checkpoint to exist
            while not os.path.exists(CHECKPOINT) and self.p_running:
                time.sleep(2)

            if not self.p_running:
                return

            model = DQN.load(CHECKPOINT)
            self._model_mtime = os.path.getmtime(CHECKPOINT)

            # model env — same wrappers as training: (84,84,4) grayscale stacked
            model_env = VecFrameStack(
                make_atari_env(ENV_ID, n_envs=1,
                               vec_env_cls=DummyVecEnv, seed=0),
                n_stack=4)

            # display env — raw RGB (210,160,3) for rendering to GUI canvas
            display_env = gym.make(ENV_ID, render_mode="rgb_array")

            self.p_log.set(
                "GreedyQPolicy active — auto-reloading checkpoint each episode")
            self.p_ep = 0
            self.p_rewards.clear()
            self.p_steps = 0

            while self.p_running:
                # reload if model updated
                new_m = self._try_reload()
                if new_m:
                    model = new_m
                    self.p_log.set(
                        f"Model reloaded at {self.sv['p_reload'].get()} "
                        f"— ep {self.p_ep}")

                obs = model_env.reset()
                display_env.reset()
                ep_r, done = 0.0, False

                while not done and self.p_running:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, rew, dones, _ = model_env.step(action)
                    done   = bool(dones[0])
                    ep_r  += float(rew[0])
                    self.p_steps += 1

                    # step display env with same action for rendering
                    _, _, term, trunc, _ = display_env.step(int(action[0]))
                    frame = display_env.render()
                    if frame is not None:
                        self._push_play(frame)
                    if term or trunc:
                        display_env.reset()

                    self.sv["p_act"].set(ACTION_NAMES[int(action[0])])
                    self.sv["p_rew"].set(f"{ep_r:.0f}")
                    self.sv["p_steps"].set(f"{self.p_steps:,}")
                    time.sleep(0.016)

                if self.p_running:
                    self.p_ep += 1
                    self.p_rewards.append(ep_r)
                    best = max(self.p_rewards)
                    self.sv["p_ep"].set(str(self.p_ep))
                    self.sv["p_best"].set(f"{best:.0f}")
                    self.root.after(0, self._redraw_play_chart)
                    self.p_log.set(
                        f"Ep {self.p_ep}  score={ep_r:.0f}  "
                        f"best={best:.0f}  "
                        f"mean={np.mean(self.p_rewards):.1f}")

            model_env.close()
            display_env.close()
            self.p_running = False
            self.p_btn.config(text=" START PLAY ", bg=ACC3, fg=TEXT)
            self.sv["p_status"].set("Stopped")

        except Exception as exc:
            self.p_log.set(f"Error: {exc}")
            self.p_running = False
            self.p_btn.config(text=" START PLAY ", bg=ACC3, fg=TEXT)
            self.sv["p_status"].set("Error")

    # ── RESET ─────────────────────────────────────────────────────────────────
    def _reset(self):
        self.training  = False
        self.p_running = False
        time.sleep(0.12)
        self.t_ep = self.p_ep = self.p_steps = 0
        self.t_rewards.clear(); self.t_lengths.clear(); self.p_rewards.clear()
        self.t_prog.set(0)
        self._model_mtime = 0
        for k, v in [("t_status","Idle"),("t_ep","0"),("t_steps","0"),
                     ("t_rew","--"),("t_best","--"),("t_sps","0"),
                     ("t_ckpt","--"),("p_status","Idle"),("p_ep","0"),
                     ("p_steps","0"),("p_rew","--"),("p_best","--"),
                     ("p_act","--"),("p_reload","--")]:
            self.sv[k].set(v)
        for ax, cv in [(self.t_ax, self.t_cv), (self.p_ax, self.p_cv)]:
            ax.clear(); ax.set_facecolor(DARK); self._sax(ax)
        self.t_fig.tight_layout(pad=0.4); self.t_cv.draw()
        self.p_fig.tight_layout(pad=0.4); self.p_cv.draw()
        self.hp_display.config(text="Set above and start training.", fg=MUTED)
        self.t_log.set("Reset.")
        self.p_log.set("Reset.")
        self._ph()
        self.t_btn.config(text=" START TRAINING ", bg=ACC1, fg="#000")
        self.p_btn.config(text=" START PLAY ",     bg=ACC3, fg=TEXT)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()