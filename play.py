"""
play.py — ALE/DemonAttack-v5
─────────────────────────────
Loads a trained model and plays with GreedyQPolicy (deterministic=True).
Auto-reloads the model every episode when using dqn_latest.zip —
so you can watch the agent improve in real time while train.py is running.

Usage:
    python3 play.py                                # watch live training
    python3 play.py --model dqn_model.zip          # play final model
    python3 play.py --model dqn_latest.zip --episodes 0   # infinite live
"""

import argparse
import os
import threading
import time
import queue

import numpy as np
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

gym.register_envs(ale_py)

ENV_ID       = "ALE/DemonAttack-v5"
ACTION_NAMES = ["NOOP","FIRE","RIGHT","LEFT","RIGHTFIRE","LEFTFIRE"]

BG    = "#06060e"
PANEL = "#0d0d22"
CARD  = "#111130"
ACC1  = "#00e5b0"
ACC2  = "#ff3860"
ACC3  = "#8b5cf6"
ACC4  = "#fbbf24"
TEXT  = "#dde4ff"
MUTED = "#44447a"
DARK  = "#03030a"
FT    = ("Courier", 8)
FB    = ("Courier", 9, "bold")


class GreedyQPolicy:
    def __init__(self, model):
        self._model = model

    def predict(self, obs):
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)


class PlayGUI:
    def __init__(self, root, args):
        self.root    = root
        self.args    = args
        self.running = False
        self._fq: queue.Queue = queue.Queue(maxsize=4)

        self.ep_rewards  = []
        self.play_ep     = 0
        self.total_steps = 0
        self._model_mtime = 0   # track file modification time for auto-reload

        root.title("DemonAttack-v5  |  GreedyQPolicy  |  Live Play")
        root.configure(bg=BG)
        root.geometry("1260x820")
        self._build()
        root.after(28, self._poll_frame)
        root.after(600, self._auto_start)

    def _auto_start(self):
        self._start()

    def _build(self):
        # header
        hdr = tk.Frame(self.root, bg=PANEL, height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="  DEMON ATTACK  |  DQN Live Play",
                 font=("Courier", 15, "bold"), bg=PANEL, fg=ACC2
                 ).pack(side=tk.LEFT, padx=20, pady=10)
        self.model_lbl = tk.Label(hdr, text=f"model: {self.args.model}",
                                  font=FT, bg=PANEL, fg=MUTED)
        self.model_lbl.pack(side=tk.LEFT, padx=10)
        tk.Label(hdr, text="GreedyQPolicy  (argmax Q)",
                 font=FT, bg=PANEL, fg=ACC1).pack(side=tk.RIGHT, padx=20)

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # canvas
        cf = tk.Frame(body, bg=CARD)
        cf.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(cf, text="LIVE FEED", font=FT, bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=10, pady=(6,0))
        self.canvas = tk.Canvas(cf, bg=DARK, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2,4))

        # action bar
        ab = tk.Frame(cf, bg=CARD)
        ab.pack(fill=tk.X, padx=8, pady=(0,6))
        tk.Label(ab, text="Action:", font=FT, bg=CARD, fg=MUTED).pack(side=tk.LEFT)
        self.act_var = tk.StringVar(value="--")
        tk.Label(ab, textvariable=self.act_var,
                 font=("Courier", 11, "bold"), bg=CARD, fg=ACC4
                 ).pack(side=tk.LEFT, padx=6)
        self.reload_lbl = tk.Label(ab, text="", font=FT, bg=CARD, fg=ACC1)
        self.reload_lbl.pack(side=tk.RIGHT, padx=8)

        # right panel
        right = tk.Frame(body, bg=BG, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(10,0))
        right.pack_propagate(False)

        # stat cards
        g = tk.Frame(right, bg=BG)
        g.pack(fill=tk.X)
        self.sv = {}
        for (lbl, init, col, r, c) in [
            ("EPISODE","0",ACC1,0,0), ("SCORE","0",ACC2,0,1),
            ("BEST","--",ACC4,1,0),   ("STEPS","0",TEXT,1,1),
        ]:
            card = tk.Frame(g, bg=CARD, padx=10, pady=8)
            card.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
            g.columnconfigure(c, weight=1)
            tk.Label(card, text=lbl, font=("Courier",7,"bold"),
                     bg=CARD, fg=MUTED).pack(anchor=tk.W)
            var = tk.StringVar(value=init)
            tk.Label(card, textvariable=var,
                     font=("Courier",16,"bold"), bg=CARD, fg=col).pack(anchor=tk.W)
            self.sv[lbl] = var

        # chart
        cc = tk.Frame(right, bg=CARD)
        cc.pack(fill=tk.BOTH, expand=True, pady=(8,0))
        tk.Label(cc, text="EPISODE SCORES", font=FT, bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=10, pady=(8,2))
        self.fig, self.ax = plt.subplots(figsize=(2.8,2.0), facecolor=CARD)
        self.ax.set_facecolor(DARK)
        self._sax()
        self.fig.tight_layout(pad=0.5)
        self.cv = FigureCanvasTkAgg(self.fig, master=cc)
        self.cv.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,8))

        # policy card
        pc = tk.Frame(right, bg=CARD, pady=8)
        pc.pack(fill=tk.X, pady=(8,0))
        tk.Label(pc, text="GREEDYQPOLICY", font=FT, bg=CARD, fg=ACC1
                 ).pack(anchor=tk.W, padx=10)
        tk.Label(pc,
                 text="action = argmax_a Q(s,a)\ndeterministic=True\neps=0 — pure exploitation\n\nAuto-reloads model\nafter each episode.",
                 font=FT, bg=CARD, fg=ACC3, justify=tk.LEFT
                 ).pack(anchor=tk.W, padx=10, pady=4)

        # buttons
        bf = tk.Frame(right, bg=BG)
        bf.pack(fill=tk.X, pady=(10,0))
        self.run_btn = tk.Button(bf, text="  STOP  ", bg=ACC2, fg=TEXT,
                                 font=FB, relief=tk.FLAT, pady=8,
                                 cursor="hand2", activebackground=ACC2,
                                 command=self._toggle)
        self.run_btn.pack(fill=tk.X, pady=2)
        tk.Button(bf, text="  RESET  ", bg=PANEL, fg=TEXT,
                  font=FB, relief=tk.FLAT, pady=6, cursor="hand2",
                  activebackground=PANEL, command=self._reset
                  ).pack(fill=tk.X, pady=2)

        self.status_var = tk.StringVar(value="Loading model...")
        tk.Label(self.root, textvariable=self.status_var,
                 font=FT, bg=PANEL, fg=MUTED, anchor=tk.W
                 ).pack(fill=tk.X, side=tk.BOTTOM, padx=12, pady=4)

        self._ph()

    def _ph(self):
        self.canvas.delete("all")
        self.canvas.create_text(400, 260, fill=MUTED,
                                font=("Courier",12), justify=tk.CENTER,
                                text="Waiting for model...\nWill start automatically.")

    def _sax(self):
        self.ax.tick_params(colors=MUTED, labelsize=7)
        for sp in self.ax.spines.values():
            sp.set_edgecolor("#1a1a40")
        self.ax.set_xlabel("Episode", color=MUTED, fontsize=7)
        self.ax.set_ylabel("Score",   color=MUTED, fontsize=7)

    def _redraw(self):
        if not self.ep_rewards: return
        self.ax.clear(); self.ax.set_facecolor(DARK); self._sax()
        xs = list(range(1, len(self.ep_rewards)+1))
        self.ax.fill_between(xs, self.ep_rewards, alpha=0.3, color=ACC2)
        self.ax.plot(xs, self.ep_rewards, color=ACC2, lw=1.3)
        if len(self.ep_rewards) >= 5:
            w  = min(5, len(self.ep_rewards))
            ma = np.convolve(self.ep_rewards, np.ones(w)/w, mode="valid")
            self.ax.plot(xs[w-1:], ma, color=ACC1, lw=2)
        self._sax()
        self.fig.tight_layout(pad=0.5)
        self.cv.draw()

    def _push(self, frame):
        if self._fq.full():
            try: self._fq.get_nowait()
            except queue.Empty: pass
        self._fq.put(frame)

    def _poll_frame(self):
        try:
            frame = self._fq.get_nowait()
            w = max(self.canvas.winfo_width(),  1)
            h = max(self.canvas.winfo_height(), 1)
            img   = Image.fromarray(frame).resize((w, h), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(w//2, h//2, image=photo, anchor=tk.CENTER)
            self.canvas._ref = photo
        except queue.Empty:
            pass
        self.root.after(28, self._poll_frame)

    def _toggle(self):
        if self.running:
            self.running = False
            self.run_btn.config(text="  START  ", bg=ACC1, fg="#000")
            self.status_var.set("Stopped.")
        else:
            self._start()

    def _start(self):
        self.running = True
        self.run_btn.config(text="  STOP  ", bg=ACC2, fg=TEXT)
        threading.Thread(target=self._play_thread, daemon=True).start()

    def _reset(self):
        self.running = False
        time.sleep(0.1)
        self.ep_rewards.clear()
        self.play_ep = self.total_steps = 0
        for k, v in [("EPISODE","0"),("SCORE","0"),("BEST","--"),("STEPS","0")]:
            self.sv[k].set(v)
        self.act_var.set("--")
        self.ax.clear(); self.ax.set_facecolor(DARK); self._sax()
        self.fig.tight_layout(pad=0.5); self.cv.draw()
        self._ph()
        self.run_btn.config(text="  START  ", bg=ACC1, fg="#000")
        self.status_var.set("Reset.")

    def _try_reload_model(self, path: str):
        """Reload model only if the file has been updated since last load."""
        try:
            mtime = os.path.getmtime(path)
            if mtime > self._model_mtime:
                model = DQN.load(path)
                self._model_mtime = mtime
                self.reload_lbl.config(
                    text=f"reloaded {datetime.now().strftime('%H:%M:%S')}")
                return model
        except Exception:
            pass
        return None

    def _play_thread(self):
        from datetime import datetime
        try:
            path = self.args.model
            self.status_var.set(f"Loading {path}...")

            # wait for model file to exist
            waited = 0
            while not os.path.exists(path) and self.running:
                self.status_var.set(
                    f"Waiting for {path}... ({waited}s)  "
                    "Start train.py in another terminal.")
                time.sleep(2)
                waited += 2

            if not self.running:
                return

            model = DQN.load(path)
            self._model_mtime = os.path.getmtime(path)
            policy = GreedyQPolicy(model)

            # model env — same wrappers as training: (84,84,4) grayscale stacked
            model_env = VecFrameStack(
                make_atari_env(ENV_ID, n_envs=1,
                               vec_env_cls=DummyVecEnv, seed=0),
                n_stack=4)

            # display env — raw RGB (210,160,3) just for rendering to GUI
            display_env = gym.make(ENV_ID, render_mode="rgb_array")

            max_ep = self.args.episodes
            self.status_var.set("GreedyQPolicy running — auto-reloading model each episode")
            ep = 0

            while self.running and (max_ep == 0 or ep < max_ep):
                # reload model if checkpoint updated
                new_model = self._try_reload_model(path)
                if new_model:
                    policy = GreedyQPolicy(new_model)

                # reset both envs
                obs = model_env.reset()
                display_env.reset()
                ep_r, done = 0.0, False

                while not done and self.running:
                    action = policy.predict(obs)       # obs shape (1,4,84,84)
                    obs, rew, dones, _ = model_env.step([action])
                    done   = bool(dones[0])
                    ep_r  += float(rew[0])
                    self.total_steps += 1

                    # step display env with same action for rendering
                    _, _, term, trunc, _ = display_env.step(action)
                    frame = display_env.render()
                    if frame is not None:
                        self._push(frame)
                    if term or trunc:
                        display_env.reset()

                    self.act_var.set(ACTION_NAMES[int(action)])
                    self.sv["SCORE"].set(f"{ep_r:.0f}")
                    self.sv["STEPS"].set(f"{self.total_steps:,}")
                    time.sleep(0.016)

                if self.running:
                    ep += 1
                    self.play_ep += 1
                    self.ep_rewards.append(ep_r)
                    best = max(self.ep_rewards)
                    self.sv["EPISODE"].set(str(self.play_ep))
                    self.sv["BEST"].set(f"{best:.0f}")
                    self.root.after(0, self._redraw)
                    self.status_var.set(
                        f"Ep {self.play_ep}  score={ep_r:.0f}  "
                        f"best={best:.0f}  "
                        f"mean={np.mean(self.ep_rewards):.1f}")

            model_env.close()
            display_env.close()
            self.running = False
            self.run_btn.config(text="  START  ", bg=ACC1, fg="#000")

        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            self.running = False
            self.run_btn.config(text="  START  ", bg=ACC1, fg="#000")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="dqn_latest.zip",
                   help="Use dqn_latest.zip to watch live training progress")
    p.add_argument("--episodes", type=int, default=0)
    args = p.parse_args()
    root = tk.Tk()
    PlayGUI(root, args)
    root.mainloop()