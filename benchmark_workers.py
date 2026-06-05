"""Quick benchmark: sequential vs parallel episode collection."""
import time, os, io, multiprocessing as mp
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

import src.config as _cfg
_cfg.EVAL_INTERVAL = 9999
_cfg.SNAPSHOT_INTERVAL = 9999
_cfg.WARMUP_GAMES = 0

import torch
from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.training.replay_buffer import ReplayBuffer
from src.training.self_play import play_episode
from src.training.train import _episode_worker
from src.config import STATE_CHANNELS_V2, NETWORK_ARCH

N = 40   # games per test

def get_bytes(agent):
    buf = io.BytesIO(); torch.save(agent.state_dict(), buf); return buf.getvalue()

def run_sequential():
    agent  = DQNAgent(8); buffer = ReplayBuffer(5000)
    from src.game.env import GameEnv
    env = GameEnv(8, "first_to_four")
    t0 = time.monotonic()
    for i in range(N):
        t1, t2, _ = play_episode(agent, RandomAgent(), env)
        for t in t1:
            buffer.push(t.state, t.action, t.reward, t.next_state,
                        t.done, t.legal_mask_next, gamma_n=t.gamma_n)
        if len(buffer) >= 128:
            for _ in range(4): agent.update(buffer.sample(128))
    return time.monotonic() - t0

def run_parallel(n_workers):
    agent  = DQNAgent(8); buffer = ReplayBuffer(5000)
    ab = get_bytes(agent)
    t0 = time.monotonic()
    with mp.Pool(n_workers) as pool:
        for i in range(0, N, n_workers):
            if i % 10 == 0: ab = get_bytes(agent)
            args = [(ab, 8, "first_to_four", 0.5, None, "random",
                     i+w, STATE_CHANNELS_V2, NETWORK_ARCH)
                    for w in range(n_workers)]
            results = pool.map(_episode_worker, args)
            for r_raw, *_ in results:
                for r in r_raw:
                    buffer.push(r["state"], r["action"], r["reward"],
                                r["next_state"], r["done"], r["legal_mask_next"],
                                gamma_n=r["gamma_n"])
            if len(buffer) >= 128:
                for _ in range(4 * n_workers): agent.update(buffer.sample(128))
    return time.monotonic() - t0

if __name__ == "__main__":
    print("Sequential (%d games)..." % N, flush=True)
    s = run_sequential()
    seq_gph = N / s * 3600
    print("  %.1fs  ->  %.0f g/h" % (s, seq_gph))

    for w in [2, 4]:
        print("Parallel  (workers=%d, %d games)..." % (w, N), flush=True)
        p = run_parallel(w)
        par_gph = N / p * 3600
        print("  %.1fs  ->  %.0f g/h  (%.1fx speedup)" % (p, par_gph, par_gph/seq_gph))
