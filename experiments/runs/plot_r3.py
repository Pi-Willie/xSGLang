import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

r3_eval = load("branch_r3/eval.jsonl")
r3_m = load("branch_r3/metrics.jsonl")
v2_eval = load("branch_v2/eval.jsonl")

# eval series (held-out greedy)
eu = [d["update_id"] for d in r3_eval]
eacc = [d["greedy_accuracy"] for d in r3_eval]
einv = [d["invalid_format_rate"] for d in r3_eval]
elen = [d["mean_response_length"] for d in r3_eval]

# per-update training metrics
mu = [d["update_id"] for d in r3_m]
rew = [d.get("accuracy_per_verifier_call", np.nan) for d in r3_m]
dis = [d.get("branch_mixed_node_rate_all", np.nan) for d in r3_m]
gn  = [d.get("grad_norm", np.nan) for d in r3_m]

def roll(x, k=10):
    x = np.array(x, float); out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i-k+1); out[i] = np.nanmean(x[lo:i+1])
    return out

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Branch-Dr.GRPO Qwen3-4B  Round 3  (root_samples 4->8, 64 leaves/prompt)\n"
             "RL'd from Round-2 best (0.309 start)", fontsize=13, fontweight="bold")

# 1) Headline: held-out greedy accuracy
a = ax[0,0]
a.plot(eu, eacc, "o-", lw=2.4, ms=8, color="#1f77b4", label="r3 held-out greedy@1 (N=256)")
a.axhline(0.309, ls="--", c="#888", label="Round-2 best 0.309 (start)")
a.axhspan(0.309-0.055, 0.309+0.055, color="#ccc", alpha=0.35, label="noise band 2sigma")
best_i = int(np.argmax(eacc))
a.annotate(f"best u{eu[best_i]}={eacc[best_i]:.3f}", (eu[best_i], eacc[best_i]),
           textcoords="offset points", xytext=(-10,12), fontweight="bold", color="#1f77b4")
a.set_title("Held-out greedy accuracy (the headline)"); a.set_xlabel("update"); a.set_ylabel("greedy acc")
a.set_ylim(0.25, 0.42); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower right")

# 2) Format validity + response length
a = ax[0,1]
a.plot(eu, [1-x for x in einv], "s-", lw=2.2, ms=7, color="#2ca02c", label="format-valid rate")
a.set_title("Format adherence + trace length"); a.set_xlabel("update")
a.set_ylabel("format-valid rate", color="#2ca02c"); a.set_ylim(0.8, 0.95); a.grid(alpha=0.3)
a2 = a.twinx(); a2.plot(eu, elen, "^--", lw=1.8, ms=6, color="#d62728", label="mean resp len (tok)")
a2.set_ylabel("mean response length (tok)", color="#d62728")
a.legend(fontsize=8, loc="upper left"); a2.legend(fontsize=8, loc="upper right")

# 3) Training reward (per-verifier-call accuracy), rolling
a = ax[1,0]
a.plot(mu, rew, color="#bbb", lw=0.8, alpha=0.7, label="per-update (noisy)")
a.plot(mu, roll(rew,10), color="#ff7f0e", lw=2.6, label="10-update rolling mean")
a.set_title("Training reward  (accuracy per verifier call, temp=1 samples)")
a.set_xlabel("update"); a.set_ylabel("reward"); a.grid(alpha=0.3); a.legend(fontsize=8, loc="lower right")

# 4) Branch sibling-disagreement (the load-bearing signal)
a = ax[1,1]
a.plot(mu, dis, color="#cbb", lw=0.8, alpha=0.7)
a.plot(mu, roll(dis,10), color="#9467bd", lw=2.6, label="mixed-node rate (10-upd roll)")
a.axhline(0.15, ls="--", c="#888", label="root=4 level (~0.15)")
a.set_title("Branch sibling-disagreement (LOO signal)\nrose to ~0.25-0.29 with richer sampling")
a.set_xlabel("update"); a.set_ylabel("mixed-node rate"); a.grid(alpha=0.3)
a.set_ylim(0, 0.4); a.legend(fontsize=8, loc="upper right")

plt.tight_layout(rect=[0,0,1,0.95])
plt.savefig("r3_training_curves.png", dpi=130)
print("saved r3_training_curves.png")
print("eval acc:", [f"u{u}={a:.3f}" for u,a in zip(eu,eacc)])
print("best:", f"u{eu[best_i]}={eacc[best_i]:.4f}")
