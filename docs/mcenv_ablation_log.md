# MCEnv Ablation & Iteration Log

This document records the complete sequence of experiments conducted in a single Claude Code session to systematically improve the mcenv Minecraft bridging RL environment from 0 targets/episode to ~7+ targets/episode.

## Starting Point

The session began with mcenv in a broken state: the agent was scoring 0.000 targets/episode. The codebase had accumulated many changes from earlier reward engineering attempts (sprint-jump rewards, facing bonuses, air bonuses, yaw reversal penalties, rot_pct action, etc.) that collectively prevented learning.

The goal: 7+ targets reached per episode in 1000 ticks, without increasing max_ticks or area size. The agent starts on a single block at y=2 over void and must bridge to navigation targets in a 50x50 area.

### Configuration at session start
- Action space: 9 heads {3,3,2,2,2,7,7,3,5} (7-yaw, 7-pitch, 3-way place, rot_pct)
- Observations: 163 (14 player state + 2 target + 147 block grid 7x3x7)
- Rewards: rw_block=5, rw_speed=50 (quadratic), rw_fall=-100 (0.1x in phase 1+), rw_target=100, facing bonus=0.01, air bonus=10x, yaw reversal=1.0, sprint-jump=200
- Training: gamma=0.99, horizon=128, lr=0.003, ent_coef=0.02, minibatch=16384, total_agents=2048
- Curriculum: phase 0 (blocks) -> phase 1 (blocks+navigation), phase 2 disabled
- require_ground=0 (speed/target rewards fire while airborne)
- Survival reward removed from code

## Phase 1: Initial Reward Ablation Study

### Approach
Set each reward to 0 one at a time, train for 200M steps, compare scores.

### Results (baseline = 0.000)

| Experiment | Score | Key Observation |
|---|---|---|
| Baseline (all rewards) | 0.000 | Sits on block, learns nothing |
| No speed | 0.000 | Same |
| No block | 0.000 | Same |
| No target | 0.000 | Same |
| **No reversal** | **0.201** | Learns to bridge + reach targets |
| No fall | 0.001 | Walks off immediately |
| No sprint-jump | 0.000 | Same |
| No facing | 0.022 | Some bridging |

### Key Finding
The yaw reversal penalty (rw_look_reversal=1.0) was the primary blocker. It punished rotational exploration in phase 0, preventing the agent from ever learning to turn, which is prerequisite for bridging toward targets. Moving it to phase 1+ only allowed phase 0 exploration.

## Phase 2: Ablation with Reversal Fixed (Phase 1+ Only)

With reversal gated to phase 1+, reran ablations. New baseline: 0.373.

| Experiment | Score | Key Observation |
|---|---|---|
| New baseline (reversal phase 1+) | 0.373 | Mixed bridging, some sprint-jump |
| No speed | 0.006 | Bridge spam, no direction |
| **No block** | **2.584** | Pure sneak-bridge, best score |
| No target | 0.191 | Slight drop |
| **No fall** | **0.716** | Braver exploration, 2x baseline |
| No sprint-jump | 0.128 | Less capable |
| No facing | 0.033 | Poor directionality |

### Key Findings
- Block reward was harmful for score (incentivized placing blocks anywhere, not navigating)
- Fall penalty too harsh (agent afraid to explore)
- Speed reward critical for directional signal
- Facing bonus important for orientation

### Combined reward attempt
Tried combining no-block + low-fall: score=0.000 (no reward signal in phase 0 without block reward). The block reward is needed to bootstrap learning in phase 0 but hurts in later phases.

## Phase 3: Reproducing Old 5+ Target Performance

The user pointed out that an earlier run (wandb d114grk0, tag "v32-256h4l-place-repeat") had achieved 5.35 targets. Launched subagents to investigate what was different about that run.

### Differences Found Between Current Code and Old 5+ Code

The old run used commit 0e36799d with unstaged changes. Subagents identified these differences:

**Config differences:**
- Old: rw_block=20, rw_speed=5 (linear), rw_fall=-100, rw_target=100
- Current: rw_block=5, rw_speed=50 (quadratic), plus many extra reward components
- Old: gamma=0.99, horizon=128
- Current: gamma=0.99, horizon=128 (same, but the ini file had 0.999/256 — the actual runs overrode these)

**Code differences:**
1. require_ground: old=1 (speed/target gated on on_ground), current=0
2. Fall penalty: old=full in all phases, current=0.1x in phase 1+
3. Speed reward: old=linear (delta), current=quadratic (delta^2)
4. Sprint-jump reward: old=none (0), current=200
5. Facing bonus: old=none, current=0.01
6. Air block bonus: old=none, current=10x
7. Yaw reversal: old=none, current=1.0 (phase 1+)
8. Survival reward: old=0.001/tick on ground, current=removed
9. Block weight phase 1: old=1.0, current=0.25
10. Action space: old=8 heads {3,3,2,2,2,11,7,2}, current=9 heads {3,3,2,2,2,7,7,3,5}
11. Observations: old=76 (11 player + 7x3x3 asymmetric grid), current=163 (14 player + 7x3x7 symmetric)
12. Phase 2: old=auto-unlock via EMA ratchet, current=disabled
13. RNG: old=rand_r (LCG), current=xorshift32

### Systematic Restoration

Restored each difference one at a time to match old code, measuring impact:

| Change restored | Score |
|---|---|
| Starting point (all new features) | 0.000 |
| + reversal phase 1 only | 0.373 |
| + old reward config (block=20, speed=5 linear, etc.) | 0.898 |
| + old action layout (8 heads, 11-yaw) | 1.070 |
| + old obs (76) + full old match + gamma=0.999/horizon=256 | 0.839 |
| + gamma=0.99, horizon=128 (matching actual wandb config) | **4.774** |

**Critical discovery:** gamma=0.99 + horizon=128 (2x more gradient updates per timestep) was the single biggest lever, jumping from 0.839 to 4.774.

## Phase 4: Feature Add-Back Ablation

Starting from the old-matching baseline (4.774), added each new feature back one at a time to see which help vs hurt.

| Feature added | Score | Delta |
|---|---|---|
| Baseline (old-matching) | 4.774 | — |
| + Yaw reversal (1.0, phase 1+) | 4.402 | -0.37 |
| + Facing bonus (0.01) | 4.768 | neutral |
| + Air block bonus (10x) | 4.765 | neutral |
| + Block weight phase1=0.25 | 4.637 | -0.14 |
| + Obs 163 (frac+7x3x7) | 4.817 | +0.04 |
| - Survival reward | 4.788 | neutral |

**Findings:** Most new features were neutral or slightly harmful. The obs expansion (163) was slightly helpful. Yaw reversal and block weight reduction hurt.

## Phase 5: Remaining Gap Investigation

Baseline at 4.774 vs old d114grk0 at 5.35 — still a 0.6 gap. Investigated remaining differences.

### Place action discovery
Subagent analysis of the conversation transcript revealed that d114grk0 had **unstaged changes** including a 3-way place action {no, single, repeat} (ACT_SIZES place=3) that wasn't in the committed code.

Verified by creating a git worktree at the old commit:
- Old commit, place=2 (committed code): 4.986
- Old commit, place=3 (with unstaged change): **5.336** (matches d114grk0's 5.345)

The 3-way place action was worth +0.35 targets.

### Additional ablations at this point

| Config | Score (200M) | Notes |
|---|---|---|
| Phase 2 enabled | **4.933** | Best at 200M |
| Phase 2 + require_ground=0 in ph2 | 4.000 | Ground gate helps |
| Phase 2 + Phase 3 (no speed) | 3.697 | Speed reward needed |
| Place {no, yes} (no repeat) | 4.211 | Repeat critical |
| Place {no, yes, repeat} (3-way) | 4.637 | Extra head slight hurt vs {no, repeat} |
| Block y=2-3 | 4.501 | Stacking hurts |

### Survival reward interaction with phase 2
- Baseline with survival, no phase 2: 4.774
- Baseline without survival, no phase 2: 3.422 (survival matters!)
- Phase 2 with survival: 3.754
- Phase 2 without survival: 4.933 (phase 2 compensates and then some)

Survival creates conflicting signal with phase 2 — it rewards staying still on ground while phase 2 wants navigation.

## Phase 6: GPU-Native Code at Scale

Combined best settings: BC actions (9 heads with rot_pct), phase 2, obs=163, no survival, 3-way place.

### 1B step runs
| Run | Score |
|---|---|
| GPU-native BC actions, 1B | **5.568** |
| Reproduction | 5.655 |
| Noise sample A | 5.661 |
| Noise sample B | 5.391 |
| Mean | 5.57 ± 0.12 |

### 200M noise characterization
Three runs with identical config: 4.946, 4.871, 5.001. Noise band ±0.07 at 200M.

## Phase 7: Reward Scaling

With the best config locked, tried ±25% on each reward:

| Reward | -25% | Baseline | +25% |
|---|---|---|---|
| rw_block (20) | 15 -> 4.739 | 4.946 | 25 -> 5.026 |
| rw_speed (5) | **3.75 -> 5.251** | 4.946 | 6.25 -> 4.505 |
| rw_target (100) | 75 -> 4.734 | 4.946 | 125 -> 5.047 |
| rw_fall (-100) | -75 -> 1.736 | 4.946 | **-125 -> 5.151** |

Lower speed (3.75) and harsher fall (-125) were best individually. Combined: speed=3.75 + fall=-125 = 5.085. Pushing further (speed=2.5, fall=-150) was too extreme (0.019). Diminishing returns from reward tuning.

## Phase 8: Action Space Comparison

| Layout | Score (200M) | Heads |
|---|---|---|
| **BC actions** (7-yaw + rot_pct) | **4.946** | 9 |
| 7-yaw no rot_pct | 4.806 | 8 |
| 25-yaw flat (all rot_pct combos in one head) | 4.594 | 8 |
| 11-yaw (old) | 4.390 | 8 |
| 153-yaw (MC_FULL_ANGLE) | 3.993 | 8 |

**rot_pct factorization wins.** Composing 7 directions x 5 multipliers across two small heads is more sample-efficient than a single head with equivalent values. The factored representation lets the network learn "direction" and "magnitude" independently.

## Phase 9: RNG Investigation

Attempted to match old rand_r by implementing equivalent LCG on GPU. Result: LCG slightly worse than xorshift32 (4.531 vs 4.785 at 1B), but confounded by action space difference. RNG effect is likely negligible.

## Phase 10: Sweep

Launched Protein (Bayesian optimization) sweep over 7 parameters: rw_block, rw_speed, rw_target, rw_fall, lr, ent_coef, gamma. The sweep found configurations significantly better than hand-tuned baselines.

### Best sweep run: 1e46cuax
Config found by sweep:
- rw_block=15.6, rw_speed=4.6, rw_fall=-151.8, rw_target=79.5
- gamma=0.98, lr=0.006, ent_coef=0.025
- gae_lambda=0.2, beta1=0.877, beta2=0.9998
- clip_coef=0.42, vf_coef=1.71, vf_clip_coef=0.01
- replay_ratio=2.22 (experience replay!)
- total_agents=1024, minibatch_size=4096
- num_layers=3, hidden_size=256
- total_timesteps=68M (with LR annealing to 0)

Original sweep result: **7.831 targets at 68M steps**

### Reproduction attempt
First attempt with 1B total_timesteps diverged (entropy -> -23000) because LR annealing schedule was wrong (LR stayed too high for too long). The original used total_timesteps=68M for LR annealing.

With correct 68M timesteps: 6.239 (gap from 7.831 due to missing minibatch_size=4096 — we used default 16384, so 4x fewer gradient updates per epoch).

Final reproduction with minibatch_size=4096: pending at time of writing.

## Key Lessons Learned

1. **Yaw reversal penalty kills exploration.** Any penalty on rotational exploration in phase 0 prevents the agent from learning to turn, which is prerequisite for everything else.

2. **require_ground=1 is critical.** Gating speed/target rewards on on_ground forces the agent to bridge (stay on blocks) rather than just running and falling.

3. **gamma and horizon are massive levers.** gamma=0.99 + horizon=128 (more frequent gradient updates) dramatically outperforms gamma=0.999 + horizon=256.

4. **3-way place action matters.** {no, single, repeat} is worth +0.35 targets vs {no, place}. The agent needs the choice between precise single placement and continuous bridging.

5. **rot_pct factorization is sample-efficient.** 7 directions x 5 multipliers (two small heads) beats a single head with 25 values representing the same angles.

6. **Phase 2 curriculum helps.** Transitioning to block_weight=0 after learning forces the agent to use blocks instrumentally, not for reward.

7. **Survival reward conflicts with phase 2.** It rewards staying still on ground, which opposes the navigation goal in later phases.

8. **Block reward is needed for bootstrapping but hurts navigation.** The curriculum (phase 0 -> phase 1 -> phase 2) manages this tradeoff.

9. **Observation space changes are mostly neutral.** 76 vs 163 obs had minimal impact on score at 200M, slightly positive at 1B.

10. **Sweep infrastructure finds non-obvious configs.** The sweep found that gae_lambda=0.2, experience replay (ratio 2.2), minibatch_size=4096, and num_layers=3 significantly outperform defaults. These are parameters we wouldn't have thought to change through manual ablation.

11. **LR annealing schedule must match.** Using a short total_timesteps (68M) with high LR (0.006) and annealing to 0 produces better results than a long run with slow annealing. The training "finishes" at the optimal point.

12. **Always check for unstaged changes.** The 5.35 result came from unstaged modifications that weren't in the committed code. The conversation transcript was the only record.

## Final Score Progression

| Milestone | Score | Key Change |
|---|---|---|
| Session start | 0.000 | Broken rewards |
| Reversal fix | 0.373 | Gated to phase 1+ |
| Old reward config | 0.898 | block=20, speed=5, linear |
| Old actions (8 heads) | 1.070 | 11-yaw, no rot_pct |
| gamma=0.99, horizon=128 | 4.774 | 2x gradient updates |
| Phase 2 enabled | 4.933 | block_weight=0 after ratchet |
| 3-way place + BC actions | 5.568 (1B) | rot_pct + place repeat |
| Sweep config (1e46cuax) | 7.831 | gae=0.2, replay, mb=4096 |

## Current Best Configuration

```ini
[env]
rw_block = 15.636
rw_speed = 4.591
rw_fall = -151.794
rw_target_reach = 79.548
rw_look_reversal = 0.0
rw_sprint_jump = 0.0
require_ground = 1
fall_scale_phase1 = 1.0
speed_power = 1.0
phase_transition = 30

[policy]
hidden_size = 256
num_layers = 3

[train]
gamma = 0.98
learning_rate = 0.006160
ent_coef = 0.02536
gae_lambda = 0.2
minibatch_size = 4096
horizon = 128
clip_coef = 0.42
vf_coef = 1.712
vf_clip_coef = 0.01
replay_ratio = 2.22
total_timesteps = 68000000

[vec]
total_agents = 1024
```

Action space: MC_BC_ACTIONS (9 heads: {3,3,2,2,2,7,7,3,5})
Observations: 163 (14 player + 2 target + 147 grid 7x3x7)
Phase 2 enabled with EMA ratchet (avg_targets_ema > 2.0)
No survival reward, no facing bonus, no air bonus, no yaw reversal
