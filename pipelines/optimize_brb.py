"""
基于现有 system_brb/module_brb 的轻量优化器，不依赖 brb_engine / brb_rules.yaml。
- 优化目标：模块层 8 条规则的权重（module_brb 原始权重被可调参数替换）。
- 系统层仍使用 system_brb.py 中的固定规则，权重不优化。
- 支持无监督（熵 + 置信度）和有监督（label_mod 监督）。

使用示例（无监督）:
    python optimize_brb.py --data feats.csv --maxiter 60

使用示例（有监督）:
    python optimize_brb.py --data feats.csv --label_col label_mod --supervised --maxiter 80
"""

import argparse
import numpy as np
import pandas as pd
import cma
from sklearn.metrics import log_loss, accuracy_score

from FMFD.BRB.system_brb import system_level_infer

# ------- 与 module_brb 一致的 labels 列表 -------
LABELS = [
    "衰减器",
    "前置放大器",
    "低频段前置低通滤波器",
    "低频段第一混频器",
    "高频段YTF滤波器",
    "高频段混频器",
    "时钟振荡器",
    "时钟合成与同步网络",
    "本振源（谐波发生器）",
    "本振混频组件",
    "校准源",
    "存储器",
    "校准信号开关",
    "中频放大器",
    "ADC",
    "数字RBW",
    "数字放大器",
    "数字检波器",
    "VBW滤波器",
    "电源模块",
    "未定义/其他",
]

# ------- 复制自 module_brb，但 rule weight 改为可调 --------
def module_level_infer_param(features, sys_probs, rule_weights):
    """
    rule_weights: 长度 8，对应原 module_brb 中 8 条 BRBRule 的 weight
    """
    def normalize_feature(x, low, high):
        if x <= low: return 0.0
        if x >= high: return 1.0
        return (x - low) / (high - low)

    md_step_raw = max(
        features["step_score"],
        features.get("switch_step_err_max", 0.0),
        features.get("nonswitch_step_max", 0.0),
    )
    md_step   = normalize_feature(md_step_raw, 0.2, 1.5)
    md_slope  = normalize_feature(abs(features["res_slope"]), 1e-12, 1e-10)
    md_ripple = normalize_feature(features["ripple_var"], 0.001, 0.02)
    md_df     = normalize_feature(abs(features["df"]), 1e6, 5e7)
    md_viol   = normalize_feature(features["viol_rate"], 0.02, 0.2)
    md_gain_bias = max(
        normalize_feature(abs(features["bias"]), 0.1, 1.0),
        normalize_feature(abs(features["gain"] - 1.0), 0.02, 0.2),
    )
    md = np.mean([md_step, md_slope, md_ripple, md_df, md_viol, md_gain_bias])

    # 规则的 belief 与原 module_brb 相同，仅 weight 可调
    rules = [
        (rule_weights[0] * sys_probs.get("参考电平失准", 0.3),
         {"衰减器": 0.60, "校准源": 0.08, "存储器": 0.06, "校准信号开关": 0.16, "未定义/其他": 0.10}),
        (rule_weights[1] * sys_probs.get("幅度失准", 0.3),
         {"前置放大器": 0.40, "中频放大器": 0.25, "数字放大器": 0.20, "衰减器": 0.10, "ADC": 0.05}),
        (rule_weights[2] * sys_probs.get("频率失准", 0.3),
         {"时钟振荡器": 0.35, "时钟合成与同步网络": 0.35, "本振源（谐波发生器）": 0.15, "本振混频组件": 0.15}),
        (rule_weights[3], {"高频段YTF滤波器": 0.60, "高频段混频器": 0.40}),
        (rule_weights[4], {"低频段前置低通滤波器": 0.60, "低频段第一混频器": 0.40}),
        (rule_weights[5], {"数字RBW": 0.30, "数字检波器": 0.35, "VBW滤波器": 0.25, "ADC": 0.10}),
        (rule_weights[6], {"电源模块": 0.80, "未定义/其他": 0.20}),
        (rule_weights[7], {"未定义/其他": 1.0}),
    ]

    acts = []
    for w, bel in rules:
        act = w * md  # 这里沿用 SimpleBRB 的“匹配度相乘”思路，简化为 w*md
        acts.append((act, bel))
    total = sum(a for a, _ in acts) + 1e-9
    out = {lab: 0.0 for lab in LABELS}
    for a, bel in acts:
        for lab in LABELS:
            out[lab] += (a / total) * bel.get(lab, 0.0)
    s = sum(out.values()) + 1e-9
    for lab in LABELS:
        out[lab] = out[lab] / s
    return out

# ------- 目标函数 -------
def unsupervised_objective(weights, feats_df, w_entropy=0.6, w_conf=0.4):
    probs = []
    for _, row in feats_df.iterrows():
        f = row.to_dict()
        sys_p = system_level_infer(f)  # 使用固定系统层
        mod_p = module_level_infer_param(f, sys_p, weights)
        probs.append([mod_p[lab] for lab in LABELS])
    probs = np.array(probs)
    eps = 1e-12
    ent = -np.sum(probs * np.log(np.clip(probs, eps, 1.0)), axis=1)
    mean_ent = float(np.nanmean(ent))
    mean_top1 = float(np.nanmean(np.max(probs, axis=1)))
    return w_entropy * mean_ent + w_conf * (1.0 - mean_top1)

def supervised_objective(weights, feats_df, label_col):
    probs = []
    labels = []
    for _, row in feats_df.iterrows():
        f = row.to_dict()
        sys_p = system_level_infer(f)
        mod_p = module_level_infer_param(f, sys_p, weights)
        probs.append([mod_p[lab] for lab in LABELS])
        labels.append(row[label_col])
    probs = np.array(probs)
    y = np.array(labels)
    # 仅保留标签在 LABELS 中的样本
    mask = np.array([lab in LABELS for lab in y])
    if not mask.any():
        return 1e6
    probs = probs[mask]
    y = y[mask]
    idx_map = {m: i for i, m in enumerate(LABELS)}
    y_idx = np.array([idx_map[v] for v in y])
    try:
        loss = log_loss(y_idx, probs, labels=list(range(len(LABELS))))
    except Exception:
        loss = 1.0 - accuracy_score(y_idx, np.argmax(probs, axis=1))
    return float(loss)

# ------- CMA-ES 优化主流程 -------
def optimize(feats_df, supervised=False, label_col=None, maxiter=80, popsize=None, seed=42, sigma0=0.3):
    # 初始权重 8 个，全为 0.5（可根据你原始权重设置 0.8/0.6/0.7/0.5/0.5/0.4/0.3/0.2）
    x0 = np.array([0.8,0.6,0.7,0.5,0.5,0.4,0.3,0.2], dtype=float)
    opts = {"seed": seed}
    if popsize:
        opts["popsize"] = popsize
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_x, best_obj = None, float("inf")
    it = 0
    while not es.stop() and it < maxiter:
        sols = es.ask()
        objs = []
        for x in sols:
            w = np.clip(x, 0.01, 3.0)  # 约束权重范围，避免负数
            try:
                if supervised and label_col:
                    obj = supervised_objective(w, feats_df, label_col)
                else:
                    obj = unsupervised_objective(w, feats_df)
            except Exception:
                obj = float("inf")
            objs.append(obj)
            if obj < best_obj:
                best_obj, best_x = obj, w.copy()
        es.tell(sols, objs)
        es.disp()
        it += 1

    if best_x is None:
        best_x = np.clip(es.result.xbest, 0.01, 3.0)
        best_obj = float(es.result.fbest)

    print(f"[INFO] best_obj={best_obj:.6f}, best_weights={best_x}")
    return best_x, best_obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="特征 CSV，列需含 gain,bias,comp,df,step_score,viol_rate,res_slope,ripple_var,switch_step_err_max,nonswitch_step_max")
    ap.add_argument("--label_col", default=None, help="有监督时的标签列（模块中文名，见 LABELS）")
    ap.add_argument("--supervised", action="store_true", help="开启有监督优化")
    ap.add_argument("--maxiter", type=int, default=80)
    ap.add_argument("--popsize", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma0", type=float, default=0.3)
    args = ap.parse_args()

    feats_df = pd.read_csv(args.data)
    # 过滤掉不含必需特征的行
    required = ["gain","bias","comp","df","step_score","viol_rate","res_slope","ripple_var","switch_step_err_max","nonswitch_step_max"]
    for col in required:
        if col not in feats_df.columns:
            raise ValueError(f"缺少特征列: {col}")

    best_w, best_obj = optimize(
        feats_df=feats_df,
        supervised=args.supervised,
        label_col=args.label_col,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        sigma0=args.sigma0,
    )

    # 保存结果
    out_path = "optimized_module_rule_weights.txt"
    np.savetxt(out_path, best_w, fmt="%.6f")
    print(f"[INFO] 最优权重已保存到 {out_path}")

if __name__ == "__main__":
    main()