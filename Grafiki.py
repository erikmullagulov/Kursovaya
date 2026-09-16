import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


COLOR_MM = "steelblue"
COLOR_MLE = "crimson"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "Grafiki")
os.makedirs(OUT, exist_ok=True)

df_homo = pd.read_csv(os.path.join(BASE_DIR, "simulation_results.csv"))
df_hetero = pd.read_csv(os.path.join(BASE_DIR, "simulation_results_hetero.csv"))
df_follow = pd.read_csv(os.path.join(BASE_DIR, "simulation_proverka_plateau.csv"))


def rmse(series):
    return np.sqrt((series ** 2).mean())


mm_homo = df_homo[df_homo.method == "MM"]

n_list = [20, 50, 100, 300, 1000]

weak_success = []
medium_success = []
strong_success = []
for n in n_list:
    weak_success.append(mm_homo[(mm_homo.delta_label == "weak") & (mm_homo.n == n)]["success"].mean() * 100)
    medium_success.append(mm_homo[(mm_homo.delta_label == "medium") & (mm_homo.n == n)]["success"].mean() * 100)
    strong_success.append(mm_homo[(mm_homo.delta_label == "strong") & (mm_homo.n == n)]["success"].mean() * 100)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n_list, weak_success, marker="o", label="δ=weak", color="peru")
ax.plot(n_list, medium_success, marker="o", label="δ=medium", color="rebeccapurple")
ax.plot(n_list, strong_success, marker="o", label="δ=strong", color="seagreen")
ax.axhline(100, color=COLOR_MLE, linestyle="--", linewidth=1.5, label="ММП")
ax.set_xscale("log")
ax.set_xlabel("Sample size")
ax.set_ylabel("Success rate, %")
ax.set_ylim(0, 105)
ax.legend(loc="lower left")
fig.savefig(os.path.join(OUT, "fig1_success_rate_by_delta.png"))
plt.close(fig)


plateau_n_list = [1500, 2000, 3000, 5000]

balanced_n = n_list + plateau_n_list
balanced_success = []
for n in n_list:
    sub = df_homo[(df_homo.method == "MM") & (df_homo.delta_label == "weak") & (df_homo.p_label == "balanced") & (df_homo.n == n)]
    balanced_success.append(sub["success"].mean() * 100)
for n in plateau_n_list:
    sub = df_follow[(df_follow.scenario == "plateau") & (df_follow.method == "MM") & (df_follow.p_label == "balanced") & (df_follow.n == n)]
    balanced_success.append(sub["success"].mean() * 100)

unbalanced_n = n_list + plateau_n_list
unbalanced_success = []
for n in n_list:
    sub = df_homo[(df_homo.method == "MM") & (df_homo.delta_label == "weak") & (df_homo.p_label == "unbalanced") & (df_homo.n == n)]
    unbalanced_success.append(sub["success"].mean() * 100)
for n in plateau_n_list:
    sub = df_follow[(df_follow.scenario == "plateau") & (df_follow.method == "MM") & (df_follow.p_label == "unbalanced") & (df_follow.n == n)]
    unbalanced_success.append(sub["success"].mean() * 100)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(balanced_n, balanced_success, marker="o", label="p=balanced", color="cornflowerblue")
ax.plot(unbalanced_n, unbalanced_success, marker="o", label="p=unbalanced", color="orangered")
ax.set_xscale("log")
ax.set_xlabel("Sample size")
ax.set_ylabel("Success rate ММ, %")
ax.axvspan(1000, 5000, alpha=0.08, color="gray")
ax.text(1900, 8, "зона плато\n(n=1000…5000)", fontsize=9, color="gray", ha="center")
ax.set_ylim(0, 80)
ax.legend()
fig.savefig(os.path.join(OUT, "fig2_plateau_weak.png"))
plt.close(fig)


cross = df_follow[df_follow.scenario == "crossover"]
cross_n_list = [100, 150, 200, 250, 300, 400, 500, 700, 1000]

balanced_mm_mae = []
balanced_mle_mae = []
for n in cross_n_list:
    sub_mm = cross[(cross.p_label == "balanced") & (cross.method == "MM") & (cross.n == n) & cross.success]
    sub_mle = cross[(cross.p_label == "balanced") & (cross.method == "MLE") & (cross.n == n) & cross.success]
    balanced_mm_mae.append(sub_mm["mu1_abs_error"].mean())
    balanced_mle_mae.append(sub_mle["mu1_abs_error"].mean())

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(cross_n_list, balanced_mm_mae, marker="o", label="ММ", color=COLOR_MM)
ax.plot(cross_n_list, balanced_mle_mae, marker="o", label="ММП", color=COLOR_MLE)
ax.set_xscale("log")
ax.set_xlabel("Sample size")
ax.set_ylabel("MAE(μ₁)")
ax.legend()
fig.savefig(os.path.join(OUT, "fig3_crossover_balanced.png"))
plt.close(fig)


unb = cross[cross.p_label == "unbalanced"]

unb_mm_mu1 = []
unb_mle_mu1 = []
unb_mm_mu2 = []
unb_mle_mu2 = []
for n in cross_n_list:
    sub_mm = unb[(unb.method == "MM") & (unb.n == n) & unb.success]
    sub_mle = unb[(unb.method == "MLE") & (unb.n == n) & unb.success]
    unb_mm_mu1.append(sub_mm["mu1_abs_error"].mean())
    unb_mle_mu1.append(sub_mle["mu1_abs_error"].mean())
    unb_mm_mu2.append(sub_mm["mu2_abs_error"].mean())
    unb_mle_mu2.append(sub_mle["mu2_abs_error"].mean())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(cross_n_list, unb_mm_mu1, marker="o", label="ММ", color=COLOR_MM)
axes[0].plot(cross_n_list, unb_mle_mu1, marker="o", label="ММП", color=COLOR_MLE)
axes[0].set_xscale("log")
axes[0].set_xlabel("Sample size")
axes[0].set_ylabel("MAE")
axes[0].legend()

axes[1].plot(cross_n_list, unb_mm_mu2, marker="o", label="ММ", color=COLOR_MM)
axes[1].plot(cross_n_list, unb_mle_mu2, marker="o", label="ММП", color=COLOR_MLE)
axes[1].set_xscale("log")
axes[1].set_xlabel("Sample size")
axes[1].set_ylabel("MAE")
axes[1].legend()

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_crossover_unbalanced.png"))
plt.close(fig)


mm_hetero_strong = df_hetero[(df_hetero.method == "MM") & (df_hetero.delta_label == "strong")]

ratio_labels = ["ratio_2", "ratio_4", "ratio_9"]
ratio_x = [2, 4, 9]

balanced_ratio_success = []
unbalanced_ratio_success = []
for r in ratio_labels:
    sub_b = mm_hetero_strong[(mm_hetero_strong.p_label == "balanced") & (mm_hetero_strong.ratio_label == r)]
    sub_u = mm_hetero_strong[(mm_hetero_strong.p_label == "unbalanced") & (mm_hetero_strong.ratio_label == r)]
    balanced_ratio_success.append(sub_b["success"].mean() * 100)
    unbalanced_ratio_success.append(sub_u["success"].mean() * 100)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ratio_x, balanced_ratio_success, marker="o", label="p=balanced", color="cornflowerblue")
ax.plot(ratio_x, unbalanced_ratio_success, marker="o", label="p=unbalanced", color="orangered")
ax.set_xlabel("Отношение дисперсий σ₂²/σ₁²")
ax.set_ylabel("Success rate ММ, %")
ax.set_xticks(ratio_x)
ax.legend()
fig.savefig(os.path.join(OUT, "fig5_hetero_success_by_ratio.png"))
plt.close(fig)


succ_hetero = df_hetero[df_hetero.success]

mm_sigma_mae = []
mle_sigma_mae = []
for r in ratio_labels:
    sub_mm = succ_hetero[(succ_hetero.method == "MM") & (succ_hetero.ratio_label == r)]
    sub_mle = succ_hetero[(succ_hetero.method == "MLE") & (succ_hetero.ratio_label == r)]
    mm_sigma_mae.append(sub_mm["sigma1_sq_abs_error"].mean())
    mle_sigma_mae.append(sub_mle["sigma1_sq_abs_error"].mean())

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(ratio_x, mm_sigma_mae, marker="o", label="ММ", color=COLOR_MM)
ax.plot(ratio_x, mle_sigma_mae, marker="o", label="ММП", color=COLOR_MLE)
ax.set_xlabel("Отношение дисперсий σ₂²/σ₁²")
ax.set_ylabel("MAE(σ₁²)")
ax.set_xticks(ratio_x)
ax.legend()
fig.savefig(os.path.join(OUT, "fig6_hetero_sigma_error.png"))
plt.close(fig)


homo_mm_rate = df_homo[df_homo.method == "MM"]["success"].mean() * 100
homo_mle_rate = df_homo[df_homo.method == "MLE"]["success"].mean() * 100
hetero_mm_rate = df_hetero[df_hetero.method == "MM"]["success"].mean() * 100
hetero_mle_rate = df_hetero[df_hetero.method == "MLE"]["success"].mean() * 100

scenario_labels = ["Гомоскедастичный", "Гетероскедастичный"]
mm_rates = [homo_mm_rate, hetero_mm_rate]
mle_rates = [homo_mle_rate, hetero_mle_rate]

x = np.arange(len(scenario_labels))
width = 0.42

succ_homo = df_homo[df_homo.success]

param_labels = ["p", "μ₁", "μ₂", "σ₁²", "σ₂²"]
param_cols_mae = ["p_abs_error", "mu1_abs_error", "mu2_abs_error", "sigma1_sq_abs_error", "sigma2_sq_abs_error"]

mm_mae_values = []
mle_mae_values = []
for col in param_cols_mae:
    mm_mae_values.append(succ_homo[succ_homo.method == "MM"][col].mean())
    mle_mae_values.append(succ_homo[succ_homo.method == "MLE"][col].mean())

x = np.arange(len(param_labels))
width = 0.42

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, mm_mae_values, width, label="ММ", color=COLOR_MM, alpha=0.85)
ax.bar(x + width / 2, mle_mae_values, width, label="ММП", color=COLOR_MLE, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(param_labels)
ax.set_ylabel("MAE")
ax.legend()
fig.savefig(os.path.join(OUT, "fig7_overall_mae.png"))
plt.close(fig)


param_cols_signed = ["p_error", "mu1_error", "mu2_error", "sigma1_sq_error", "sigma2_sq_error"]

mm_rmse_values = []
mle_rmse_values = []
for col in param_cols_signed:
    mm_rmse_values.append(rmse(succ_homo[succ_homo.method == "MM"][col]))
    mle_rmse_values.append(rmse(succ_homo[succ_homo.method == "MLE"][col]))

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width / 2, mm_rmse_values, width, label="ММ", color=COLOR_MM, alpha=0.85)
ax.bar(x + width / 2, mle_rmse_values, width, label="ММП", color=COLOR_MLE, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(param_labels)
ax.set_ylabel("RMSE")
ax.legend()
fig.savefig(os.path.join(OUT, "fig8_overall_rmse.png"))
plt.close(fig)


delta_list = ["weak", "medium", "strong"]
p_list = ["balanced", "unbalanced"]

heatmap_rows_11 = []
row_labels_11 = []
for delta_label in delta_list:
    for p_label in p_list:
        row_values = []
        for n in n_list:
            sub = mm_homo[(mm_homo.delta_label == delta_label) & (mm_homo.p_label == p_label) & (mm_homo.n == n)]
            row_values.append(sub["success"].mean() * 100)
        heatmap_rows_11.append(row_values)
        row_labels_11.append("δ=" + delta_label + ", p=" + p_label)

data_11 = np.array(heatmap_rows_11)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(data_11, cmap="RdBu", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(n_list)))
ax.set_xticklabels(n_list)
ax.set_yticks(range(len(row_labels_11)))
ax.set_yticklabels(row_labels_11)
ax.set_xlabel("Sample size")
for i in range(data_11.shape[0]):
    for j in range(data_11.shape[1]):
        value = data_11[i, j]
        text_color = "white" if value < 50 else "black"
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=text_color, fontsize=10)
fig.colorbar(im, ax=ax, label="Success rate, %")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_heatmap_success_homo.png"))
plt.close(fig)


mm_hetero_all = df_hetero[df_hetero.method == "MM"]
hetero_n_list = [50, 300, 1000]
ratio_display = ["2", "4", "9"]

heatmap_rows_12 = []
row_labels_12 = []
for i in range(len(ratio_labels)):
    ratio_label = ratio_labels[i]
    ratio_text = ratio_display[i]
    for delta_label in delta_list:
        for p_label in p_list:
            row_values = []
            for n in hetero_n_list:
                sub = mm_hetero_all[(mm_hetero_all.delta_label == delta_label) & (mm_hetero_all.p_label == p_label)
                                     & (mm_hetero_all.ratio_label == ratio_label) & (mm_hetero_all.n == n)]
                row_values.append(sub["success"].mean() * 100)
            heatmap_rows_12.append(row_values)
            row_labels_12.append("ratio=" + ratio_text + ", δ=" + delta_label + ", p=" + p_label)

data12 = np.array(heatmap_rows_12)

fig, ax = plt.subplots(figsize=(7.5, 10))
im = ax.imshow(data12, cmap="RdBu", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(len(hetero_n_list)))
ax.set_xticklabels(hetero_n_list)
ax.set_yticks(range(len(row_labels_12)))
ax.set_yticklabels(row_labels_12, fontsize=9)
ax.set_xlabel("Sample size")
for i in range(data12.shape[0]):
    for j in range(data12.shape[1]):
        value = data12[i, j]
        text_color = "white" if value < 50 else "black"
        ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=text_color, fontsize=9)
fig.colorbar(im, ax=ax, label="Success rate, %", shrink=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10_heatmap_success_hetero.png"))
plt.close(fig)


heatmap_rows_13 = []
row_labels_13 = []
for delta_label in delta_list:
    for p_label in p_list:
        row_values = []
        for n in n_list:
            sub = succ_homo[(succ_homo.delta_label == delta_label) & (succ_homo.p_label == p_label) & (succ_homo.n == n)]
            mae_mm = sub[sub.method == "MM"]["mu1_abs_error"].mean()
            mae_mle = sub[sub.method == "MLE"]["mu1_abs_error"].mean()
            row_values.append(mae_mm / mae_mle)
        heatmap_rows_13.append(row_values)
        row_labels_13.append("δ=" + delta_label + ", p=" + p_label)

data13 = np.array(heatmap_rows_13)

fig, ax = plt.subplots(figsize=(7, 6))
norm13 = mcolors.TwoSlopeNorm(vmin=max(0.3, data13.min() * 0.9), vcenter=1.0, vmax=min(3.0, data13.max() * 1.1))
im = ax.imshow(data13, cmap="RdBu_r", norm=norm13, aspect="auto")
ax.set_xticks(range(len(n_list)))
ax.set_xticklabels(n_list)
ax.set_yticks(range(len(row_labels_13)))
ax.set_yticklabels(row_labels_13)
ax.set_xlabel("Объём выборки n")
for i in range(data13.shape[0]):
    for j in range(data13.shape[1]):
        value = data13[i, j]
        text_color = "white" if abs(value - 1) > 0.5 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=10)
fig.colorbar(im, ax=ax, label="MAE(ММ) / MAE(ММП)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig11_heatmap_ratio_homo.png"))
plt.close(fig)


heatmap_rows_14 = []
row_labels_14 = []
for i in range(len(ratio_labels)):
    ratio_label = ratio_labels[i]
    ratio_text = ratio_display[i]
    for delta_label in delta_list:
        for p_label in p_list:
            row_values = []
            for n in hetero_n_list:
                sub = succ_hetero[(succ_hetero.delta_label == delta_label) & (succ_hetero.p_label == p_label)
                                   & (succ_hetero.ratio_label == ratio_label) & (succ_hetero.n == n)]
                mae_mm = sub[sub.method == "MM"]["mu1_abs_error"].mean()
                mae_mle = sub[sub.method == "MLE"]["mu1_abs_error"].mean()
                row_values.append(mae_mm / mae_mle)
            heatmap_rows_14.append(row_values)
            row_labels_14.append("ratio=" + ratio_text + ", δ=" + delta_label + ", p=" + p_label)

data14 = np.array(heatmap_rows_14)

fig, ax = plt.subplots(figsize=(7.5, 10))
offset = mcolors.TwoSlopeNorm(vmin=max(0.3, data14.min() * 0.9), vcenter=1.0, vmax=min(3.0, data14.max() * 1.1))
im = ax.imshow(data14, cmap="RdBu_r", norm=offset, aspect="auto")
ax.set_xticks(range(len(hetero_n_list)))
ax.set_xticklabels(hetero_n_list)
ax.set_yticks(range(len(row_labels_14)))
ax.set_yticklabels(row_labels_14, fontsize=9)
ax.set_xlabel("Sample size")
for i in range(data14.shape[0]):
    for j in range(data14.shape[1]):
        value = data14[i, j]
        text_color = "white" if abs(value - 1) > 0.5 else "black"
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9)
fig.colorbar(im, ax=ax, label="MAE(ММ) / MAE(ММП)", shrink=0.6)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig12_heatmap_ratio_hetero.png"))
plt.close(fig)