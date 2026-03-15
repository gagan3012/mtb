# import json
# import matplotlib.pyplot as plt
# import numpy as np
# import scienceplots

# # Set style for two-column publication
# plt.style.use(['science'])
# plt.rcParams['figure.figsize'] = (7, 2.5)
# plt.rcParams['font.size'] = 8

# # Load the data
# with open('results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/analysis_summary.json', 'r') as f:
#     data = json.load(f)

# # Extract layer numbers
# layers = [layer_data['layer'] for layer_data in data['layers']]
# languages = data['languages']

# # Extract temporal linearity for each language
# temporal_linearity = {lang: [] for lang in languages}
# for layer_data in data['layers']:
#     for lang in languages:
#         temporal_linearity[lang].append(layer_data['temporal_linearity'][lang])

# # Extract calendar disentanglement metrics
# # Old metric (1 - avg|cos|)
# calendar_disentanglement = [layer_data['avg_calendar_disentanglement'] for layer_data in data['layers']]

# # New geometric metric: Parallelepiped Volume (Gram determinant)
# # m = sqrt(det(G)) where G = X^T X, X = [v_year, v_month, v_day] normalized
# # m = 0: collapsed (directions linearly dependent)
# # m = 1: unit cube (directions mutually orthogonal)
# # parallelepiped_volume = [
# #     layer_data.get('avg_parallelepiped_volume', layer_data.get('avg_calendar_disentanglement', 0)) 
# #     for layer_data in data['layers']
# # ]
# parallel_volume = {lang: [] for lang in languages}
# for layer_data in data['layers']:
#     for lang in languages:
#         parallel_volume[lang].append(layer_data['parallelepiped_volume_per_lang'][lang])

# # Create figure with two subplots for two-column layout
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 3))

# # Plot 1: Temporal Linearity
# colors = plt.cm.Set1(np.linspace(0, 1, len(languages)))
# for idx, lang in enumerate(languages):
#     ax1.plot(layers, temporal_linearity[lang], marker='o', label=lang.upper(), 
#              color=colors[idx], linewidth=0.8, markersize=2)

# ax1.set_xlabel('Layer')
# ax1.set_ylabel('Temporal Linearity')
# ax1.set_title('(a) Temporal Linearity', loc='left')
# ax1.legend(loc='best', frameon=True, fontsize=6, ncol=2)
# ax1.set_xlim(0, max(layers))

# # Plot 2: Calendar Disentanglement (Parallelepiped Volume)
# # Use the geometric metric (Gram determinant volume)
# for idx, lang in enumerate(languages):
#     ax2.plot(layers, parallel_volume[lang], marker='o', label=f'{lang.upper()}',
#              color=colors[idx], linewidth=0.8, markersize=2)
# # # Optionally show old metric for comparison
# # ax2.plot(layers, calendar_disentanglement, marker='s', color='C1', 
# #          linewidth=0.8, markersize=2, alpha=0.5, label='1-avg|cos|')

# ax2.set_xlabel('Layer')
# ax2.set_ylabel('Parallelepiped Volume')
# ax2.set_title('(b) Calendar Disentanglement', loc='left')
# ax2.set_xlim(0, max(layers))
# ax2.set_ylim(0, 1.05)
# ax2.legend(loc='best', frameon=True, fontsize=6, ncol=2)
# # Add annotation for interpretation
# ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
# ax2.text(max(layers)*0.02, 0.95, 'm=1: orthogonal', fontsize=5, alpha=0.5)

# plt.tight_layout()
# plt.savefig('results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/layer_progression.png', 
#             dpi=600, bbox_inches='tight')
# plt.savefig('results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/layer_progression.pdf', 
#             bbox_inches='tight')
# print("✓ Saved plot to: results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/layer_progression.png")
# print("✓ Saved plot to: results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/layer_progression.pdf")

# # # Also create a single-plot combined view for single column
# # fig2, ax = plt.subplots(figsize=(3.5, 2.5))

# # # Plot average temporal linearity
# # avg_temporal_linearity = [np.mean([temporal_linearity[lang][i] for lang in languages]) 
# #                           for i in range(len(layers))]

# # ax_twin = ax.twinx()

# # # Temporal linearity on left axis
# # line1 = ax.plot(layers, avg_temporal_linearity, marker='o', color='C1', 
# #                 linewidth=0.8, markersize=2, label='Temporal Linearity')
# # ax.fill_between(layers, avg_temporal_linearity, alpha=0.2, color='C1')

# # # Parallelepiped volume (Gram determinant) on right axis
# # line2 = ax_twin.plot(layers, parallelepiped_volume, marker='s', color='C0', 
# #                      linewidth=0.8, markersize=2, label='Gram Volume (m)')
# # ax_twin.fill_between(layers, parallelepiped_volume, alpha=0.2, color='C0')

# # ax.set_xlabel('Layer')
# # ax.set_ylabel('Temporal Linearity', color='C1')
# # ax_twin.set_ylabel('Parallelepiped Volume (m)', color='C0')

# # ax.tick_params(axis='y', labelcolor='C1')
# # ax_twin.tick_params(axis='y', labelcolor='C0')

# # ax.set_xlim(0, max(layers))
# # ax_twin.set_ylim(0, 1.05)

# # Combine legends
# # lines = line1 + line2
# # labels = [l.get_label() for l in lines]
# # ax.legend(lines, labels, loc='best', frameon=True, fontsize=6)

# # plt.tight_layout()
# # plt.savefig('results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/layer_progression_combined.png', 
# #             dpi=600, bbox_inches='tight')
# # plt.savefig('results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/layer_progression_combined.pdf', 
# #             bbox_inches='tight')
# # print("✓ Saved combined plot to: results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/layer_progression_combined.png")
# # print("✓ Saved combined plot to: results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/layer_progression_combined.pdf")

# # plt.show()

import json
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Set style for two-column publication
plt.style.use(['science'])
plt.rcParams['figure.figsize'] = (7, 4)
plt.rcParams['font.size'] = 8

# Load the data
with open('results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/analysis_summary.json', 'r') as f:
    data = json.load(f)

# Extract layer numbers
layers = [layer_data['layer'] for layer_data in data['layers']]
languages = data['languages']

# Extract R² linearity for each component
components = ['year', 'month', 'day', 'weekday']
r2_linearity = {comp: [] for comp in components}

for layer_data in data['layers']:
    for comp in components:
        r2_linearity[comp].append(layer_data['component_linearity_r2'][comp])

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(7, 4))
axes = axes.flatten()

# Individual component plots
component_names = {'year': 'Year', 'month': 'Month', 'day': 'Day', 'weekday': 'Weekday'}
colors_comp = {'year': 'C0', 'month': 'C1', 'day': 'C2', 'weekday': 'C3'}

for idx, comp in enumerate(components):
    ax = axes[idx]
    
    ax.plot(layers, r2_linearity[comp], marker='o', color=colors_comp[comp],
            linewidth=1, markersize=2.5, label=f'{component_names[comp]} R²')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('R² Linearity')
    ax.set_title(f'({chr(97+idx)}) {component_names[comp]} Linearity', loc='left')
    ax.set_xlim(0, max(layers))
    ax.set_ylim(-0.3, 1.05)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/r2_linearity_analysis.png', 
            dpi=600, bbox_inches='tight')
plt.savefig('results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/r2_linearity_analysis.pdf', 
            bbox_inches='tight')
print("✓ Saved R² linearity analysis to:")
print("  - results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/r2_linearity_analysis.png")
print("  - results/temporal_geometry/Qwen_Qwen2.5_3B_Instruct/r2_linearity_analysis.pdf")

# Print summary statistics
print("\n=== R² Linearity Summary ===")
for comp in components:
    avg_r2 = np.mean(r2_linearity[comp])
    min_r2 = np.min(r2_linearity[comp])
    max_r2 = np.max(r2_linearity[comp])
    print(f"{component_names[comp]:8s}: avg={avg_r2:.3f}, min={min_r2:.3f}, max={max_r2:.3f}")