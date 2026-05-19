"""
遮挡评估结果可视化工具

使用方法：
    python visualize_occlusion_results.py --result_path exp/occlusion_eval/occlusion_results.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_occlusion_performance(csv_path, output_dir=None):
    """
    绘制遮挡性能曲线

    Args:
        csv_path: CSV结果文件路径
        output_dir: 输出目录（默认为CSV文件所在目录）
    """
    # 读取结果
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 设置绘图风格
    plt.style.use('seaborn-v0_8-darkgrid')

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Occlusion Robustness Evaluation Results', fontsize=16, fontweight='bold')

    # ========== 子图1: MPJPE vs 遮挡率 ==========
    ax1 = axes[0, 0]
    ax1.plot(df['Occlusion_Ratio'] * 100, df['MPJPE_mm'],
             marker='o', linewidth=2.5, markersize=8, color='#e74c3c', label='MPJPE')
    ax1.fill_between(df['Occlusion_Ratio'] * 100, 0, df['MPJPE_mm'],
                      alpha=0.2, color='#e74c3c')
    ax1.set_xlabel('Occlusion Ratio (%)', fontsize=12)
    ax1.set_ylabel('MPJPE (mm)', fontsize=12)
    ax1.set_title('MPJPE vs Occlusion Ratio', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # 标注数值
    for i, (x, y) in enumerate(zip(df['Occlusion_Ratio'] * 100, df['MPJPE_mm'])):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # ========== 子图2: PA-MPJPE vs 遮挡率 ==========
    ax2 = axes[0, 1]
    ax2.plot(df['Occlusion_Ratio'] * 100, df['PA_MPJPE_mm'],
             marker='s', linewidth=2.5, markersize=8, color='#3498db', label='PA-MPJPE')
    ax2.fill_between(df['Occlusion_Ratio'] * 100, 0, df['PA_MPJPE_mm'],
                      alpha=0.2, color='#3498db')
    ax2.set_xlabel('Occlusion Ratio (%)', fontsize=12)
    ax2.set_ylabel('PA-MPJPE (mm)', fontsize=12)
    ax2.set_title('PA-MPJPE vs Occlusion Ratio', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # 标注数值
    for i, (x, y) in enumerate(zip(df['Occlusion_Ratio'] * 100, df['PA_MPJPE_mm'])):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # ========== 子图3: 性能退化率 ==========
    ax3 = axes[1, 0]
    baseline_mpjpe = df['MPJPE_mm'].iloc[0]
    degradation = ((df['MPJPE_mm'] - baseline_mpjpe) / baseline_mpjpe * 100).values

    colors = ['#2ecc71' if d < 50 else '#f39c12' if d < 100 else '#e74c3c'
              for d in degradation]

    bars = ax3.bar(df['Occlusion_Ratio'] * 100, degradation, color=colors, alpha=0.7, width=8)
    ax3.set_xlabel('Occlusion Ratio (%)', fontsize=12)
    ax3.set_ylabel('Performance Degradation (%)', fontsize=12)
    ax3.set_title('Performance Degradation Relative to Baseline', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # 标注数值
    for bar, deg in zip(bars, degradation):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height,
                f'{deg:.1f}%', ha='center', va='bottom', fontsize=9)

    # ========== 子图4: MPJPE和PA-MPJPE对比 ==========
    ax4 = axes[1, 1]
    x = np.arange(len(df))
    width = 0.35

    bars1 = ax4.bar(x - width / 2, df['MPJPE_mm'], width, label='MPJPE',
                    color='#e74c3c', alpha=0.8)
    bars2 = ax4.bar(x + width / 2, df['PA_MPJPE_mm'], width, label='PA-MPJPE',
                    color='#3498db', alpha=0.8)

    ax4.set_xlabel('Occlusion Ratio (%)', fontsize=12)
    ax4.set_ylabel('Error (mm)', fontsize=12)
    ax4.set_title('MPJPE vs PA-MPJPE Comparison', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{int(r * 100)}' for r in df['Occlusion_Ratio']])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存图形
    output_path = os.path.join(output_dir, 'occlusion_performance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图形已保存到: {output_path}")

    # 也保存PDF版本
    pdf_path = os.path.join(output_dir, 'occlusion_performance.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF已保存到: {pdf_path}")

    plt.show()


def generate_summary_table(csv_path, output_dir=None):
    """
    生成汇总表格图像

    Args:
        csv_path: CSV结果文件路径
        output_dir: 输出目录
    """
    df = pd.read_csv(csv_path)

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    # 计算性能退化
    baseline_mpjpe = df['MPJPE_mm'].iloc[0]
    baseline_pa = df['PA_MPJPE_mm'].iloc[0]

    df['MPJPE_Degradation'] = ((df['MPJPE_mm'] - baseline_mpjpe) / baseline_mpjpe * 100).round(1)
    df['PA_MPJPE_Degradation'] = ((df['PA_MPJPE_mm'] - baseline_pa) / baseline_pa * 100).round(1)

    # 格式化显示
    df['Occlusion (%)'] = (df['Occlusion_Ratio'] * 100).astype(int)
    df['MPJPE'] = df['MPJPE_mm'].round(2).astype(str) + ' mm'
    df['PA-MPJPE'] = df['PA_MPJPE_mm'].round(2).astype(str) + ' mm'
    df['MPJPE Deg.'] = df['MPJPE_Degradation'].astype(str) + '%'
    df['PA-MPJPE Deg.'] = df['PA_MPJPE_Degradation'].astype(str) + '%'

    # 选择要显示的列
    display_df = df[['Occlusion (%)', 'MPJPE', 'PA-MPJPE', 'MPJPE Deg.', 'PA-MPJPE Deg.']]

    # 创建表格图像
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=display_df.values,
                     colLabels=display_df.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15, 0.2, 0.2, 0.2, 0.25])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 设置行颜色（交替）
    for i in range(1, len(display_df) + 1):
        for j in range(len(display_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title('Occlusion Robustness Summary', fontsize=14, fontweight='bold', pad=20)

    # 保存
    output_path = os.path.join(output_dir, 'occlusion_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"汇总表格已保存到: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Visualize Occlusion Evaluation Results')
    parser.add_argument('--result_path', type=str, required=True,
                       help='Path to occlusion_results.csv')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as result file)')

    args = parser.parse_args()

    if not os.path.exists(args.result_path):
        print(f"错误: 文件不存在 {args.result_path}")
        return

    print("正在生成可视化结果...")

    # 绘制性能曲线
    plot_occlusion_performance(args.result_path, args.output_dir)

    # 生成汇总表格
    generate_summary_table(args.result_path, args.output_dir)

    print("\n可视化完成！")


if __name__ == "__main__":
    main()
