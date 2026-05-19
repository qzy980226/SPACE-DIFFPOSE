"""
诊断检查点文件问题
"""
import torch
import sys

checkpoint_path = 'exp/speedplus_v2_diffpose_uvxyz_gt/best_model.pth'

print("="*80)
print("检查点诊断工具")
print("="*80)

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"\n✓ 成功加载检查点: {checkpoint_path}")
    print(f"  - 类型: {type(checkpoint)}")
    print(f"  - 长度: {len(checkpoint)}")

    if isinstance(checkpoint, list):
        print(f"\n检查点内容:")
        print(f"  [0] 模型参数 (state_dict)")
        print(f"  [1] 优化器参数")
        print(f"  [2] Epoch: {checkpoint[2] if len(checkpoint) > 2 else 'N/A'}")
        print(f"  [3] Step: {checkpoint[3] if len(checkpoint) > 3 else 'N/A'}")
        print(f"  [4] EMA参数: {'存在' if len(checkpoint) > 4 else '不存在'}")

    model_dict = checkpoint[0]
    print(f"\n模型state_dict信息:")
    print(f"  - 总键数: {len(model_dict)}")

    # 检查是否有 module. 前缀
    has_module_prefix = any(k.startswith('module.') for k in model_dict.keys())
    print(f"  - 使用 DataParallel: {'是' if has_module_prefix else '否'}")

    # 检查可见性嵌入
    print(f"\n可见性嵌入相关键:")
    vis_keys = [k for k in model_dict.keys() if 'visibility' in k.lower()]
    if vis_keys:
        print(f"  ✓ 找到 {len(vis_keys)} 个可见性相关键:")
        for k in vis_keys:
            print(f"    - {k}: {model_dict[k].shape}")
    else:
        print(f"  ✗ 未找到任何可见性嵌入参数!")
        print(f"\n  这说明模型是用旧版本代码训练的（没有可见性嵌入功能）")

    # 显示前20个键
    print(f"\n前20个模型键:")
    for i, key in enumerate(list(model_dict.keys())[:20], 1):
        shape = model_dict[key].shape
        print(f"  {i:2d}. {key}: {shape}")

    if len(model_dict) > 20:
        print(f"  ... (还有 {len(model_dict) - 20} 个键)")

    # 检查融合层
    fusion_keys = [k for k in model_dict.keys() if 'fusion' in k.lower()]
    print(f"\n融合层相关键:")
    if fusion_keys:
        print(f"  ✓ 找到 {len(fusion_keys)} 个融合层键:")
        for k in fusion_keys:
            print(f"    - {k}: {model_dict[k].shape}")
    else:
        print(f"  ✗ 未找到融合层参数")

    print(f"\n" + "="*80)
    print("诊断结论:")
    print("="*80)

    if not vis_keys and not fusion_keys:
        print("\n❌ 问题确认: 检查点文件缺少可见性嵌入和融合层参数")
        print("\n原因: 此模型是用不包含可见性嵌入功能的旧版本代码训练的")
        print("\n解决方案:")
        print("  1. 重新训练模型（推荐）")
        print("  2. 修改评估代码，使用 strict=False 加载模型")
        print("     （缺失的参数将随机初始化，可能影响性能）")
        print("  3. 从检查点恢复到旧版本代码进行评估")
    else:
        print("\n✓ 检查点包含可见性嵌入参数，问题可能在其他地方")

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
