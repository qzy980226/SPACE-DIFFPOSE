@echo off
REM 遮挡鲁棒性评估脚本 (Windows)
REM 用法: run_occlusion_eval.bat

echo ================================================
echo 遮挡鲁棒性评估
echo ================================================

REM 设置配置文件路径
set CONFIG=configs/speedplus_v2_diffpose.yml

REM 最佳模型路径（请根据实际情况修改）
set MODEL_PATH=exp/speedplus_v2_diffpose_uvxyz_gt/best_model.pth

REM 实验输出目录
set EXP_DIR=exp
set DOC_NAME=occlusion_eval

REM 遮挡比例列表
set OCCLUSION_RATIOS=0.0 0.2 0.4 0.6 0.8

echo 配置文件: %CONFIG%
echo 模型路径: %MODEL_PATH%
echo 遮挡比例: %OCCLUSION_RATIOS%
echo 输出目录: %EXP_DIR%/%DOC_NAME%
echo ================================================
echo.

REM 运行评估
python evaluate_occlusion_robustness.py ^
    --config %CONFIG% ^
    --model_path %MODEL_PATH% ^
    --occlusion_ratios %OCCLUSION_RATIOS% ^
    --doc %DOC_NAME% ^
    --exp %EXP_DIR% ^
    --skip_type uniform ^
    --eta 0.0

echo.
echo ================================================
echo 评估完成！结果保存在: %EXP_DIR%/%DOC_NAME%/
echo ================================================
pause
