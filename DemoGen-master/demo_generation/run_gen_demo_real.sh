#!/bin/bash
# 真实环境数据增强脚本
#
# 用法：
#   bash run_gen_demo_real.sh

# 脚本已在 demo_generation 目录中，无需 cd

# 配置参数
task=real_demo_1
gen_range=test
gen_mode=grid
n_gen_per_source=16
render_video=true

data_root=../data

echo "======================================"
echo "真实环境DemoGen数据增强"
echo "======================================"
echo "源数据: ${task}"
echo "生成范围: ${gen_range}"
echo "生成模式: ${gen_mode}"
echo "每个源生成数量: ${n_gen_per_source}"
echo "渲染视频: ${render_video}"
echo "======================================"

python gen_demo_real.py --config-name=${task} \
                                data_root=${data_root} \
                                generation.range_name=${gen_range} \
                                generation.mode=${gen_mode} \
                                generation.n_gen_per_source=${n_gen_per_source} \
                                generation.render_video=${render_video}
