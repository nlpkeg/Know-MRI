#!/bin/bash

# 配置参数
REPO_URL="https://github.com/zjunlp/EasyEdit.git"  # 替换为实际地址
COMMIT_HASH="a64270758473946da48565c66da3f8af2dd36d04"  # 替换为你基于的 commit 哈希
PATCH_FILE="../evaluation/Easyedit.patch"  # 你的 patch 文件路径
TARGET_DIR="./EasyEdit"  # 克隆目标目录

# 1. 克隆仓库
git clone "$REPO_URL" "$TARGET_DIR"

# 2. 进入目录
cd "$TARGET_DIR"

# 3. 切换到特定 commit
git checkout "$COMMIT_HASH"

# 4. 应用 patch
git apply "$PATCH_FILE"

echo "完成！"