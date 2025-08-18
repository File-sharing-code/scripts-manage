#!/bin/bash

# 检查是否传入了目录和文件名
if [ -z "$1" ]; then
  echo "用法: $0 <文件路径 /{businessType}/{device}/{account}/{time}>"
  exit 1
fi

# 指定目录和文件名
BASEPATH="/web/uploadfile"
DIRECTORY="$BASEPATH$1"
FILENAME=$(uuidgen | sed 's/-//g')

# 检查目录是否存在，如果不存在则创建
if [ ! -d "$DIRECTORY" ]; then
  mkdir -p "$DIRECTORY"
fi

# 创建文件
touch "$DIRECTORY/$FILENAME"

echo "文件 $FILENAME 已在目录 $DIRECTORY 下创建。"

# ./create_log_file.sh /cx/android/89000999/20250806