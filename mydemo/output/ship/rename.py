# -*- coding: utf-8 -*- 
# 导入python内置的os模块和sys模块
import os
import sys
 
# 程序入口
if __name__ == "__main__":
    # 获取本目录下所有的文件名
    old_names = os.listdir()
    # 遍历目录下的文件名
    for old_name in old_names:
        # 跳过本脚本文件
        if old_name != sys.argv[0]:
            # 用新的文件名替换旧的文件名
            new_name = old_name.replace('微信图片_', '')
            os.rename(old_name, new_name)