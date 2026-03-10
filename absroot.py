import json
import os
import re

# 配置参数（根据你的需求修改）
# 需要替换的绝对路径前缀
ABS_PATH_PREFIX = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/avrgo2_link/"
# 目标JSON文件夹路径
JSON_FOLDER = "/inspire/qb-ilm/project/wuliqifa/chenxinyan-240108120066/songbur-data/camsim_lyh/avrgo2_json"
# 替换后的相对路径前缀（空字符串表示直接保留train/...，也可改为./train/）
REL_PATH_PREFIX = ""

def replace_abs_to_rel(file_path):
    """修改单个JSON文件中的绝对路径为相对路径"""
    try:
        # 读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 定义路径替换函数（复用逻辑）
        def replace_path(path):
            if isinstance(path, str) and path.startswith(ABS_PATH_PREFIX):
                # 去掉绝对路径前缀，保留后续部分
                rel_path = path.replace(ABS_PATH_PREFIX, REL_PATH_PREFIX)
                return rel_path
            return path
        
        # 修改path字段
        if "path" in data:
            data["path"] = replace_path(data["path"])
        
        # 修改files数组中的所有路径
        if "files" in data and isinstance(data["files"], list):
            data["files"] = [replace_path(f) for f in data["files"]]
        
        # 写回修改后的内容（保留JSON格式）
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 成功修改：{file_path}")
    
    except Exception as e:
        print(f"❌ 处理失败 {file_path}：{str(e)}")

def batch_process_json():
    """批量处理文件夹下所有JSON文件"""
    # 检查目标文件夹是否存在
    if not os.path.exists(JSON_FOLDER):
        print(f"❌ 文件夹不存在：{JSON_FOLDER}")
        return
    
    # 遍历所有JSON文件
    json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith('.json')]
    if not json_files:
        print(f"❌ 文件夹中未找到JSON文件：{JSON_FOLDER}")
        return
    
    print(f"📌 开始处理 {len(json_files)} 个JSON文件...")
    for json_file in json_files:
        file_path = os.path.join(JSON_FOLDER, json_file)
        replace_abs_to_rel(file_path)
    
    print("\n🎉 所有JSON文件处理完成！")

if __name__ == "__main__":
    batch_process_json()