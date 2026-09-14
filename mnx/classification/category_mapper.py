"""
统一类别映射管理模块
确保整个系统中所有涉及预测类别的地方都使用一致的类别名称和映射关系
严格遵照训练神经网络模型时上传的类别标签表格文件（category_map.json）
"""
import json
from pathlib import Path
from typing import Dict, Optional

# 项目根目录
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent


class CategoryMapper:
    """统一的类别映射管理器"""
    
    _instance = None
    _category_map: Dict[int, str] = {}
    _reverse_map: Dict[str, int] = {}
    _initialized = False
    _last_modified_time = None  # ✅ 记录最后修改时间
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._load_category_map()
            self._initialized = True
        else:
            # ✅ 检查文件是否更新，如果更新则重新加载
            self._check_and_reload()
    
    def _load_category_map(self):
        """从 category_map.json 加载类别映射"""
        map_path = CURRENT_DIR / "category_map.json"
        
        # ✅ 记录文件修改时间
        if map_path.exists():
            self._last_modified_time = map_path.stat().st_mtime
        
        # 默认映射（与训练时使用的类别标签一致）
        default_map = {
            0: "非公益诉讼",
            1: "生态环境和资源保护",
            2: "食品药品安全",
            3: "国有财产保护",
            4: "国有土地使用权出让",
            5: "英雄烈士保护",
            6: "未成年人保护",
            7: "安全生产",
            8: "个人信息保护",
            9: "军人地位和权益保障",
            10: "反垄断和反不正当竞争",
            11: "反电信网络诈骗",
            12: "农产品质量安全",
            13: "妇女权益保障",
            14: "无障碍环境建设",
            15: "文物和文化遗产保护",
            16: "国防和军事利益保护"
        }
        
        if map_path.exists():
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    loaded_map = json.load(f)
                # JSON 中的键是字符串，需要转换为整数
                self._category_map = {int(k): str(v) for k, v in loaded_map.items()}
                print(f"[CategoryMapper] 成功加载 category_map.json，共 {len(self._category_map)} 个类别")
            except Exception as e:
                print(f"[CategoryMapper] 加载 category_map.json 失败: {e}，使用默认映射")
                self._category_map = default_map.copy()
        else:
            print(f"[CategoryMapper] category_map.json 不存在，使用默认映射")
            self._category_map = default_map.copy()
        
        # 构建反向映射
        self._reverse_map = {v: k for k, v in self._category_map.items()}
    
    def get_category_name(self, category_id: int) -> str:
        """根据类别ID获取类别名称"""
        return self._category_map.get(category_id, f"未知类别({category_id})")
    
    def get_category_id(self, category_name: str) -> Optional[int]:
        """根据类别名称获取类别ID"""
        return self._reverse_map.get(category_name, None)
    
    def get_all_categories(self) -> Dict[int, str]:
        """获取所有类别映射"""
        return self._category_map.copy()
    
    def get_all_category_names(self) -> list:
        """获取所有类别名称列表"""
        return list(self._category_map.values())
    
    def get_all_category_ids(self) -> list:
        """获取所有类别ID列表"""
        return list(self._category_map.keys())
    
    def validate_category_name(self, category_name: str) -> bool:
        """验证类别名称是否有效"""
        return category_name in self._reverse_map
    
    def validate_category_id(self, category_id: int) -> bool:
        """验证类别ID是否有效"""
        return category_id in self._category_map
    
    def get_api_category_map(self) -> Dict[int, str]:
        """
        获取用于 API 托底推理的类别映射
        返回完整的类别编号到名称的映射，确保 API 返回的类别能正确转换
        """
        return self._category_map.copy()
    
    def reload(self):
        """重新加载类别映射（用于训练后更新）"""
        self._initialized = False
        self._load_category_map()
        self._initialized = True
    
    def _check_and_reload(self):
        """检查文件是否更新，如果更新则重新加载"""
        map_path = CURRENT_DIR / "category_map.json"
        
        if not map_path.exists():
            return
        
        current_mtime = map_path.stat().st_mtime
        
        # 如果文件修改时间比记录的时间新，则重新加载
        if self._last_modified_time is None or current_mtime > self._last_modified_time:
            print(f"[CategoryMapper] 🔄 检测到 category_map.json 已更新，重新加载...")
            self._load_category_map()
            print(f"[CategoryMapper] ✅ 重新加载完成，共 {len(self._category_map)} 个类别")


# 创建全局单例实例
_mapper = CategoryMapper()


def get_category_name(category_id: int) -> str:
    """便捷函数：根据类别ID获取类别名称"""
    return _mapper.get_category_name(category_id)


def get_category_id(category_name: str) -> Optional[int]:
    """便捷函数：根据类别名称获取类别ID"""
    return _mapper.get_category_id(category_name)


def get_all_categories() -> Dict[int, str]:
    """便捷函数：获取所有类别映射"""
    return _mapper.get_all_categories()


def get_all_category_names() -> list:
    """便捷函数：获取所有类别名称列表"""
    return _mapper.get_all_category_names()


def get_api_category_map() -> Dict[int, str]:
    """便捷函数：获取 API 使用的类别映射"""
    return _mapper.get_api_category_map()


def validate_and_fix_category_name(category_name: str, default_id: int = 0) -> tuple:
    """
    验证并修正类别名称
    如果类别名称无效，返回默认类别ID和名称
    
    Returns:
        (category_id, category_name) 元组
    """
    category_id = _mapper.get_category_id(category_name)
    if category_id is None:
        # 类别名称无效，使用默认值
        print(f"[CategoryMapper] ⚠️ 无效的类别名称: '{category_name}'，使用默认类别 {default_id}")
        return default_id, _mapper.get_category_name(default_id)
    return category_id, category_name


if __name__ == "__main__":
    # 测试代码
    print("=" * 60)
    print("类别映射管理器测试")
    print("=" * 60)
    
    mapper = CategoryMapper()
    
    print("\n所有类别:")
    for cat_id, cat_name in mapper.get_all_categories().items():
        print(f"  {cat_id}: {cat_name}")
    
    print("\n测试正向映射 (ID -> Name):")
    test_ids = [0, 1, 5, 10, 15, 16, 99]
    for tid in test_ids:
        name = mapper.get_category_name(tid)
        print(f"  {tid} -> {name}")
    
    print("\n测试反向映射 (Name -> ID):")
    test_names = ["生态环境和资源保护", "食品药品安全", "不存在的类别"]
    for tname in test_names:
        cid = mapper.get_category_id(tname)
        print(f"  '{tname}' -> {cid}")
    
    print("\nAPI 类别映射:")
    api_map = mapper.get_api_category_map()
    print(f"  共 {len(api_map)} 个类别")
    for cid, cname in list(api_map.items())[:5]:
        print(f"  {cid}: {cname}")
