import re
import jieba
import jieba.posseg as pseg
from typing import Union, List, Dict, Any
from functools import lru_cache
import threading

class PrivacyMasker:
    """个人隐私信息脱敏器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式，避免重复初始化jieba词典"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # 定义正则表达式模式（按优先级排序）
        self.patterns = [
            # 身份证号（15位或18位，18位最后一位可为X）
            ('id_card', r'[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]'),
            # 手机号（11位，1开头）
            ('phone', r'1[3-9]\d{9}'),
            # 邮箱地址（优先匹配，避免@符号干扰）
            ('email', r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            # 微信号（需要前后有明确标识，避免误匹配）
            ('wechat', r'(?:微信|微信号|WX|wx)[：:\s]*([a-zA-Z][a-zA-Z0-9_-]{5,19})'),
            # 座机号（区号-号码 或 号码格式）
            ('landline', r'(?:0\d{2,3}-?)[2-9]\d{6,7}'),
            # 银行卡号（16-19位数字，需要边界检查）
            ('bank_card', r'(?<!\d)\d{16,19}(?!\d)'),
        ]
        
        # 地址关键词（用于jieba分词辅助识别）
        self.address_keywords = [
            '省', '市', '区', '县', '镇', '乡', '村', '街', '路', '号', '栋', '单元', '室',
            '小区', '公寓', '大厦', '广场', '中心', '园区', '开发区', '工业园', '写字楼',
            '公司', '单位', '地址', '住址', '家', '住在', '位于'
        ]
        
        # 姓名常用姓氏（辅助判断）
        self.surnames = [
            '王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴', '徐', '孙', '马', '朱', '胡',
            '郭', '林', '何', '高', '罗', '郑', '梁', '谢', '宋', '唐', '许', '韩', '冯', '邓', '曹',
            '彭', '曾', '肖', '田', '董', '袁', '潘', '于', '蒋', '蔡', '余', '杜', '叶', '程', '苏',
            '魏', '吕', '丁', '任', '沈', '姚', '卢', '姜', '崔', '钟', '谭', '陆', '汪', '范', '金',
            '石', '廖', '贾', '夏', '韦', '付', '方', '白', '邹', '孟', '熊', '秦', '邱', '江', '尹',
            '薛', '闫', '段', '雷', '侯', '龙', '史', '陶', '黎', '贺', '顾', '毛', '郝', '龚', '邵',
            '万', '钱', '严', '覃', '武', '戴', '莫', '孔', '向', '汤', '欧阳', '太史', '端木', '上官',
            '司马', '东方', '独孤', '南宫', '夏侯', '诸葛', '尉迟', '皇甫', '公孙', '慕容', '司徒', '司空'
        ]
        
        # 添加地址词典到jieba
        for word in self.address_keywords:
            jieba.add_word(word, tag='ns')
        
        self._initialized = True
    
    def mask_by_regex(self, text: str) -> tuple:
        """使用正则表达式进行基础脱敏"""
        masked_text = text
        mask_info = []
        
        for info_type, pattern in self.patterns:
            matches = list(re.finditer(pattern, text))
            for match in reversed(matches):
                start, end = match.span()
                
                # 对于分组匹配（如微信号），使用分组内容
                if match.lastindex and match.lastindex >= 1:
                    original = match.group(1)
                    group_start, group_end = match.start(1), match.end(1)
                else:
                    original = match.group()
                    group_start, group_end = start, end
                
                # 根据类型生成脱敏文本
                masked = self._generate_mask(info_type, original)
                
                masked_text = masked_text[:group_start] + masked + masked_text[group_end:]
                mask_info.append({
                    'type': info_type,
                    'original': original,
                    'masked': masked,
                    'position': (group_start, group_end)
                })
        
        return masked_text, mask_info
    
    def _generate_mask(self, info_type: str, original: str) -> str:
        """根据信息类型生成脱敏文本"""
        masks = {
            'id_card': lambda x: x[:6] + '*' * 8 + x[-4:],
            'phone': lambda x: x[:3] + '*' * 4 + x[-4:],
            'wechat': lambda x: x[:2] + '*' * (len(x) - 4) + x[-2:],
            'landline': lambda x: x[:3] + '*' * (len(x) - 6) + x[-3:],
            'email': lambda x: x.split('@')[0][:2] + '***@' + x.split('@')[1],
            'bank_card': lambda x: x[:4] + '*' * (len(x) - 8) + x[-4:],
            'name': lambda x: x[0] + '*' * (len(x) - 1),
            'address': self._mask_address
        }
        return masks.get(info_type, lambda x: '*' * len(x))(original)
    
    def _mask_address(self, address: str) -> str:
        """
        地址脱敏策略：更激进的隐藏
        - 保留省市级别（前6-8个字符）
        - 其余全部用***替代
        """
        if len(address) <= 8:
            return address[:2] + '***'
        elif len(address) <= 15:
            return address[:4] + '***'
        else:
            # 提取省市信息，其余隐藏
            return address[:6] + '***'
    
    @lru_cache(maxsize=1000)
    def extract_names_by_jieba(self, text: str) -> List[str]:
        """使用jieba分词识别人名（带缓存）"""
        words = pseg.cut(text)
        names = []
        
        for word, flag in words:
            if flag == 'nr' and 2 <= len(word) <= 4:
                if any(word.startswith(surname) for surname in self.surnames):
                    names.append(word)
        
        return names
    
    @lru_cache(maxsize=1000)
    def extract_addresses_by_jieba(self, text: str) -> List[str]:
        """使用jieba分词识别地址（带缓存）"""
        words = list(pseg.cut(text))
        addresses = []
        current_address = []
        
        for i, (word, flag) in enumerate(words):
            if flag == 'ns' or word in self.address_keywords:
                current_address.append(word)
            else:
                if current_address and re.match(r'^\d+[号栋单元室层]?$', word):
                    current_address.append(word)
                else:
                    if len(current_address) >= 3:
                        address_str = ''.join(current_address)
                        if len(address_str) >= 8:
                            addresses.append(address_str)
                    current_address = []
        
        if len(current_address) >= 3:
            address_str = ''.join(current_address)
            if len(address_str) >= 8:
                addresses.append(address_str)
        
        return addresses
    
    def mask_text(self, text: str) -> Dict[str, Any]:
        """完整的脱敏处理流程"""
        if not text or not isinstance(text, str):
            return {
                'original': text,
                'masked': text,
                'regex_matches': [],
                'names_found': [],
                'addresses_found': []
            }
        
        try:
            # 第一步：正则脱敏
            masked_text, regex_info = self.mask_by_regex(text)
            
            # 第二步：jieba识别人名并脱敏
            names = self.extract_names_by_jieba(text)
            for name in names:
                masked_name = self._generate_mask('name', name)
                masked_text = masked_text.replace(name, masked_name)
            
            # 第三步：jieba识别地址并脱敏
            addresses = self.extract_addresses_by_jieba(text)
            for address in addresses:
                if address in masked_text and '*' not in address:
                    masked_address = self._generate_mask('address', address)
                    masked_text = masked_text.replace(address, masked_address)
            
            return {
                'original': text,
                'masked': masked_text,
                'regex_matches': regex_info,
                'names_found': names,
                'addresses_found': addresses
            }
        except Exception as e:
            print(f"脱敏处理出错: {e}")
            return {
                'original': text,
                'masked': text,
                'regex_matches': [],
                'names_found': [],
                'addresses_found': [],
                'error': str(e)
            }


# ==================== 简化的函数接口 ====================

def mask_text_simple(text: str) -> str:
    """
    简单接口：输入字符串，返回脱敏后的字符串
    
    Args:
        text: 待脱敏的文本
        
    Returns:
        脱敏后的文本
    """
    masker = PrivacyMasker()
    result = masker.mask_text(text)
    return result['masked']


def mask_batch_texts(texts: List[str]) -> List[str]:
    """
    批量处理接口：处理多个文本
    
    Args:
        texts: 待脱敏的文本列表
        
    Returns:
        脱敏后的文本列表
    """
    masker = PrivacyMasker()
    results = []
    for text in texts:
        result = masker.mask_text(text)
        results.append(result['masked'])
    return results


def mask_text_detailed(text: str) -> Dict[str, Any]:
    """
    详细接口：返回完整的脱敏信息
    
    Args:
        text: 待脱敏的文本
        
    Returns:
        包含原文、脱敏文、匹配信息的字典
    """
    masker = PrivacyMasker()
    return masker.mask_text(text)


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 示例1：使用简单接口
    print("=" * 80)
    print("简单接口演示")
    print("=" * 80)
    
    test_text = "我叫张伟，身份证号是11010119900307893X，手机号13812345678，住在北京市朝阳区建国路88号SOHO现代城3号楼2单元1502室"
    
    masked = mask_text_simple(test_text)
    print(f"原文：{test_text}")
    print(f"脱敏：{masked}\n")