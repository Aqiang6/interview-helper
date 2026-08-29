"""简历解析服务 - 支持文本输入和 PDF 文件解析"""

from __future__ import annotations

import io
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ResumeData:
    """解析后的简历数据"""
    raw_text: str = ""
    name: str = ""
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    experience_years: str = ""
    education: str = ""
    summary: str = ""


# AI Agent 相关核心关键词（优先匹配）
_AI_AGENT_KEYWORDS: list[str] = [
    "语义缓存", "Semantic Cache", "向量相似度", "Embedding", "sentence-transformers",
    "Token管理", "Token压缩", "上下文窗口", "Context Window", "tiktoken",
    "RAG", "检索增强生成", "向量数据库", "Vector DB", "FAISS", "Milvus", "Pinecone",
    "LLM", "大语言模型", "GPT", "Claude", "通义千问", "DeepSeek", "智谱GLM",
    "Prompt Engineering", "提示工程", "系统提示词", "Few-shot", "Chain of Thought",
    "对话摘要", "Summarization", "上下文压缩", "Cache Hit", "缓存命中率",
    "向量运算", "NumPy", "余弦相似度", "Cosine Similarity",
    "可观测性", "OpenTelemetry", "TTFT", "Token用量", "降级", "Fallback",
    "Provider Registry", "模型热切换", "多模型适配", "API加密", "Fernet",
    "异步调用", "asyncio", "httpx", "SSE", "WebSocket",
]

# 架构设计模式关键词（优先匹配）
_ARCHITECTURE_KEYWORDS: list[str] = [
    "分布式锁", "分库分表", "水平分片", "幂等性", "接口限流", "QPS限流",
    "缓存策略", "消息去重", "异步解耦", "自动续期", "超卖问题",
    "分布式事务", "事务一致性", "分布式ID", "分布式缓存", "分布式Session",
    "服务发现", "熔断降级", "服务网格", "API网关", "限流熔断",
    "读写分离", "主从复制", "负载均衡", "灰度发布", "蓝绿部署",
    "CQRS", "Event Sourcing", "领域驱动设计", "DDD", "微服务",
]

# 常见技术栈关键词
_SKILL_KEYWORDS: list[str] = [
    "Python", "Java", "JavaScript", "TypeScript", "Go", "Rust", "C++", "C#",
    "React", "Vue", "Vue 3", "Angular", "Next.js", "Nuxt.js",
    "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "Spring Security",
    "Docker", "Kubernetes", "K8s", "AWS", "Azure", "GCP",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Kafka", "RabbitMQ", "RocketMQ", "gRPC", "GraphQL", "REST",
    "PyTorch", "TensorFlow", "Transformer",
    "Git", "CI/CD", "Jenkins", "GitHub Actions",
    "Linux", "Nginx", "Tomcat",
    "机器学习", "深度学习", "NLP", "计算机视觉",
    "HTML", "CSS", "Node.js", "Express",
    "MyBatis", "MyBatis-Plus", "Hibernate", "JPA",
    "Oracle", "SQLite", "ClickHouse",
    "Terraform", "Ansible", "Prometheus", "Grafana",
    "Flutter", "Swift", "Kotlin", "Android", "iOS",
    # 细粒度组件
    "Redisson", "Sentinel", "ShardingSphere", "Logback", "Logstash",
    "JWT", "BCrypt", "LFU", "LRU", "Vite", "Element Plus", "ElementPlus",
    "Vue Router", "Axios", "Webpack", "Rollup",
    # 中间件
    "Nacos", "Dubbo", "Seata", "Saga", "Ribbon", "Feign",
    # 数据库相关
    "MyISAM", "InnoDB", "TiDB", "OceanBase",
]

# 项目名称匹配模式
_PROJECT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:项目(?:名称|经历|经验)|负责|主导|参与)[：:]\s*(.+?)(?:\n|$)"),
    re.compile(r"^[\u4e00-\u9fa5]{2,12}(?:项目|系统|平台|工程|服务)$", re.MULTILINE),
    re.compile(r"[\u4e00-\u9fa5]{2,12}(?:项目|系统|平台|工程|服务)(?=\s|$)"),
]


def parse_text(text: str) -> ResumeData:
    """解析纯文本简历"""
    data = ResumeData(raw_text=text.strip())
    if not data.raw_text:
        return data

    # 记录文本质量
    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    total_chars = len(text.strip())
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    logger.info(f"文本解析 - 总字符: {total_chars}, 中文字符: {chinese_chars}, 中文占比: {chinese_ratio:.2f}")

    # 提取姓名：多种模式匹配
    name_patterns = [
        re.compile(r"姓\s*名\s*[：:]\s*([\u4e00-\u9fa5]{2,4})"),
        re.compile(r"^([\u4e00-\u9fa5]{2,4})$", re.MULTILINE),
        re.compile(r"([\u4e00-\u9fa5]{2,4})\s*[同志先生女士小姐]"),
        re.compile(r"([\u4e00-\u9fa5]{2,4})\s*\|"),
        re.compile(r"([\u4e00-\u9fa5]{2,4})\s*/"),
        re.compile(r"([\u4e00-\u9fa5]{2,4})\s+-"),
    ]
    for pattern in name_patterns:
        match = pattern.search(text)
        if match and match.lastindex:
            name = match.group(1).strip()
            if len(name) >= 2 and len(name) <= 4:
                data.name = name
                break
    
    if not data.name:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if lines and len(lines[0]) <= 20 and not any(keyword in lines[0] for keyword in ["简历", "个人", "应聘", "求职"]):
            data.name = lines[0]

    # 提取技能：优先提取 AI Agent 和架构关键词，再提取通用技术栈
    lower_text = text.lower()
    found_skills: set[str] = set()
    
    # 优先匹配 AI Agent 核心关键词
    for skill in _AI_AGENT_KEYWORDS:
        if skill.lower() in lower_text:
            found_skills.add(skill)
    
    # 优先匹配架构设计模式关键词
    for skill in _ARCHITECTURE_KEYWORDS:
        if skill.lower() in lower_text:
            found_skills.add(skill)
    
    # 再匹配通用技术栈
    for skill in _SKILL_KEYWORDS:
        if skill.lower() in lower_text:
            found_skills.add(skill)
    
    data.skills = sorted(found_skills)

    # 提取项目：优先提取明确的项目名称，过滤无意义片段
    projects: list[str] = []
    for pattern in _PROJECT_PATTERNS:
        for match in pattern.finditer(text):
            proj = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
            proj = proj.strip()
            # 过滤条件：长度在3-60之间，且不以标点符号结尾，且包含项目/系统/平台/工程/服务
            if 3 < len(proj) < 60 and proj[-1] not in "，。、；:":
                if "项目" in proj or "系统" in proj or "平台" in proj or "工程" in proj or "服务" in proj:
                    if proj not in projects:
                        projects.append(proj)
    data.projects = projects[:10]

    # 提取工作年限：多种模式匹配
    year_patterns = [
        re.compile(r"(\d{1,2})\s*[年年]\s*(?:以上)?(?:工作)?(?:经验|经历)"),
        re.compile(r"(?:工作)?(?:经验|经历)[：:]\s*(\d{1,2})\s*年"),
        re.compile(r"(\d{1,2})\s*年\s*(?:工作|开发|编程|项目)?(?:经验|经历)"),
        re.compile(r"(?:从业)?(?:经验|工龄)[：:]\s*(\d{1,2})\s*年"),
        re.compile(r"(\d{1,2})\s*年(?:以上|以下)?\s*(?:经验|经验值)"),
    ]
    for pattern in year_patterns:
        year_match = pattern.search(text)
        if year_match:
            data.experience_years = f"{year_match.group(1)}年"
            break

    # 提取学历：多种模式匹配
    edu_keywords = ["本科", "硕士", "博士", "大专", "学士", "MBA", "EMBA", "高中", "中专", "研究生", "专科"]
    edu_patterns = [
        re.compile(r"学\s*历\s*[：:]\s*(本科|硕士|博士|大专|学士|MBA|EMBA|高中|中专|研究生|专科)"),
        re.compile(r"(本科|硕士|博士|大专|学士|MBA|EMBA|高中|中专|研究生|专科)\s*(学历|毕业)"),
        re.compile(r"(本科|硕士|博士|大专|学士|MBA|EMBA|高中|中专|研究生|专科)\s*(在读|毕业|文凭|学位)"),
        re.compile(r"(本科|硕士|博士|大专|学士|MBA|EMBA|高中|中专|研究生|专科)\s*(及以上|及以下)"),
    ]
    for pattern in edu_patterns:
        edu_match = pattern.search(text)
        if edu_match:
            data.education = edu_match.group(1)
            break
    
    if not data.education:
        for keyword in edu_keywords:
            if keyword in text:
                data.education = keyword
                break

    # 生成摘要
    data.summary = _build_summary(data)

    return data


def parse_pdf(file_bytes: bytes) -> ResumeData:
    """解析 PDF 简历：先查缓存，命中直接返回；未命中则用 Apache Tika + OCR 解析后缓存

    解析策略（取消所有兜底机制，只用 Tika + OCR）：
    1. Apache Tika 提取文本（适用于文本型 PDF）
    2. 如果 Tika 提取结果有效性过低（扫描件），用 OCR 重新提取
    3. 解析完成后保存到缓存（含向量化），下次上传同一 PDF 直接返回
    """
    from ai_interviewer.resume_cache import get_cached, save_cached

    # ── 缓存命中：直接返回 ──
    cached = get_cached(file_bytes)
    if cached is not None:
        logger.info("简历缓存命中，跳过解析和向量化")
        return ResumeData(
            raw_text=cached.raw_text,
            name=cached.name,
            skills=cached.skills,
            projects=cached.projects,
            experience_years=cached.experience_years,
            education=cached.education,
            summary=cached.summary,
        )

    # ── 缓存未命中：用 Apache Tika 解析 ──
    try:
        full_text = _extract_with_tika(file_bytes)
        validity = _check_text_validity(full_text)
        logger.info("Tika 解析结果: 文本长度=%d, 有效性=%.2f", len(full_text), validity)

        # 有效性过低 → 判定为扫描件，用 OCR 重新提取
        if validity < 0.15:
            logger.info("Tika 提取有效性过低(%.2f < 0.15)，判定为扫描件，启用 OCR", validity)
            ocr_text = _extract_with_ocr(file_bytes)
            if _check_text_validity(ocr_text) > validity:
                full_text = ocr_text
                logger.info("OCR 解析结果: 文本长度=%d", len(full_text))

        if not full_text.strip():
            logger.warning("PDF 解析结果为空（Tika + OCR 均无有效文本）")
            return ResumeData(raw_text="[PDF 解析失败：Tika 和 OCR 均未能提取有效文本，请手动粘贴简历文本]")

        # 解析简历结构
        rd = parse_text(full_text)

        # ── 保存到缓存（已取消向量化，只存文本解析结果） ──
        save_cached(file_bytes, rd)

        return rd
    except Exception as e:
        logger.error("PDF 解析失败: %s", e)
        return ResumeData(raw_text=f"[PDF 解析失败: {e}]")


def _extract_with_tika(file_bytes: bytes) -> str:
    """使用 Apache Tika 提取 PDF 文本

    tika-python 会在首次调用时自动下载 tika-server.jar 并启动本地服务，
    需要 Java 运行环境（JRE）。如果 Java 未安装会直接报错。
    """
    import tika
    from tika import parser as tika_parser

    # tika.init_remote_only=False 表示用本地 Tika server
    parsed = tika_parser.from_buffer(file_bytes)
    text = parsed.get("content", "") or ""
    return text.strip()


def _check_text_validity(text: str) -> float:
    """检查文本有效性（返回0-1的分数），识别乱码并降低分数"""
    if not text:
        return 0
    total_chars = len(text.strip())
    if total_chars == 0:
        return 0

    chinese_chars = len(re.findall(r'[\u4e00-\u9fa5]', text))
    # 检测乱码：非中文、非ASCII、非空白、非标点的字符
    garbled_chars = len(re.findall(r'[^\u4e00-\u9fa5\u0000-\u007f\u3000-\u303f\uff00-\uffef\s]', text))
    printable_chars = len(re.findall(r'[a-zA-Z\u4e00-\u9fa50-9\s]', text))

    chinese_ratio = chinese_chars / total_chars
    garbled_ratio = garbled_chars / total_chars
    printable_ratio = printable_chars / total_chars

    logger.info(f"文本有效性 - 中文: {chinese_ratio:.2f}, 乱码: {garbled_ratio:.2f}, 可打印: {printable_ratio:.2f}")

    # 乱码占比高 → 大幅降低分数
    if garbled_ratio > 0.3:
        return 0.05
    if garbled_ratio > 0.15:
        return 0.1 + (chinese_ratio * 2 + printable_ratio) / 6

    return (chinese_ratio * 2 + printable_ratio) / 3


def _find_tesseract() -> str | None:
    """自动查找 Tesseract 可执行文件路径"""
    import os as _os
    candidate_paths = [
        r"D:\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in candidate_paths:
        if _os.path.exists(path):
            return path
    return None


def _extract_with_ocr(file_bytes: bytes) -> str:
    """使用 OCR 提取 PDF 文本（处理扫描件）

    使用 pdf2image 将 PDF 页面转为图片（需要系统安装 poppler），
    再用 pytesseract 进行 OCR 识别（需要系统安装 Tesseract）。
    """
    import pytesseract
    from pdf2image import convert_from_bytes
    from PIL import Image

    # 自动检测 Tesseract 路径
    tesseract_path = _find_tesseract()
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logger.info("使用 Tesseract: %s", tesseract_path)

    # 将 PDF 每页转为 300dpi 的 PIL Image
    images = convert_from_bytes(file_bytes, dpi=300)
    text_parts: list[str] = []
    for img in images:
        page_text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        if page_text:
            text_parts.append(page_text)
    return "\n".join(text_parts)


def _build_summary(data: ResumeData) -> str:
    """根据提取的信息生成简历摘要"""
    parts: list[str] = []
    if data.name:
        parts.append(f"候选人: {data.name}")
    if data.experience_years:
        parts.append(f"工作经验: {data.experience_years}")
    if data.education:
        parts.append(f"学历: {data.education}")
    if data.skills:
        parts.append(f"技术栈: {', '.join(data.skills[:15])}")
    if data.projects:
        parts.append(f"项目经历: {', '.join(data.projects[:5])}")
    return "\n".join(parts) if parts else data.raw_text[:500]
