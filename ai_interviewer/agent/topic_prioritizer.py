"""话题优先级规划器 - 可配置 + 可扩展的多因子评分系统

替代硬编码的 topic_priority = {"redis": 1, "java": 2, ...}。

## 设计思路：不再用死数字，而是 4 层可插拔评分因子

1. **Category 基础权重**（最大影响因子）：把技术栈分大类，每类自带优先级等级
   - AI_AGENT_CORE（最高，先考察核心工程化能力）
   - DISTRIBUTED_INFRA（次高，分布式/高并发是后端硬通货）
   - DATA_LAYER（数据库/缓存/存储）
   - LANGUAGE_FRAMEWORK（语言 + 通用框架）
   - TOOLING_AUXILIARY（容器/CI/CD等辅助工具，最低）

2. **Resume Match 深度加成**（简历命中深度）：
   - 简历项目描述里具体提到，有上下文 → 加成高
   - 只是 skills 列表里的干关键词 → 加成低
   - 完全没出现在简历里 → 惩罚（避免面试官自嗨问一堆不相关的）

3. **Knowledge Base 覆盖加成**：RAG 知识库有对应题目的技能 → 加分，保证有参考题可问

4. **Topic Clustering 聚类连续性**：同类话题聚类在一起连续问，避免 Redis→Python→Kafka 这种跳来跳去打断思路

5. **可插拔自定义规则**：`PrioritizerRule` 协议 + `register_rule()`，扩展时不需要改核心
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
#  第一层：分类基础权重（数字越小优先级越高）
# ═══════════════════════════════════════════

class TopicCategory:
    """技术栈分类 + 基础优先级。数值越小 → 越先问。"""
    AI_AGENT_CORE = 10          # RAG / Embedding / LLM / Semantic Cache / Agent框架
    DISTRIBUTED_INFRA = 20      # 分布式 / 高并发 / 消息队列 / 分布式事务 / 缓存策略
    DATA_LAYER = 30             # 数据库 / Redis / ES / 存储引擎
    LANGUAGE_FRAMEWORK = 40     # Java / Python / Go / Spring / Django / FastAPI
    TOOLING_AUXILIARY = 60      # Docker / K8s / CI/CD / 监控 / 云平台
    UNCLASSIFIED = 80           # 未知分类兜底


# ═══════════════════════════════════════════
#  技能 → 分类映射 + 别名归一化（可扩展，随时加条目）
# ═══════════════════════════════════════════

# key 是**小写归一化后的别名**，value 是 (标准技能名, 分类权重)
# 新增技能只需在这里加一行，不用改核心代码
_SKILL_ALIAS_TABLE: dict[str, tuple[str, int]] = {
    # ── AI Agent 核心（最高优先级） ──
    "rag": ("RAG", TopicCategory.AI_AGENT_CORE),
    "retrieval": ("RAG", TopicCategory.AI_AGENT_CORE),
    "retrieval augmented generation": ("RAG", TopicCategory.AI_AGENT_CORE),
    "embedding": ("Embedding", TopicCategory.AI_AGENT_CORE),
    "向量": ("Embedding", TopicCategory.AI_AGENT_CORE),
    "llm": ("LLM", TopicCategory.AI_AGENT_CORE),
    "大模型": ("LLM", TopicCategory.AI_AGENT_CORE),
    "gpt": ("LLM", TopicCategory.AI_AGENT_CORE),
    "prompt": ("Prompt Engineering", TopicCategory.AI_AGENT_CORE),
    "prompt engineering": ("Prompt Engineering", TopicCategory.AI_AGENT_CORE),
    "语义缓存": ("Semantic Cache", TopicCategory.AI_AGENT_CORE),
    "semantic cache": ("Semantic Cache", TopicCategory.AI_AGENT_CORE),
    "agent": ("AI Agent", TopicCategory.AI_AGENT_CORE),
    "ai agent": ("AI Agent", TopicCategory.AI_AGENT_CORE),
    "langchain": ("LangChain/LangGraph", TopicCategory.AI_AGENT_CORE),
    "langgraph": ("LangChain/LangGraph", TopicCategory.AI_AGENT_CORE),
    "function calling": ("Function Calling", TopicCategory.AI_AGENT_CORE),
    "工具调用": ("Function Calling", TopicCategory.AI_AGENT_CORE),
    "react": ("ReAct Agent", TopicCategory.AI_AGENT_CORE),  # 注意：和前端 React 重名，需上下文区分
    "react agent": ("ReAct Agent", TopicCategory.AI_AGENT_CORE),

    # ── 分布式基础设施（次高） ──
    "分布式": ("分布式系统", TopicCategory.DISTRIBUTED_INFRA),
    "distributed": ("分布式系统", TopicCategory.DISTRIBUTED_INFRA),
    "分布式锁": ("分布式锁", TopicCategory.DISTRIBUTED_INFRA),
    "高并发": ("高并发系统", TopicCategory.DISTRIBUTED_INFRA),
    "高可用": ("高可用架构", TopicCategory.DISTRIBUTED_INFRA),
    "cap": ("CAP 理论", TopicCategory.DISTRIBUTED_INFRA),
    "raft": ("Raft/Paxos", TopicCategory.DISTRIBUTED_INFRA),
    "paxos": ("Raft/Paxos", TopicCategory.DISTRIBUTED_INFRA),
    "一致性": ("分布式一致性", TopicCategory.DISTRIBUTED_INFRA),
    "幂等": ("幂等性", TopicCategory.DISTRIBUTED_INFRA),
    "kafka": ("Kafka", TopicCategory.DISTRIBUTED_INFRA),
    "消息队列": ("消息队列", TopicCategory.DISTRIBUTED_INFRA),
    "mq": ("消息队列", TopicCategory.DISTRIBUTED_INFRA),
    "rabbitmq": ("消息队列", TopicCategory.DISTRIBUTED_INFRA),
    "rocketmq": ("消息队列", TopicCategory.DISTRIBUTED_INFRA),
    "分布式事务": ("分布式事务", TopicCategory.DISTRIBUTED_INFRA),
    "seata": ("分布式事务", TopicCategory.DISTRIBUTED_INFRA),
    "tcc": ("分布式事务", TopicCategory.DISTRIBUTED_INFRA),
    "saga": ("分布式事务", TopicCategory.DISTRIBUTED_INFRA),
    "缓存穿透": ("缓存三件套", TopicCategory.DISTRIBUTED_INFRA),
    "缓存击穿": ("缓存三件套", TopicCategory.DISTRIBUTED_INFRA),
    "缓存雪崩": ("缓存三件套", TopicCategory.DISTRIBUTED_INFRA),
    "限流": ("接口限流", TopicCategory.DISTRIBUTED_INFRA),
    "sentinel": ("接口限流", TopicCategory.DISTRIBUTED_INFRA),
    "分库分表": ("分库分表", TopicCategory.DISTRIBUTED_INFRA),
    "sharding": ("分库分表", TopicCategory.DISTRIBUTED_INFRA),
    "shardingsphere": ("分库分表", TopicCategory.DISTRIBUTED_INFRA),

    # ── 数据层 ──
    "mysql": ("MySQL", TopicCategory.DATA_LAYER),
    "数据库": ("MySQL", TopicCategory.DATA_LAYER),
    "数据库原理": ("MySQL", TopicCategory.DATA_LAYER),
    "索引": ("MySQL 索引", TopicCategory.DATA_LAYER),
    "mvcc": ("MySQL 事务", TopicCategory.DATA_LAYER),
    "事务": ("MySQL 事务", TopicCategory.DATA_LAYER),
    "acid": ("MySQL 事务", TopicCategory.DATA_LAYER),
    "redis": ("Redis", TopicCategory.DATA_LAYER),
    "redisson": ("Redis", TopicCategory.DATA_LAYER),
    "缓存": ("Redis", TopicCategory.DATA_LAYER),
    "mongodb": ("MongoDB", TopicCategory.DATA_LAYER),
    "elasticsearch": ("Elasticsearch", TopicCategory.DATA_LAYER),
    "es": ("Elasticsearch", TopicCategory.DATA_LAYER),
    "postgresql": ("PostgreSQL", TopicCategory.DATA_LAYER),
    "pg": ("PostgreSQL", TopicCategory.DATA_LAYER),

    # ── 语言 & 框架 ──
    "java": ("Java", TopicCategory.LANGUAGE_FRAMEWORK),
    "jvm": ("JVM", TopicCategory.LANGUAGE_FRAMEWORK),
    "gc": ("JVM GC", TopicCategory.LANGUAGE_FRAMEWORK),
    "垃圾回收": ("JVM GC", TopicCategory.LANGUAGE_FRAMEWORK),
    "并发": ("Java 并发", TopicCategory.LANGUAGE_FRAMEWORK),
    "多线程": ("Java 并发", TopicCategory.LANGUAGE_FRAMEWORK),
    "线程池": ("Java 并发", TopicCategory.LANGUAGE_FRAMEWORK),
    "锁": ("Java 锁机制", TopicCategory.LANGUAGE_FRAMEWORK),
    "synchronized": ("Java 锁机制", TopicCategory.LANGUAGE_FRAMEWORK),
    "aqs": ("Java 锁机制", TopicCategory.LANGUAGE_FRAMEWORK),
    "spring": ("Spring", TopicCategory.LANGUAGE_FRAMEWORK),
    "springboot": ("Spring Boot", TopicCategory.LANGUAGE_FRAMEWORK),
    "spring boot": ("Spring Boot", TopicCategory.LANGUAGE_FRAMEWORK),
    "spring cloud": ("Spring Cloud", TopicCategory.LANGUAGE_FRAMEWORK),
    "ioc": ("Spring IoC/AOP", TopicCategory.LANGUAGE_FRAMEWORK),
    "aop": ("Spring IoC/AOP", TopicCategory.LANGUAGE_FRAMEWORK),
    "mybatis": ("MyBatis", TopicCategory.LANGUAGE_FRAMEWORK),
    "python": ("Python", TopicCategory.LANGUAGE_FRAMEWORK),
    "fastapi": ("FastAPI", TopicCategory.LANGUAGE_FRAMEWORK),
    "django": ("Django", TopicCategory.LANGUAGE_FRAMEWORK),
    "flask": ("Flask", TopicCategory.LANGUAGE_FRAMEWORK),
    "go": ("Go", TopicCategory.LANGUAGE_FRAMEWORK),
    "golang": ("Go", TopicCategory.LANGUAGE_FRAMEWORK),
    "rust": ("Rust", TopicCategory.LANGUAGE_FRAMEWORK),
    "c++": ("C++", TopicCategory.LANGUAGE_FRAMEWORK),
    "javascript": ("JavaScript", TopicCategory.LANGUAGE_FRAMEWORK),
    "js": ("JavaScript", TopicCategory.LANGUAGE_FRAMEWORK),
    "typescript": ("TypeScript", TopicCategory.LANGUAGE_FRAMEWORK),
    "ts": ("TypeScript", TopicCategory.LANGUAGE_FRAMEWORK),

    # ── 工具 & 辅助（最低优先级） ──
    "docker": ("Docker", TopicCategory.TOOLING_AUXILIARY),
    "容器": ("Docker", TopicCategory.TOOLING_AUXILIARY),
    "kubernetes": ("Kubernetes", TopicCategory.TOOLING_AUXILIARY),
    "k8s": ("Kubernetes", TopicCategory.TOOLING_AUXILIARY),
    "jenkins": ("CI/CD", TopicCategory.TOOLING_AUXILIARY),
    "ci/cd": ("CI/CD", TopicCategory.TOOLING_AUXILIARY),
    "cicd": ("CI/CD", TopicCategory.TOOLING_AUXILIARY),
    "devops": ("DevOps", TopicCategory.TOOLING_AUXILIARY),
    "git": ("Git", TopicCategory.TOOLING_AUXILIARY),
    "linux": ("Linux", TopicCategory.TOOLING_AUXILIARY),
    "nginx": ("Nginx", TopicCategory.TOOLING_AUXILIARY),
    "aws": ("云平台", TopicCategory.TOOLING_AUXILIARY),
    "azure": ("云平台", TopicCategory.TOOLING_AUXILIARY),
    "gcp": ("云平台", TopicCategory.TOOLING_AUXILIARY),
    "云原生": ("云原生", TopicCategory.TOOLING_AUXILIARY),
    "微服务": ("微服务架构", TopicCategory.LANGUAGE_FRAMEWORK),  # 放在框架类，略高于工具
    "microservices": ("微服务架构", TopicCategory.LANGUAGE_FRAMEWORK),
}


# 同义词聚类组：同一组的话题排在一起连续问，避免跳题
_TOPIC_CLUSTERS: list[set[str]] = [
    {"RAG", "Embedding", "Semantic Cache", "LLM", "Prompt Engineering",
     "AI Agent", "LangChain/LangGraph", "Function Calling", "ReAct Agent"},
    {"分布式系统", "CAP 理论", "Raft/Paxos", "分布式一致性", "分布式锁",
     "分布式事务", "幂等性"},
    {"高并发系统", "高可用架构", "缓存三件套", "接口限流"},
    {"Kafka", "消息队列"},
    {"分库分表", "MySQL", "MySQL 索引", "MySQL 事务"},
    {"Redis"},
    {"MongoDB", "Elasticsearch", "PostgreSQL"},
    {"Java", "JVM", "JVM GC", "Java 并发", "Java 锁机制"},
    {"Spring", "Spring Boot", "Spring IoC/AOP", "MyBatis", "Spring Cloud"},
    {"Python", "FastAPI", "Django", "Flask"},
    {"Docker", "Kubernetes", "微服务架构"},
    {"CI/CD", "DevOps", "云原生", "云平台", "Linux", "Nginx", "Git"},
]


# ═══════════════════════════════════════════
#  评分规则协议：扩展时实现并注册，不需要改核心逻辑
# ═══════════════════════════════════════════

class PrioritizerContext:
    """评分上下文：所有评分因子共享的输入数据。"""

    def __init__(
        self,
        skills: Iterable[str],
        resume_text: str = "",
        kb_topics: Iterable[str] | None = None,
    ) -> None:
        self.skills: list[str] = [s.strip() for s in skills if s and s.strip()]
        self.resume_text_lower: str = (resume_text or "").lower()
        self.kb_topics: set[str] = {t.lower() for t in (kb_topics or [])}


@dataclass
class SkillScore:
    """单技能评分结果。"""
    original: str                        # 简历原始技能名
    canonical: str                       # 归一化后的标准技能名
    category: int                        # 分类权重（越小越优先）
    resume_depth: int = 0                # 简历命中深度加成（越大越优先）
    kb_bonus: int = 0                    # 知识库覆盖加成
    cluster_group: int = -1              # 聚类组 id（同组话题排在一起）
    custom_bonus: int = 0                # 自定义规则加成总和

    @property
    def total(self) -> int:
        """综合得分（越小越优先，类似 Unix nice 值）。"""
        return self.category + self.resume_depth + self.kb_bonus + self.custom_bonus


# 评分规则函数签名：输入(ctx, skill_score) → 返回 bonus 整数（负数=更优先）
PrioritizerRule = Callable[[PrioritizerContext, SkillScore], int]


# ═══════════════════════════════════════════
#  默认内置规则（4 个因子）
# ═══════════════════════════════════════════

def _rule_resume_depth(ctx: PrioritizerContext, score: SkillScore) -> int:
    """规则1：简历命中深度 —— 项目描述里具体提到的加分，只有干名词的不加。"""
    text = ctx.resume_text_lower
    if not text:
        return 0
    original = score.original.lower()
    canonical = score.canonical.lower()
    bonus = 0
    # 在简历正文（非技能列表区）里出现 2 次以上 → 说明有项目经验，加成 -8（更优先）
    hit_count = max(text.count(original), text.count(canonical))
    if hit_count >= 3:
        bonus -= 10
    elif hit_count >= 2:
        bonus -= 6
    elif hit_count == 1:
        bonus -= 2
    # 包含"项目" / "经验" / "负责" 等上下文词 → 再额外 -3
    context_markers = ["项目", "经验", "负责", "设计", "实现", "优化", "线上", "生产"]
    if any(m in text for m in context_markers) and (hit_count >= 1):
        bonus -= 3
    return bonus


def _rule_knowledge_base_coverage(ctx: PrioritizerContext, score: SkillScore) -> int:
    """规则2：知识库覆盖度 —— RAG 里有的题优先，保证面试官能参考着问。"""
    if not ctx.kb_topics:
        return 0
    canon = score.canonical.lower()
    orig = score.original.lower()
    for kb_topic in ctx.kb_topics:
        if canon in kb_topic or kb_topic in canon or orig in kb_topic or kb_topic in orig:
            return -5  # 有知识库支撑，更优先
    return 3  # 没有知识库支撑，适当延后（仍会问，但不排在最前面）


def _rule_canonical_alias_match(_ctx: PrioritizerContext, score: SkillScore) -> int:
    """规则3：归一化一致性 —— 能映射到标准技能名的优先，未知分类兜底惩罚。"""
    if score.category == TopicCategory.UNCLASSIFIED:
        return 12  # 分类未知，放后面
    return 0


_DEFAULT_RULES: list[tuple[str, PrioritizerRule]] = [
    ("resume_depth", _rule_resume_depth),
    ("kb_coverage", _rule_knowledge_base_coverage),
    ("alias_match", _rule_canonical_alias_match),
]


# ═══════════════════════════════════════════
#  TopicPrioritizer 核心类
# ═══════════════════════════════════════════

class TopicPrioritizer:
    """可配置 + 可扩展的话题优先级规划器。

    用法：
        prioritizer = TopicPrioritizer()

        # 扩展：注册自定义规则（可选）
        @prioritizer.register_rule("my_rule", weight=1)
        def prefer_kafka(ctx, score):
            if score.canonical == "Kafka":
                return -20  # 我特别想先问 Kafka
            return 0

        # 排序：返回 [(原始技能名, 标准技能名)]
        ordered = prioritizer.prioritize(
            skills=["Redis", "Java", "RAG", "Docker", "Kafka"],
            resume_text="...简历全文...",
        )
    """

    def __init__(self) -> None:
        # 规则列表：(name, rule_fn)，按注册顺序执行
        self._rules: list[tuple[str, PrioritizerRule]] = list(_DEFAULT_RULES)
        # 允许外部覆盖别名表（不改源码时用）
        self.extra_aliases: dict[str, tuple[str, int]] = {}

    # ── 扩展 API ──

    def register_rule(self, name: str, rule: PrioritizerRule | None = None,
                      weight: int = 1) -> Callable[[PrioritizerRule], PrioritizerRule]:
        """注册自定义评分规则。支持装饰器或直接传函数。

        Args:
            name: 规则名，用于去重和日志
            rule: 规则函数（装饰器模式下为 None）
            weight: 预留权重参数（当前规则本身返回已带符号，默认×1）
        """
        def _decorator(fn: PrioritizerRule) -> PrioritizerRule:
            # 同名去重：先移除旧的
            self._rules = [(n, r) for (n, r) in self._rules if n != name]
            if weight == 1:
                self._rules.append((name, fn))
            else:
                wrapped: PrioritizerRule = lambda ctx, sc: fn(ctx, sc) * weight
                self._rules.append((name, wrapped))
            logger.info("已注册话题优先级规则: %s (weight=%d)", name, weight)
            return fn

        if rule is not None:
            return _decorator(rule)
        return _decorator

    def add_aliases(self, aliases: dict[str, tuple[str, int]]) -> None:
        """批量添加技能别名映射（不修改源码时扩展）。

        aliases 格式: {"别名小写": ("标准名", 分类权重TopicCategory.XXX)}
        """
        self.extra_aliases.update({k.lower(): v for k, v in aliases.items()})

    # ── 核心排序 ──

    def prioritize(
        self,
        skills: Iterable[str],
        resume_text: str = "",
        kb_topics: Iterable[str] | None = None,
    ) -> list[tuple[str, str]]:
        """对技能列表进行优先级排序。

        Returns:
            list of (原始技能名, 标准技能名)，按优先级从高到低排列
        """
        ctx = PrioritizerContext(skills, resume_text=resume_text, kb_topics=kb_topics)
        if not ctx.skills:
            return [("技术基础", "技术基础")]

        # Step 1: 每个技能 → 归一化 SkillScore
        scored: list[SkillScore] = []
        seen_canonical: set[str] = set()
        for skill in ctx.skills:
            score = self._resolve_skill(skill)
            # 聚类组：同一 cluster 的话题后面排在一起
            score.cluster_group = self._find_cluster(score.canonical)
            scored.append(score)

        # Step 2: 应用所有评分规则
        for name, rule in self._rules:
            for s in scored:
                try:
                    bonus = rule(ctx, s)
                    s.custom_bonus += int(bonus or 0)
                except Exception as e:
                    logger.warning("规则[%s]执行异常，跳过: %s", name, e)

        # Step 3: 排序
        # 主键: total 得分（越小越优先）；次键: 聚类组（同类连续问）；三键: 原始顺序稳定
        scored_with_index = list(enumerate(scored))
        scored_with_index.sort(key=lambda item: (
            item[1].total,
            1000 if item[1].cluster_group < 0 else item[1].cluster_group,
            item[0],
        ))

        # Step 4: 聚类重排（在总分类相近的前提下，同一 cluster 尽量贴在一起）
        ordered = self._cluster_aware_reorder([s for _, s in scored_with_index])

        # Step 5: 去重（相同标准名保留第一个），并组装结果
        result: list[tuple[str, str]] = []
        for s in ordered:
            canon = s.canonical
            if canon in seen_canonical:
                # 标准名重复：保留原始名但标注为变体（加后缀），实际面试只问一次
                continue
            seen_canonical.add(canon)
            result.append((s.original, s.canonical))

        if not result:
            return [("技术基础", "技术基础")]

        # 日志：结果按顺序打印 (标准名=得分)，ordered 和 result 一一对应（长度一致）
        log_items = []
        for score_obj, (orig_name, canon_name) in zip(ordered, result):
            log_items.append(f"{canon_name}={score_obj.total}")
        logger.info("话题规划排序结果(前8): %s", " → ".join(log_items[:8]))
        return result

    # ── 内部辅助 ──

    def _resolve_skill(self, skill: str) -> SkillScore:
        """把简历原始技能名 → 归一化 (标准名, 分类权重)。"""
        key = skill.strip().lower()
        # 1) 先查用户自定义别名
        if key in self.extra_aliases:
            canonical, category = self.extra_aliases[key]
            return SkillScore(original=skill, canonical=canonical, category=category)
        # 2) 查内置别名表
        if key in _SKILL_ALIAS_TABLE:
            canonical, category = _SKILL_ALIAS_TABLE[key]
            return SkillScore(original=skill, canonical=canonical, category=category)
        # 3) 模糊匹配：别名表 key 是技能名的子串
        for alias_key, (canonical, category) in _SKILL_ALIAS_TABLE.items():
            if alias_key and (alias_key in key or key in alias_key):
                return SkillScore(original=skill, canonical=canonical, category=category)
        # 4) 兜底：未分类
        return SkillScore(original=skill, canonical=skill.strip() or "未知技能",
                          category=TopicCategory.UNCLASSIFIED)

    @staticmethod
    def _find_cluster(canonical: str) -> int:
        """查找技能所属的聚类组 id（同类连续问）。"""
        for idx, cluster in enumerate(_TOPIC_CLUSTERS):
            if canonical in cluster:
                return idx
        return -1

    @staticmethod
    def _cluster_aware_reorder(scored: list[SkillScore]) -> list[SkillScore]:
        """聚类感知的轻量重排：在得分相近的相邻项中，把同组贴在一起。

        不改变大的得分顺序，只在局部（分差<=15）内把同 cluster 聚拢。
        """
        if len(scored) < 3:
            return scored
        result = list(scored)
        changed = True
        rounds = 0
        while changed and rounds < 4:  # 最多 4 轮冒泡式重排
            changed = False
            rounds += 1
            for i in range(1, len(result)):
                prev, cur = result[i - 1], result[i]
                if prev.cluster_group == cur.cluster_group and prev.cluster_group >= 0:
                    continue
                # 如果 cur 和 result[i-2] 同组，且 prev 和 cur 分差 <= 15，则交换 prev 和 cur
                if (i >= 2 and result[i - 2].cluster_group == cur.cluster_group >= 0
                        and abs(prev.total - cur.total) <= 15):
                    result[i - 1], result[i] = cur, prev
                    changed = True
        return result


# ═══════════════════════════════════════════
#  JD 驱动的 LLM 动态话题排序（主路径）
# ═══════════════════════════════════════════

@dataclass
class JDPriorityResult:
    """LLM 根据 JD 输出的结构化优先级结果。"""
    ordered_topics: list[str]                            # 从高到低排好序的标准话题名
    ordered_aliases: list[tuple[str, str]]               # [(简历原始技能名, 标准名)] 兼容老接口
    skill_importance: dict[str, str]                     # {标准名: "required"|"preferred"|"bonus"}
    suggested_max_questions: int                         # 根据岗位密度建议的最大题数
    topics_per_skill: dict[str, int]                     # {"required":4, "preferred":3, "bonus":2}
    source: str = "llm"                                  # "llm" 或 "fallback"（排障用）

    def importance_of(self, topic: str) -> str:
        """按标准名查要求强度，找不到则兜底 'preferred'。"""
        imp = self.skill_importance.get(topic)
        if imp in {"required", "preferred", "bonus"}:
            return imp
        # 模糊匹配：键是 topic 的子串或反
        for k, v in self.skill_importance.items():
            if k and (k in topic or topic in k):
                return v if v in {"required", "preferred", "bonus"} else "preferred"
        return "preferred"


def prioritize_by_jd(
    *,
    api_key: str,
    base_url: str,
    model: str,
    skills: Iterable[str],
    jd_position_name: str = "",
    jd_raw_text: str = "",
    temperature: float = 0.05,
) -> JDPriorityResult:
    """JD 驱动的话题优先级排序。

    预置 JD：优先读缓存（首次调 LLM 生成后保存），秒级返回。
    自定义 JD：每次现场调 LLM 生成。
    任何失败直接抛错，不降级。
    """
    from ai_interviewer.jd_engine.jd_loader import PRIORITY_SYSTEM_PROMPT

    skill_list = [s.strip() for s in skills if s and s.strip()]

    # ── 1. 尝试读 JD 缓存（预置 JD 首次生成后保存，后续秒读）──
    cached = _get_jd_cache(jd_raw_text)
    if cached is not None:
        return _build_result_from_cache(skill_list, cached, jd_position_name)
    user_prompt = f"""## 目标岗位
{jd_position_name or "未指定岗位"}

## 岗位 JD 原文
{jd_raw_text or "(未提供 JD，按通用技术岗排序)"}

## 候选人简历技能列表（仅参考：候选人会什么；话题本身必须来自 JD 原文，技能列表不是话题来源）
{', '.join(skill_list) if skill_list else "(未提供，话题完全从 JD 原文提取)"}

按规则输出结构化 JSON。"""

    topics_per_skill_default = {"required": 4, "preferred": 3, "bonus": 2}
    topics_per_skill = dict(topics_per_skill_default)
    try:
        from openai import OpenAI
        import json as _json

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60, max_retries=0)
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=4096,
            messages=[
                {"role": "system", "content": PRIORITY_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = resp.choices[0].message.content or ""
        if not content:
            raise RuntimeError("JD 排序 LLM 调用返回空内容")
        data = _extract_json(content)
        ordered_topics = [str(t).strip() for t in data.get("ordered_topics", []) if str(t).strip()]
        importance_raw = data.get("skill_importance", {}) or {}
        importance = {}
        for k, val in importance_raw.items():
            v = str(val).strip().lower()
            importance[str(k).strip()] = v if v in {"required", "preferred", "bonus"} else "preferred"
        max_q = int(data.get("suggested_max_questions", 15) or 15)
        tps_raw = data.get("topics_per_skill", {}) or {}
        for k, default_v in topics_per_skill_default.items():
            if k in tps_raw:
                try:
                    topics_per_skill[k] = int(tps_raw[k])
                except (TypeError, ValueError):
                    topics_per_skill[k] = default_v

        if not ordered_topics:
            raise ValueError("LLM 返回的 ordered_topics 为空，请检查模型是否遵循 JSON 输出格式")
    except Exception as e:
        # 任何失败都不做降级（API Key 错了继续跑毫无意义），把错误上下文包装后直接抛出
        err_type = type(e).__name__
        # 脱敏 api_key（只显示前 8 位后 4 位，防止日志/前端泄露 key）
        if not api_key:
            masked_key = "(空)"
        elif len(api_key) <= 8:
            masked_key = "***" + (api_key[-4:] if len(api_key) > 4 else "")
        else:
            masked_key = api_key[:8] + "..." + api_key[-4:]
        msg = (
            f"JD 话题排序失败，面试无法开始：[{err_type}] {str(e) or '(无错误消息)'}。"
            f" | model={model} | base_url={base_url} | api_key={masked_key}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e

    # 建立 标准名 → 简历原始名 映射（LLM 返回 ordered_topics 是标准名，要回造 aliases）
    aliases: list[tuple[str, str]] = []
    lower_to_original = {s.lower(): s for s in skill_list}
    for canon in ordered_topics:
        orig = lower_to_original.get(canon.lower(), canon)
        # 模糊：canon 是 orig 子串 / 反
        if orig == canon:
            for orig_s in skill_list:
                if canon.lower() in orig_s.lower() or orig_s.lower() in canon.lower():
                    orig = orig_s
                    break
        aliases.append((orig, canon))

    # max_q 合法范围夹取（LLM 乱给数字时的兜底，不是规则降级）
    max_q = max(8, min(25, int(max_q)))

    logger.info(
        "JD 话题排序: position=%s max_q=%d topics=%s",
        jd_position_name, max_q, ordered_topics[:10],
    )

    # ── 保存 JD 缓存（下次同 JD 秒读）──
    _save_jd_cache(jd_raw_text, {
        "skill_importance": importance,
        "suggested_max_questions": max_q,
        "topics_per_skill": topics_per_skill,
    })

    return JDPriorityResult(
        ordered_topics=ordered_topics,
        ordered_aliases=aliases,
        skill_importance=importance,
        suggested_max_questions=max_q,
        topics_per_skill=topics_per_skill,
        source="llm",
    )


# ── JD 缓存（预置 JD 首次 LLM 生成后保存，后续秒读，自定义 JD 也缓存）──

_JD_CACHE_DIR = Path(__file__).resolve().parent.parent / ".jd_cache"

# 话题名后缀词（归一化用：'RAG技术'/'RAG系统'/'RAG实践' → 'RAG'）
_TOPIC_SUFFIXES = (
    "架构", "原理", "系统", "技术", "实践", "基础", "能力", "框架", "工具",
    "经验", "设计", "开发", "应用", "相关", "语言", "协议", "编排", "协作",
)


def _topic_core(topic: str) -> str:
    """剥离话题名尾部修饰词，取核心词（'RAG技术'→'rag'），用于宽松匹配。"""
    t = topic.strip().lower()
    changed = True
    while changed and t:
        changed = False
        for suf in _TOPIC_SUFFIXES:
            if t.endswith(suf) and len(t) - len(suf) >= 2:
                t = t[: -len(suf)]
                changed = True
    return t


def _jd_cache_path(jd_raw_text: str) -> Path:
    h = hashlib.sha256(jd_raw_text.strip().encode("utf-8")).hexdigest()[:16]
    return _JD_CACHE_DIR / f"{h}.json"


def _get_jd_cache(jd_raw_text: str) -> dict | None:
    """读取 JD 缓存（skill_importance + max_questions + topics_per_skill）"""
    if not jd_raw_text or not jd_raw_text.strip():
        return None
    p = _jd_cache_path(jd_raw_text)
    if not p.exists():
        return None
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
        logger.info("JD 缓存命中，跳过 LLM 排序调用")
        return data
    except Exception as e:
        logger.warning("JD 缓存读取失败，将重新调 LLM: %s", e)
        return None


def _save_jd_cache(jd_raw_text: str, data: dict) -> None:
    """保存 JD 缓存"""
    if not jd_raw_text or not jd_raw_text.strip():
        return
    try:
        _JD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _jd_cache_path(jd_raw_text)
        p.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("JD 缓存已保存: %s", p.name)
    except Exception as e:
        logger.warning("JD 缓存保存失败: %s", e)


def _build_result_from_cache(skill_list: list[str], cache: dict, jd_position_name: str) -> JDPriorityResult:
    """用缓存的 skill_importance + 候选人 skills 排序，构建 JDPriorityResult

    匹配策略（宽松归一化）：
    1. 精确命中
    2. 互为子串（'Redis' ↔ 'Redis缓存'）
    3. 核心词互为子串（'RAG技术' ↔ 'RAG系统' → core 'rag'）
    """
    importance = cache.get("skill_importance", {})
    max_q = int(cache.get("suggested_max_questions", 15) or 15)
    topics_per_skill = cache.get("topics_per_skill", {"required": 4, "preferred": 3, "bonus": 2})

    imp_values = {"required", "preferred", "bonus"}
    cores = {k: _topic_core(k) for k in importance}

    def _get_imp(skill: str) -> str:
        imp = importance.get(skill)
        if imp in imp_values:
            return imp
        sl = skill.lower()
        sk_core = _topic_core(sl)
        for k, v in importance.items():
            if v not in imp_values:
                continue
            kl = k.lower()
            if kl in sl or sl in kl:
                return v
            kc = cores.get(k, "")
            if kc and sk_core and (kc in sk_core or sk_core in kc):
                return v
        return "preferred"

    rank_map = {"required": 0, "preferred": 1, "bonus": 2}
    ordered_topics = sorted(skill_list, key=lambda s: rank_map.get(_get_imp(s), 1))
    aliases = [(t, t) for t in ordered_topics]

    logger.info("JD 缓存排序: position=%s topics=%s", jd_position_name, ordered_topics[:10])
    return JDPriorityResult(
        ordered_topics=ordered_topics,
        ordered_aliases=aliases,
        skill_importance=importance,
        suggested_max_questions=max_q,
        topics_per_skill=topics_per_skill,
        source="cache",
    )


def _extract_json(text: str) -> dict:
    """从 LLM 输出里抽取 JSON，容忍 markdown 代码块/前后冗余文本。"""
    import json as _json
    if not text:
        return {}
    content = text.strip()
    # 去除 ```json ... ```
    if content.startswith("```"):
        parts = content.split("```")
        for part in parts[1:]:
            inner = part.strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            if inner.startswith("{"):
                content = inner
                break
    try:
        return _json.loads(content)
    except _json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return _json.loads(content[start:end + 1])
        except _json.JSONDecodeError:
            pass
    return {}


# ═══════════════════════════════════════════
#  路由自然转换：按配额 + 题数决定追问 / 换题 / 结束
#  （已取消 quality_level 评分机制，不再依赖回答质量等级）
# ═══════════════════════════════════════════

def make_route_decision(
    *,
    importance: str,                              # required / preferred / bonus（决定配额）
    questions_on_this_topic: int,                 # 当前话题已经问了几轮
    questions_per_skill: dict[str, int],          # {"required":4, "preferred":3, "bonus":2}
    question_count: int,                          # 当前总题数
    max_questions: int,                           # 最大题数（兜底 15）
    current_topic_index: int,                     # 当前话题索引
    total_topics: int,                            # 总话题数
) -> str:
    """自然状态转换：不依赖回答质量，默认在本话题追问到配额满再换题。

    转换规则（从上到下短路）：
    1. 题数到顶 → end
    2. 本话题配额已满且还有后续话题 → next_topic
    3. 已是最后一个话题 → followup（在末话题深挖到题数到顶）
    4. 剩题不足以覆盖剩余话题 → next_topic（压缩追问，快速扫完）
    5. 默认 → followup（自然追问）

    返回值与原 decide_next 兼容："end" / "followup" / "next_topic"。
    """
    importance = importance if importance in {"required", "preferred", "bonus"} else "preferred"
    quota = int((questions_per_skill or {}).get(importance, 3))

    # ── 1. 题数到顶 → 直接结束 ──
    if question_count >= max_questions:
        return "end"
    # 最后一题刚好到上限 → 结束
    if question_count + 1 > max_questions:
        return "end"

    # ── 2. 本话题配额已满且还有后续话题 → 换题 ──
    if questions_on_this_topic >= quota and current_topic_index < total_topics - 1:
        decision = "next_topic"
    # ── 3. 已是最后一个话题 → 继续深挖（直到题数到顶由规则 1 兜底） ──
    elif total_topics > 0 and current_topic_index >= total_topics - 1:
        decision = "followup"
    # ── 4. 剩题不足以覆盖剩余话题 → 压缩追问，快速换题扫完 ──
    elif (total_topics - current_topic_index - 1) > 1 and \
         (max_questions - question_count - 1) < (total_topics - current_topic_index - 1):
        decision = "next_topic"
    # ── 5. 默认：自然追问 ──
    else:
        decision = "followup"

    logger.info(
        "路由决策: importance=%s decision=%s | topic_round=%d/%d q_count=%d/%d idx=%d/%d",
        importance, decision,
        questions_on_this_topic, quota,
        question_count, max_questions,
        current_topic_index, max(1, total_topics),
    )
    return decision


# ═══════════════════════════════════════════
#  全局单例（大部分场景直接用默认实例即可）
# ═══════════════════════════════════════════

_default_prioritizer: TopicPrioritizer | None = None


def get_topic_prioritizer() -> TopicPrioritizer:
    """获取默认 TopicPrioritizer 单例（需要扩展时创建自己的实例）。"""
    global _default_prioritizer
    if _default_prioritizer is None:
        _default_prioritizer = TopicPrioritizer()
    return _default_prioritizer
