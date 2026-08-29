"""数据模型定义 - 项目全局共享的 Pydantic 模型"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════
#  API 管理相关模型
# ═══════════════════════════════════════════

class ModelStatus(str, Enum):
    """模型状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEGRADED = "degraded"


class ProviderType(str, Enum):
    """Provider 类型枚举"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    CUSTOM = "custom"


class ModelConfig(BaseModel):
    """模型配置"""
    id: str = Field(description="模型配置唯一 ID")
    name: str = Field(description="模型显示名称")
    provider: ProviderType = Field(default=ProviderType.OPENAI)
    model_name: str = Field(description="模型标识符，如 gpt-4o")
    api_key_encrypted: str = Field(default="", description="加密后的 API Key")
    base_url: str = Field(default="https://api.openai.com/v1")
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    status: ModelStatus = Field(default=ModelStatus.INACTIVE)
    is_primary: bool = Field(default=False, description="是否为主模型")
    fallback_priority: int = Field(default=0, ge=0, description="Fallback 优先级，数字越小优先级越高")
    enabled: bool = Field(default=False)


class ModelConfigCreate(BaseModel):
    """创建模型配置请求"""
    name: str
    provider: ProviderType = ProviderType.OPENAI
    model_name: str
    api_key: str = Field(description="明文 API Key，存储时自动加密")
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    is_primary: bool = False
    fallback_priority: int = 0
    enabled: bool = False


class ModelConfigUpdate(BaseModel):
    """更新模型配置请求（所有字段可选）"""
    name: Optional[str] = None
    provider: Optional[ProviderType] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = Field(default=None, description="明文 API Key，传入时重新加密")
    base_url: Optional[str] = None
    timeout_seconds: Optional[float] = None
    max_retries: Optional[int] = None
    is_primary: Optional[bool] = None
    fallback_priority: Optional[int] = None
    enabled: Optional[bool] = None


class ModelConfigResponse(BaseModel):
    """模型配置响应（API Key 脱敏）"""
    id: str
    name: str
    provider: ProviderType
    model_name: str
    api_key_masked: str = Field(description="脱敏后的 API Key")
    base_url: str
    timeout_seconds: float
    max_retries: int
    status: ModelStatus
    is_primary: bool
    fallback_priority: int
    enabled: bool


class SwitchRequest(BaseModel):
    """模型切换请求"""
    target_model_id: str = Field(description="目标模型配置 ID")


class SwitchResult(BaseModel):
    """模型切换结果"""
    success: bool
    previous_model_id: str
    new_model_id: str
    message: str
    duration_ms: float


class FallbackEvent(BaseModel):
    """降级事件记录"""
    timestamp: float = Field(default_factory=time.time)
    from_model_id: str
    to_model_id: str
    reason: str = Field(description="降级原因：timeout / rate_limit / error")
    error_detail: str = ""


class HealthCheckResult(BaseModel):
    """健康检查结果"""
    model_id: str
    status: ModelStatus
    latency_ms: float
    message: str = ""


# ═══════════════════════════════════════════
#  监控指标模型
# ═══════════════════════════════════════════

class ModelMetrics(BaseModel):
    """单模型监控指标"""
    model_id: str
    model_name: str
    call_count: int = 0
    ttft_p95_ms: float = 0.0  # Time To First Token P95
    cache_hit_rate: float = 0.0
    fallback_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    status: ModelStatus = ModelStatus.INACTIVE


class DashboardData(BaseModel):
    """监控看板数据"""
    models: list[ModelMetrics] = Field(default_factory=list)
    total_calls: int = 0
    global_cache_hit_rate: float = 0.0
    total_fallbacks: int = 0
    timestamp: float = Field(default_factory=time.time)


class SSEEvent(BaseModel):
    """SSE 事件"""
    event: str = Field(description="事件类型：model_switch / metrics_update / fallback / error")
    data: dict[str, Any] = Field(default_factory=dict)
