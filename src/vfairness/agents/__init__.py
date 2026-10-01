"""
vfairness.agents: AI Agent Fairness Testing Module

Tools for measuring bias in AI agents including:
- Correspondence testing (paired artifacts)
- Tool selection bias auditing
- RAG retrieval bias detection
- Multi-stage pipeline tracking
- Temporal trajectory analysis
- Action & delegation bias measurement
"""

from .action_bias import ActionBiasAnalyzer, ActionBiasResult
from .correspondence import CorrespondenceResult, CorrespondenceTester
from .pipeline_tracker import PipelineTracker, StageResult
from .rag_bias import RAGBiasAnalyzer, RAGBiasResult
from .temporal import TemporalTracker, TrajectoryResult
from .tool_bias import ToolBiasAuditor, ToolBiasResult

__all__ = [
    "CorrespondenceTester",
    "CorrespondenceResult",
    "ToolBiasAuditor",
    "ToolBiasResult",
    "RAGBiasAnalyzer",
    "RAGBiasResult",
    "PipelineTracker",
    "StageResult",
    "TemporalTracker",
    "TrajectoryResult",
    "ActionBiasAnalyzer",
    "ActionBiasResult",
]
