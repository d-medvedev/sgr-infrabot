"""Tool for querying metrics from Victoria Metrics."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Optional

import httpx
from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from sgr_agent_core.agent_config import GlobalConfig

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext

logger = logging.getLogger(__name__)


class VictoriaMetricsQueryTool(BaseTool):
    """Query metrics from Victoria Metrics time-series database.
    
    Use this tool to:
    - Get current metric values
    - Query metric history over time ranges
    - Analyze resource usage (CPU, memory, disk, GPU)
    - Investigate performance issues
    - Compare metrics across different time periods
    
    Examples:
    - Instant query: query='infra_monitor_cpu_usage_percent{monitor_id="1"}'
    - Range query: Use start_time, end_time, and step parameters
    - Aggregations: query='avg(infra_monitor_cpu_usage_percent)'
    - Rate calculations: query='rate(infra_monitor_cpu_usage_percent[5m])'
    """
    
    tool_name: ClassVar[str] = "victoria_metrics_query"
    
    reasoning: str = Field(description="Why this query is needed and what metrics you're investigating")
    query: str = Field(description="MetricsQL/PromQL query string")
    vm_url: Optional[str] = Field(
        default=None,
        description="Victoria Metrics API endpoint URL (e.g., http://localhost:8428). If not provided, will use configuration."
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Start time in ISO format for range queries. If not provided, performs instant query."
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End time in ISO format for range queries. Defaults to now if start_time is provided."
    )
    step: str = Field(
        default="1m",
        description="Query resolution step for range queries (e.g., '1m', '5m', '1h')"
    )

    def _get_vm_url(self) -> str:
        """Get Victoria Metrics URL from parameter, environment variable, or config."""
        if self.vm_url:
            return self.vm_url
        
        # Try environment variable
        env_url = os.getenv("SGR__MONITORING__VICTORIA_METRICS_URL") or os.getenv("VICTORIA_METRICS_URL") or os.getenv("VM_URL")
        if env_url:
            return env_url
        
        # Try config (if monitoring section exists)
        try:
            config = GlobalConfig()
            if hasattr(config, 'monitoring') and hasattr(config.monitoring, 'victoria_metrics_url'):
                return config.monitoring.victoria_metrics_url
        except Exception:
            pass
        
        # Default fallback
        return "http://localhost:8428"

    async def __call__(self, context: AgentContext, config: AgentConfig, **_) -> str:
        """Execute Victoria Metrics query and return formatted results."""
        
        vm_url = self._get_vm_url()
        logger.info(f"📊 Victoria Metrics query: '{self.query}' (URL: {vm_url})")
        
        try:
            base_url = vm_url.rstrip('/')
            
            # Determine if this is a range query or instant query
            is_range_query = self.start_time is not None
            
            if is_range_query:
                query_url = f"{base_url}/api/v1/query_range"
                
                # Parse time range
                start_time = datetime.fromisoformat(self.start_time.replace('Z', '+00:00'))
                if self.end_time:
                    end_time = datetime.fromisoformat(self.end_time.replace('Z', '+00:00'))
                else:
                    end_time = datetime.utcnow()
                
                params = {
                    "query": self.query,
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "step": self.step
                }
            else:
                query_url = f"{base_url}/api/v1/query"
                params = {"query": self.query}
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    query_url,
                    params=params,
                    timeout=30.0,
                    follow_redirects=True
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "success":
                    return f"Error: Victoria Metrics returned status '{data.get('status')}': {data.get('error', 'Unknown error')}"
                
                result = data.get("data", {}).get("result", [])
                
                if not result:
                    return f"No metrics found for query '{self.query}'"
                
                # Format results
                if is_range_query:
                    formatted_result = f"Victoria Metrics Range Query Results:\n"
                    formatted_result += f"Query: {self.query}\n"
                    formatted_result += f"Time Range: {start_time.isoformat()} to {end_time.isoformat()}\n"
                    formatted_result += f"Step: {self.step}\n"
                    formatted_result += f"Time Series: {len(result)}\n\n"
                    
                    for series_idx, series in enumerate(result, 1):
                        metric = series.get("metric", {})
                        values = series.get("values", [])
                        
                        formatted_result += f"Series {series_idx}:\n"
                        formatted_result += f"  Labels: {metric}\n"
                        formatted_result += f"  Data Points: {len(values)}\n"
                        
                        if values:
                            # Show first and last few values
                            formatted_result += "  Sample Values:\n"
                            for timestamp, value in values[:5]:
                                dt = datetime.fromtimestamp(timestamp)
                                formatted_result += f"    [{dt.isoformat()}] {value}\n"
                            
                            if len(values) > 5:
                                formatted_result += f"    ... ({len(values) - 5} more data points)\n"
                                
                                # Show last value
                                last_timestamp, last_value = values[-1]
                                last_dt = datetime.fromtimestamp(last_timestamp)
                                formatted_result += f"    [{last_dt.isoformat()}] {last_value} (latest)\n"
                            
                            # Calculate statistics
                            numeric_values = [float(v[1]) for v in values if v[1] != 'NaN']
                            if numeric_values:
                                avg = sum(numeric_values) / len(numeric_values)
                                min_val = min(numeric_values)
                                max_val = max(numeric_values)
                                formatted_result += f"  Statistics: min={min_val:.2f}, avg={avg:.2f}, max={max_val:.2f}\n"
                        
                        formatted_result += "\n"
                else:
                    formatted_result = f"Victoria Metrics Instant Query Results:\n"
                    formatted_result += f"Query: {self.query}\n"
                    formatted_result += f"Results: {len(result)}\n\n"
                    
                    for series_idx, series in enumerate(result, 1):
                        metric = series.get("metric", {})
                        value = series.get("value", [])
                        
                        formatted_result += f"Result {series_idx}:\n"
                        formatted_result += f"  Labels: {metric}\n"
                        
                        if value:
                            timestamp, metric_value = value
                            dt = datetime.fromtimestamp(timestamp)
                            formatted_result += f"  Value: {metric_value} (at {dt.isoformat()})\n"
                        
                        formatted_result += "\n"
                
                logger.debug(f"Victoria Metrics query returned {len(result)} series")
                return formatted_result
                
        except httpx.HTTPError as e:
            error_msg = f"HTTP error querying Victoria Metrics: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error querying Victoria Metrics: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

