"""Tool for querying logs from Loki."""
from __future__ import annotations

import base64
import logging
import os
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar, Optional

import httpx
from pydantic import Field

from sgr_agent_core.base_tool import BaseTool
from sgr_agent_core.agent_config import GlobalConfig

if TYPE_CHECKING:
    from sgr_agent_core.agent_definition import AgentConfig
    from sgr_agent_core.models import AgentContext

logger = logging.getLogger(__name__)


class LokiQueryTool(BaseTool):
    """Query logs from Loki log aggregation system.
    
    Use this tool to:
    - Search for error logs in specific containers
    - Find logs matching patterns or keywords
    - Get logs for a specific time range
    - Investigate incidents by analyzing log entries
    
    Examples:
    - Query all logs: query='{container_name="my-app"}'
    - Query errors: query='{container_name="my-app"} |= "ERROR"'
    - Query with time range: Use start_time and end_time parameters
    - Query specific patterns: query='{job="docker"} |~ "connection.*failed"'
    """
    
    tool_name: ClassVar[str] = "loki_query"
    
    reasoning: str = Field(description="Why this query is needed and what information you're looking for")
    query: str = Field(description="LogQL query string (e.g., '{container_name=\"app\"} |= \"ERROR\"')")
    loki_url: Optional[str] = Field(
        default=None,
        description="Loki API endpoint URL (e.g., http://localhost:3100). If not provided, will use configuration."
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Start time in ISO format (e.g., '2024-01-01T00:00:00Z'). Defaults to 1 hour ago."
    )
    end_time: Optional[str] = Field(
        default=None,
        description="End time in ISO format (e.g., '2024-01-01T01:00:00Z'). Defaults to now."
    )
    limit: int = Field(
        default=100,
        description="Maximum number of log entries to return",
        ge=1,
        le=1000
    )

    def _get_loki_url(self) -> str:
        """Get Loki URL from parameter, environment variable, or config."""
        if self.loki_url:
            return self.loki_url
        
        # Try environment variable
        env_url = os.getenv("SGR__MONITORING__LOKI_URL") or os.getenv("LOKI_URL")
        if env_url:
            return env_url
        
        # Try config (if monitoring section exists)
        try:
            config = GlobalConfig()
            if hasattr(config, 'monitoring') and hasattr(config.monitoring, 'loki_url'):
                return config.monitoring.loki_url
        except Exception:
            pass
        
        # Default fallback
        return "http://localhost:3100"

    async def __call__(self, context: AgentContext, config: AgentConfig, **_) -> str:
        """Execute Loki query and return formatted results."""
        
        loki_url = self._get_loki_url()
        logger.info(f"🔍 Loki query: '{self.query}' (URL: {loki_url})")
        
        try:
            base_url = loki_url.rstrip('/')
            query_url = f"{base_url}/loki/api/v1/query_range"
            
            # Parse time range
            if self.end_time:
                end_time = datetime.fromisoformat(self.end_time.replace('Z', '+00:00'))
            else:
                end_time = datetime.utcnow()
            
            if self.start_time:
                start_time = datetime.fromisoformat(self.start_time.replace('Z', '+00:00'))
            else:
                start_time = end_time - timedelta(hours=1)
            
            # Convert to nanoseconds (Loki uses nanoseconds)
            start_ns = int(start_time.timestamp() * 1e9)
            end_ns = int(end_time.timestamp() * 1e9)
            
            async with httpx.AsyncClient() as client:
                params = {
                    "query": self.query,
                    "start": start_ns,
                    "end": end_ns,
                    "limit": self.limit
                }
                
                response = await client.get(
                    query_url,
                    params=params,
                    timeout=30.0,
                    follow_redirects=True
                )
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") != "success":
                    return f"Error: Loki returned status '{data.get('status')}': {data.get('error', 'Unknown error')}"
                
                result = data.get("data", {}).get("result", [])
                
                if not result:
                    return f"No logs found for query '{self.query}' in time range {start_time.isoformat()} to {end_time.isoformat()}"
                
                # Format results
                formatted_result = f"Loki Query Results:\n"
                formatted_result += f"Query: {self.query}\n"
                formatted_result += f"Time Range: {start_time.isoformat()} to {end_time.isoformat()}\n"
                formatted_result += f"Streams Found: {len(result)}\n\n"
                
                total_entries = 0
                for stream_idx, stream in enumerate(result, 1):
                    labels = stream.get("stream", {})
                    values = stream.get("values", [])
                    total_entries += len(values)
                    
                    formatted_result += f"Stream {stream_idx} (Labels: {labels}):\n"
                    formatted_result += f"  Entries: {len(values)}\n"
                    
                    # Show first few entries
                    for entry_idx, entry in enumerate(values[:10], 1):
                        if len(entry) >= 2:
                            timestamp_ns = entry[0]
                            log_line = entry[1]
                            
                            # Decode base64 if needed
                            try:
                                log_line = base64.b64decode(log_line).decode('utf-8')
                            except Exception:
                                pass  # Already decoded
                            
                            # Convert timestamp to readable format
                            timestamp = datetime.fromtimestamp(int(timestamp_ns) / 1e9)
                            
                            formatted_result += f"  [{timestamp.isoformat()}] {log_line[:200]}\n"
                    
                    if len(values) > 10:
                        formatted_result += f"  ... and {len(values) - 10} more entries\n"
                    
                    formatted_result += "\n"
                
                formatted_result += f"Total log entries: {total_entries}\n"
                
                logger.debug(f"Loki query returned {total_entries} entries")
                return formatted_result
                
        except httpx.HTTPError as e:
            error_msg = f"HTTP error querying Loki: {e}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error querying Loki: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

