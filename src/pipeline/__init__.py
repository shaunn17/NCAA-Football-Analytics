"""
Production Pipeline Package for NCAA Football Analytics Platform

This package provides production-ready data pipeline capabilities including:
- Automated scheduling and execution
- Comprehensive error handling and recovery
- Monitoring and logging
- Configuration management
- Performance optimization
- Data validation and quality checks
- Notifications and alerts
- Incremental updates
- Backup and recovery
- Health checks
"""

from .production_pipeline import ProductionPipeline, PipelineConfig, PipelineStatus

__all__ = ['ProductionPipeline', 'PipelineConfig', 'PipelineStatus']


