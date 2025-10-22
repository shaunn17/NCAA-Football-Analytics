"""
Production Pipeline Monitoring Dashboard

This dashboard provides real-time monitoring of the production pipeline including:
- Pipeline status and health
- Performance metrics
- Error tracking and alerts
- Data quality monitoring
- Historical run statistics
- System resource usage
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import sys
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def get_pipeline_runs(limit: int = 50) -> pd.DataFrame:
    """Get recent pipeline runs from the state database."""
    try:
        state_db_path = Path("data/pipeline_state.db")
        if not state_db_path.exists():
            return pd.DataFrame()
        
        conn = sqlite3.connect(str(state_db_path))
        
        query = """
            SELECT start_time, end_time, status, duration, records_processed, 
                   data_quality_score, errors, warnings
            FROM pipeline_runs 
            ORDER BY start_time DESC 
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        # Convert timestamps
        if not df.empty:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            df['date'] = df['start_time'].dt.date
        
        return df
        
    except Exception as e:
        st.error(f"Failed to load pipeline runs: {e}")
        return pd.DataFrame()

def get_system_metrics() -> dict:
    """Get system metrics."""
    try:
        import shutil
        import psutil
        
        # Disk usage
        disk_usage = shutil.disk_usage(Path("data"))
        disk_free_gb = disk_usage.free / (1024**3)
        disk_total_gb = disk_usage.total / (1024**3)
        disk_used_gb = disk_total_gb - disk_free_gb
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "disk_used_gb": disk_used_gb,
            "disk_total_gb": disk_total_gb,
            "disk_free_gb": disk_free_gb,
            "memory_used_gb": memory_used_gb,
            "memory_total_gb": memory_total_gb,
            "cpu_percent": cpu_percent
        }
        
    except ImportError:
        return {
            "disk_used_gb": 0,
            "disk_total_gb": 0,
            "disk_free_gb": 0,
            "memory_used_gb": 0,
            "memory_total_gb": 0,
            "cpu_percent": 0
        }
    except Exception as e:
        st.error(f"Failed to get system metrics: {e}")
        return {}

def main():
    """Main monitoring dashboard."""
    st.set_page_config(
        page_title="Production Pipeline Monitor",
        page_layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏭 Production Pipeline Monitor")
    st.markdown("---")
    
    # Get data
    runs_df = get_pipeline_runs()
    system_metrics = get_system_metrics()
    
    if runs_df.empty:
        st.warning("No pipeline runs found. Start the production pipeline to see monitoring data.")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Monitoring Controls")
    
    # Time range filter
    if not runs_df.empty:
        min_date = runs_df['date'].min()
        max_date = runs_df['date'].max()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(max_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        if len(date_range) == 2:
            filtered_runs = runs_df[
                (runs_df['date'] >= date_range[0]) & 
                (runs_df['date'] <= date_range[1])
            ]
        else:
            filtered_runs = runs_df
    else:
        filtered_runs = runs_df
    
    # Status filter
    status_options = ["All"] + list(runs_df['status'].unique())
    selected_status = st.sidebar.selectbox("Filter by Status", status_options)
    
    if selected_status != "All":
        filtered_runs = filtered_runs[filtered_runs['status'] == selected_status]
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        total_runs = len(filtered_runs)
        st.metric("Total Runs", total_runs)
    
    with col2:
        successful_runs = len(filtered_runs[filtered_runs['status'] == 'completed'])
        success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_duration = filtered_runs['duration'].mean() if not filtered_runs.empty else 0
        st.metric("Avg Duration", f"{avg_duration:.1f}s")
    
    with col4:
        total_records = filtered_runs['records_processed'].sum() if not filtered_runs.empty else 0
        st.metric("Total Records", f"{total_records:,}")
    
    st.markdown("---")
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Pipeline Runs Over Time")
        
        if not filtered_runs.empty:
            # Status over time
            status_counts = filtered_runs.groupby(['date', 'status']).size().unstack(fill_value=0)
            
            fig = go.Figure()
            for status in status_counts.columns:
                fig.add_trace(go.Scatter(
                    x=status_counts.index,
                    y=status_counts[status],
                    mode='lines+markers',
                    name=status.title(),
                    line=dict(width=3)
                ))
            
            fig.update_layout(
                title="Pipeline Runs by Status",
                xaxis_title="Date",
                yaxis_title="Number of Runs",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    with col2:
        st.subheader("⏱️ Pipeline Duration Trends")
        
        if not filtered_runs.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_runs['start_time'],
                y=filtered_runs['duration'],
                mode='lines+markers',
                name='Duration',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Pipeline Duration Over Time",
                xaxis_title="Run Time",
                yaxis_title="Duration (seconds)",
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    # Data quality section
    st.subheader("📊 Data Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if not filtered_runs.empty:
            quality_scores = filtered_runs['data_quality_score'].dropna()
            if not quality_scores.empty:
                avg_quality = quality_scores.mean()
                st.metric("Average Quality Score", f"{avg_quality:.3f}")
                
                # Quality score distribution
                fig = go.Figure(data=[go.Histogram(x=quality_scores, nbinsx=20)])
                fig.update_layout(
                    title="Data Quality Score Distribution",
                    xaxis_title="Quality Score",
                    yaxis_title="Frequency",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No quality scores available")
    
    with col2:
        if not filtered_runs.empty:
            records_processed = filtered_runs['records_processed'].dropna()
            if not records_processed.empty:
                avg_records = records_processed.mean()
                st.metric("Average Records Processed", f"{avg_records:.0f}")
                
                # Records processed over time
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=filtered_runs['start_time'],
                    y=filtered_runs['records_processed'],
                    mode='lines+markers',
                    name='Records Processed',
                    line=dict(color='green', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title="Records Processed Over Time",
                    xaxis_title="Run Time",
                    yaxis_title="Records Processed",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No records data available")
    
    # System metrics section
    st.subheader("💻 System Metrics")
    
    if system_metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            disk_usage_pct = (system_metrics['disk_used_gb'] / system_metrics['disk_total_gb'] * 100) if system_metrics['disk_total_gb'] > 0 else 0
            st.metric(
                "Disk Usage", 
                f"{system_metrics['disk_used_gb']:.1f}GB / {system_metrics['disk_total_gb']:.1f}GB",
                delta=f"{disk_usage_pct:.1f}%"
            )
        
        with col2:
            memory_usage_pct = (system_metrics['memory_used_gb'] / system_metrics['memory_total_gb'] * 100) if system_metrics['memory_total_gb'] > 0 else 0
            st.metric(
                "Memory Usage",
                f"{system_metrics['memory_used_gb']:.1f}GB / {system_metrics['memory_total_gb']:.1f}GB",
                delta=f"{memory_usage_pct:.1f}%"
            )
        
        with col3:
            st.metric("CPU Usage", f"{system_metrics['cpu_percent']:.1f}%")
    
    # Recent runs table
    st.subheader("📋 Recent Pipeline Runs")
    
    if not filtered_runs.empty:
        # Prepare table data
        display_runs = filtered_runs.head(20).copy()
        display_runs['start_time'] = display_runs['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_runs['end_time'] = display_runs['end_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_runs['duration'] = display_runs['duration'].round(1)
        display_runs['data_quality_score'] = display_runs['data_quality_score'].round(3)
        
        # Select columns to display
        display_columns = [
            'start_time', 'status', 'duration', 'records_processed', 
            'data_quality_score'
        ]
        
        st.dataframe(
            display_runs[display_columns],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No runs found for the selected filters")
    
    # Error and warning summary
    if not filtered_runs.empty:
        st.subheader("⚠️ Errors and Warnings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Count runs with errors
            runs_with_errors = filtered_runs[filtered_runs['errors'].notna() & (filtered_runs['errors'] != '[]')]
            st.metric("Runs with Errors", len(runs_with_errors))
        
        with col2:
            # Count runs with warnings
            runs_with_warnings = filtered_runs[filtered_runs['warnings'].notna() & (filtered_runs['warnings'] != '[]')]
            st.metric("Runs with Warnings", len(runs_with_warnings))
        
        # Show recent errors
        if len(runs_with_errors) > 0:
            st.subheader("🔴 Recent Errors")
            for _, run in runs_with_errors.head(5).iterrows():
                with st.expander(f"Error on {run['start_time'].strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.code(run['errors'])
        
        # Show recent warnings
        if len(runs_with_warnings) > 0:
            st.subheader("🟡 Recent Warnings")
            for _, run in runs_with_warnings.head(5).iterrows():
                with st.expander(f"Warning on {run['start_time'].strftime('%Y-%m-%d %H:%M:%S')}"):
                    st.code(run['warnings'])

if __name__ == "__main__":
    main()


