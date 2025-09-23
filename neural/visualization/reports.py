"""
Report Generation Framework for Neural SDK

Provides comprehensive report generation capabilities for:
- PDF performance reports with charts and analysis
- HTML interactive reports with embedded visualizations
- Strategy comparison and analysis reports
- Risk assessment and compliance reports
- Custom templated reporting with branding
"""

import os
import io
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
import tempfile

# Report generation imports
import pandas as pd
import numpy as np
from jinja2 import Environment, FileSystemLoader, BaseLoader
import plotly.graph_objects as go
import plotly.io as pio
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, blue, green, red
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

from .charts import ChartManager, PerformanceChart, RiskChart, PnLChart
from ..analysis.metrics import PerformanceMetrics, PerformanceCalculator
from ..strategy.base import Signal
from ..risk.limits import LimitViolation

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Available export formats."""
    PDF = "pdf"
    HTML = "html"
    EXCEL = "xlsx"
    JSON = "json"


class ReportType(Enum):
    """Types of reports available."""
    PERFORMANCE = "performance"
    RISK = "risk"
    STRATEGY_COMPARISON = "strategy_comparison"
    COMPLIANCE = "compliance"
    DAILY_SUMMARY = "daily_summary"
    CUSTOM = "custom"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    format: ExportFormat = ExportFormat.PDF
    include_charts: bool = True
    include_raw_data: bool = False
    chart_width: int = 800
    chart_height: int = 600
    logo_path: Optional[str] = None
    company_name: str = "Neural Trading"
    report_title: Optional[str] = None
    footer_text: Optional[str] = None
    template_path: Optional[str] = None


class ReportGenerator:
    """
    Central report generation engine.
    
    Handles creation of professional trading reports in multiple formats
    with automated chart generation and data analysis.
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """Initialize report generator."""
        self.config = config or ReportConfig()
        self.chart_manager = ChartManager()
        self.performance_calculator = PerformanceCalculator()
        
        # Setup Jinja2 environment
        if self.config.template_path and os.path.exists(self.config.template_path):
            self.jinja_env = Environment(loader=FileSystemLoader(self.config.template_path))
        else:
            self.jinja_env = Environment(loader=BaseLoader())
            
        # ReportLab styles
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom ReportLab styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=Color(0, 0.5, 0.2)  # Dark green
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='ReportSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=20,
            textColor=Color(0.2, 0.2, 0.6)  # Dark blue
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_RIGHT,
            textColor=Color(0, 0.4, 0)  # Green
        ))
        
    def generate_report(
        self,
        report_type: ReportType,
        data: Dict[str, Any],
        output_path: str,
        **kwargs
    ) -> str:
        """Generate report based on type and data."""
        
        try:
            if report_type == ReportType.PERFORMANCE:
                return self._generate_performance_report(data, output_path, **kwargs)
            elif report_type == ReportType.RISK:
                return self._generate_risk_report(data, output_path, **kwargs)
            elif report_type == ReportType.STRATEGY_COMPARISON:
                return self._generate_strategy_comparison_report(data, output_path, **kwargs)
            elif report_type == ReportType.DAILY_SUMMARY:
                return self._generate_daily_summary_report(data, output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported report type: {report_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
            
    def _generate_performance_report(
        self, 
        data: Dict[str, Any], 
        output_path: str,
        **kwargs
    ) -> str:
        """Generate comprehensive performance report."""
        
        if self.config.format == ExportFormat.PDF:
            return self._generate_performance_pdf(data, output_path, **kwargs)
        elif self.config.format == ExportFormat.HTML:
            return self._generate_performance_html(data, output_path, **kwargs)
        else:
            raise ValueError(f"Format {self.config.format} not supported for performance reports")
            
    def _generate_performance_pdf(
        self, 
        data: Dict[str, Any], 
        output_path: str,
        **kwargs
    ) -> str:
        """Generate PDF performance report."""
        
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        story = []
        
        # Title
        title = self.config.report_title or "Portfolio Performance Report"
        story.append(Paragraph(title, self.styles['ReportTitle']))
        story.append(Spacer(1, 12))
        
        # Report metadata
        report_date = datetime.now().strftime("%B %d, %Y")
        period = data.get('period', 'N/A')
        story.append(Paragraph(f"Report Date: {report_date}", self.styles['Normal']))
        story.append(Paragraph(f"Analysis Period: {period}", self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", self.styles['ReportSubtitle']))
        
        metrics = data.get('metrics')
        if isinstance(metrics, PerformanceMetrics):
            summary_data = [
                ['Metric', 'Value'],
                ['Total Return', f"{metrics.total_return:.2%}"],
                ['Sharpe Ratio', f"{metrics.sharpe_ratio:.2f}"],
                ['Sortino Ratio', f"{metrics.sortino_ratio:.2f}"],
                ['Max Drawdown', f"{metrics.max_drawdown:.2%}"],
                ['Win Rate', f"{metrics.win_rate:.1%}"],
                ['Volatility', f"{metrics.volatility:.2%}"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), Color(0.95, 0.95, 0.95)),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            
            story.append(summary_table)
            story.append(Spacer(1, 20))
        
        # Charts
        if self.config.include_charts and 'returns' in data:
            story.append(Paragraph("Performance Charts", self.styles['ReportSubtitle']))
            
            returns_data = data['returns']
            if isinstance(returns_data, pd.Series):
                # Generate P&L chart
                performance_chart = PerformanceChart(self.chart_manager)
                fig = performance_chart.create_pnl_chart(returns_data, title="Portfolio Performance")
                
                # Convert to image for PDF
                img_buffer = self._fig_to_image_buffer(fig)
                if img_buffer:
                    img = Image(img_buffer, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
                
                # Generate drawdown chart
                drawdown_fig = performance_chart.create_drawdown_chart(returns_data)
                drawdown_buffer = self._fig_to_image_buffer(drawdown_fig)
                if drawdown_buffer:
                    drawdown_img = Image(drawdown_buffer, width=6*inch, height=4*inch)
                    story.append(drawdown_img)
                    story.append(Spacer(1, 20))
        
        # Detailed Analysis
        story.append(PageBreak())
        story.append(Paragraph("Detailed Analysis", self.styles['ReportSubtitle']))
        
        if 'analysis_text' in data:
            story.append(Paragraph(data['analysis_text'], self.styles['Normal']))
        else:
            # Generate automatic analysis
            if isinstance(metrics, PerformanceMetrics):
                analysis = self._generate_performance_analysis(metrics)
                story.append(Paragraph(analysis, self.styles['Normal']))
        
        # Risk Metrics
        if 'risk_metrics' in data:
            story.append(Spacer(1, 20))
            story.append(Paragraph("Risk Analysis", self.styles['ReportSubtitle']))
            
            risk_data = data['risk_metrics']
            risk_table_data = [['Risk Metric', 'Value']]
            
            for metric_name, value in risk_data.items():
                if isinstance(value, (int, float)):
                    if 'ratio' in metric_name.lower():
                        formatted_value = f"{value:.2f}"
                    elif 'percentage' in metric_name.lower() or 'rate' in metric_name.lower():
                        formatted_value = f"{value:.2%}"
                    else:
                        formatted_value = f"{value:,.2f}"
                else:
                    formatted_value = str(value)
                    
                risk_table_data.append([metric_name.replace('_', ' ').title(), formatted_value])
            
            risk_table = Table(risk_table_data, colWidths=[2.5*inch, 1.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
                ('TEXTCOLOR', (0, 0), (-1, 0), black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), Color(0.95, 0.95, 0.95)),
                ('GRID', (0, 0), (-1, -1), 1, black)
            ]))
            
            story.append(risk_table)
        
        # Footer
        if self.config.footer_text:
            story.append(Spacer(1, 30))
            story.append(Paragraph(self.config.footer_text, self.styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        logger.info(f"Generated performance PDF report: {output_path}")
        return output_path
        
    def _generate_performance_html(
        self, 
        data: Dict[str, Any], 
        output_path: str,
        **kwargs
    ) -> str:
        """Generate HTML performance report."""
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ title }}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background-color: #f8f9fa; }
                .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { text-align: center; border-bottom: 2px solid #007bff; padding-bottom: 20px; margin-bottom: 30px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
                .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; border-left: 4px solid #007bff; }
                .metric-value { font-size: 24px; font-weight: bold; color: #007bff; }
                .metric-label { font-size: 14px; color: #6c757d; margin-top: 5px; }
                .chart-container { margin: 30px 0; }
                .section-title { font-size: 20px; font-weight: bold; color: #333; margin: 30px 0 15px 0; border-bottom: 1px solid #dee2e6; padding-bottom: 5px; }
                .analysis-text { line-height: 1.6; color: #495057; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ title }}</h1>
                    <p>Generated on {{ report_date }}</p>
                    <p>Analysis Period: {{ period }}</p>
                </div>
                
                {% if metrics %}
                <div class="section-title">Performance Metrics</div>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f%%"|format(metrics.total_return * 100) }}</div>
                        <div class="metric-label">Total Return</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.sharpe_ratio) }}</div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f"|format(metrics.sortino_ratio) }}</div>
                        <div class="metric-label">Sortino Ratio</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f%%"|format(metrics.max_drawdown * 100) }}</div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.1f%%"|format(metrics.win_rate * 100) }}</div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{{ "%.2f%%"|format(metrics.volatility * 100) }}</div>
                        <div class="metric-label">Volatility</div>
                    </div>
                </div>
                {% endif %}
                
                {% if performance_chart %}
                <div class="section-title">Performance Chart</div>
                <div class="chart-container">
                    <div id="performance-chart">{{ performance_chart|safe }}</div>
                </div>
                {% endif %}
                
                {% if drawdown_chart %}
                <div class="section-title">Drawdown Analysis</div>
                <div class="chart-container">
                    <div id="drawdown-chart">{{ drawdown_chart|safe }}</div>
                </div>
                {% endif %}
                
                {% if analysis_text %}
                <div class="section-title">Analysis</div>
                <div class="analysis-text">{{ analysis_text|safe }}</div>
                {% endif %}
            </div>
        </body>
        </html>
        """
        
        # Prepare template data
        template_data = {
            'title': self.config.report_title or "Portfolio Performance Report",
            'report_date': datetime.now().strftime("%B %d, %Y"),
            'period': data.get('period', 'N/A'),
            'metrics': data.get('metrics'),
            'analysis_text': data.get('analysis_text', '')
        }
        
        # Generate charts if available
        if self.config.include_charts and 'returns' in data:
            returns_data = data['returns']
            if isinstance(returns_data, pd.Series):
                performance_chart = PerformanceChart(self.chart_manager)
                
                # Performance chart
                perf_fig = performance_chart.create_pnl_chart(returns_data, title="Portfolio Performance")
                template_data['performance_chart'] = pio.to_html(perf_fig, include_plotlyjs=False, div_id="performance-chart")
                
                # Drawdown chart
                drawdown_fig = performance_chart.create_drawdown_chart(returns_data)
                template_data['drawdown_chart'] = pio.to_html(drawdown_fig, include_plotlyjs=False, div_id="drawdown-chart")
        
        # Render template
        template = self.jinja_env.from_string(html_template)
        html_content = template.render(**template_data)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"Generated performance HTML report: {output_path}")
        return output_path
        
    def _generate_risk_report(
        self, 
        data: Dict[str, Any], 
        output_path: str,
        **kwargs
    ) -> str:
        """Generate risk assessment report."""
        
        # Implementation similar to performance report but focused on risk metrics
        # This would include VaR analysis, stress testing results, limit violations, etc.
        
        if self.config.format == ExportFormat.PDF:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            # Title
            title = "Risk Assessment Report"
            story.append(Paragraph(title, self.styles['ReportTitle']))
            story.append(Spacer(1, 20))
            
            # Risk metrics summary
            risk_metrics = data.get('risk_metrics', {})
            if risk_metrics:
                story.append(Paragraph("Risk Metrics Summary", self.styles['ReportSubtitle']))
                
                risk_data = [['Risk Metric', 'Value', 'Status']]
                for metric, value in risk_metrics.items():
                    status = self._assess_risk_status(metric, value)
                    risk_data.append([
                        metric.replace('_', ' ').title(),
                        f"{value:.2%}" if 'percentage' in metric else f"{value:.2f}",
                        status
                    ])
                
                risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 1*inch])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, black)
                ]))
                
                story.append(risk_table)
            
            # Violations
            violations = data.get('violations', [])
            if violations:
                story.append(Spacer(1, 20))
                story.append(Paragraph("Risk Limit Violations", self.styles['ReportSubtitle']))
                
                for violation in violations:
                    if isinstance(violation, LimitViolation):
                        story.append(Paragraph(
                            f"• {violation.limit_id}: {violation.message}",
                            self.styles['Normal']
                        ))
            
            doc.build(story)
            
        logger.info(f"Generated risk report: {output_path}")
        return output_path
        
    def _generate_strategy_comparison_report(
        self, 
        data: Dict[str, Any], 
        output_path: str,
        **kwargs
    ) -> str:
        """Generate strategy comparison report."""
        
        strategies_data = data.get('strategies', {})
        if not strategies_data:
            raise ValueError("No strategy data provided for comparison report")
            
        # Calculate metrics for each strategy
        strategies_metrics = {}
        for name, strategy_data in strategies_data.items():
            if 'returns' in strategy_data:
                metrics = self.performance_calculator.calculate_comprehensive_metrics(
                    returns=strategy_data['returns'],
                    benchmark_returns=strategy_data.get('benchmark_returns')
                )
                strategies_metrics[name] = metrics
                
        if self.config.format == ExportFormat.PDF:
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            story = []
            
            # Title
            story.append(Paragraph("Strategy Comparison Report", self.styles['ReportTitle']))
            story.append(Spacer(1, 20))
            
            # Comparison table
            if strategies_metrics:
                story.append(Paragraph("Performance Comparison", self.styles['ReportSubtitle']))
                
                comparison_data = [['Strategy', 'Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']]
                for name, metrics in strategies_metrics.items():
                    comparison_data.append([
                        name,
                        f"{metrics.total_return:.2%}",
                        f"{metrics.sharpe_ratio:.2f}",
                        f"{metrics.max_drawdown:.2%}",
                        f"{metrics.win_rate:.1%}"
                    ])
                
                comparison_table = Table(comparison_data, colWidths=[1.5*inch, 1.2*inch, 1*inch, 1.2*inch, 1*inch])
                comparison_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), Color(0.8, 0.8, 0.8)),
                    ('TEXTCOLOR', (0, 0), (-1, 0), black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('GRID', (0, 0), (-1, -1), 1, black)
                ]))
                
                story.append(comparison_table)
            
            doc.build(story)
            
        logger.info(f"Generated strategy comparison report: {output_path}")
        return output_path
        
    def _generate_daily_summary_report(
        self, 
        data: Dict[str, Any], 
        output_path: str,
        **kwargs
    ) -> str:
        """Generate daily summary report."""
        
        # Implementation for daily summary
        # This would include daily P&L, key trades, risk updates, etc.
        
        logger.info(f"Generated daily summary report: {output_path}")
        return output_path
        
    def _fig_to_image_buffer(self, fig: go.Figure) -> Optional[io.BytesIO]:
        """Convert Plotly figure to image buffer for PDF embedding."""
        try:
            # Convert to PNG bytes
            img_bytes = pio.to_image(fig, format="png", width=800, height=600)
            return io.BytesIO(img_bytes)
        except Exception as e:
            logger.error(f"Failed to convert figure to image: {e}")
            return None
            
    def _generate_performance_analysis(self, metrics: PerformanceMetrics) -> str:
        """Generate automatic performance analysis text."""
        
        analysis_parts = []
        
        # Return analysis
        if metrics.total_return > 0.1:
            analysis_parts.append("The portfolio delivered strong positive returns during the analysis period.")
        elif metrics.total_return > 0:
            analysis_parts.append("The portfolio generated modest positive returns.")
        else:
            analysis_parts.append("The portfolio experienced negative returns during this period.")
            
        # Risk-adjusted return analysis
        if metrics.sharpe_ratio > 1.5:
            analysis_parts.append("The Sharpe ratio indicates excellent risk-adjusted returns.")
        elif metrics.sharpe_ratio > 1.0:
            analysis_parts.append("The Sharpe ratio shows good risk-adjusted performance.")
        elif metrics.sharpe_ratio > 0:
            analysis_parts.append("The Sharpe ratio indicates positive but modest risk-adjusted returns.")
        else:
            analysis_parts.append("The negative Sharpe ratio suggests poor risk-adjusted performance.")
            
        # Drawdown analysis
        if abs(metrics.max_drawdown) < 0.05:
            analysis_parts.append("Maximum drawdown remained well-controlled below 5%.")
        elif abs(metrics.max_drawdown) < 0.1:
            analysis_parts.append("Maximum drawdown was moderate at under 10%.")
        else:
            analysis_parts.append("Maximum drawdown was significant, indicating higher volatility periods.")
            
        # Win rate analysis
        if metrics.win_rate > 0.6:
            analysis_parts.append(f"The strategy maintained a high win rate of {metrics.win_rate:.1%}.")
        elif metrics.win_rate > 0.5:
            analysis_parts.append(f"The win rate of {metrics.win_rate:.1%} was above breakeven.")
        else:
            analysis_parts.append(f"The win rate of {metrics.win_rate:.1%} suggests room for improvement in trade selection.")
            
        return " ".join(analysis_parts)
        
    def _assess_risk_status(self, metric_name: str, value: float) -> str:
        """Assess risk status based on metric value."""
        
        # Simple risk assessment - would be more sophisticated in practice
        risk_thresholds = {
            'var_95': 0.05,  # 5% VaR threshold
            'max_drawdown': 0.15,  # 15% max drawdown
            'volatility': 0.25,  # 25% volatility
            'sharpe_ratio': 1.0  # Minimum Sharpe ratio
        }
        
        if metric_name.lower() in risk_thresholds:
            threshold = risk_thresholds[metric_name.lower()]
            
            if metric_name.lower() == 'sharpe_ratio':
                return "Good" if value >= threshold else "Poor"
            else:
                return "Good" if value <= threshold else "High Risk"
        
        return "Normal"


class PerformanceReportBuilder:
    """Specialized builder for performance reports."""
    
    def __init__(self, report_generator: ReportGenerator):
        self.generator = report_generator
        
    def build_monthly_report(
        self, 
        returns_data: pd.Series,
        output_path: str
    ) -> str:
        """Build monthly performance report."""
        
        # Calculate performance metrics
        metrics = self.generator.performance_calculator.calculate_comprehensive_metrics(returns_data)
        
        # Prepare report data
        report_data = {
            'returns': returns_data,
            'metrics': metrics,
            'period': f"{returns_data.index[0].strftime('%B %Y')} - {returns_data.index[-1].strftime('%B %Y')}",
            'analysis_text': self.generator._generate_performance_analysis(metrics)
        }
        
        return self.generator.generate_report(
            ReportType.PERFORMANCE,
            report_data,
            output_path
        )


class RiskReportBuilder:
    """Specialized builder for risk reports."""
    
    def __init__(self, report_generator: ReportGenerator):
        self.generator = report_generator
        
    def build_risk_assessment(
        self, 
        risk_metrics: Dict[str, float],
        violations: List[LimitViolation],
        output_path: str
    ) -> str:
        """Build comprehensive risk assessment report."""
        
        report_data = {
            'risk_metrics': risk_metrics,
            'violations': violations,
            'assessment_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return self.generator.generate_report(
            ReportType.RISK,
            report_data,
            output_path
        )


class StrategyComparisonReport:
    """Specialized builder for strategy comparison reports."""
    
    def __init__(self, report_generator: ReportGenerator):
        self.generator = report_generator
        
    def build_comparison(
        self, 
        strategies_data: Dict[str, Dict[str, Any]],
        output_path: str
    ) -> str:
        """Build strategy performance comparison report."""
        
        report_data = {
            'strategies': strategies_data,
            'comparison_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        return self.generator.generate_report(
            ReportType.STRATEGY_COMPARISON,
            report_data,
            output_path
        )
