# ======================================================
# GENERADOR PDF PROFESIONAL – NIVEL PRODUCCIÓN
# ======================================================

import os
import io
import json
import tempfile
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER


# ======================================================
# CACHÉ DE GRÁFICOS PLOTLY (OPTIMIZACIÓN)
# ======================================================

@st.cache_data(show_spinner=False)
def generate_plotly_charts_cached(activities_json, metrics_json):
    activities = json.loads(activities_json)
    metrics = json.loads(metrics_json)

    figs = []

    # --- Pie estado ---
    figs.append(go.Figure(data=[go.Pie(
        labels=['Completadas', 'En Progreso', 'Retrasadas', 'En Riesgo'],
        values=[
            metrics['completed_activities'],
            metrics['total_activities'] - metrics['completed_activities'] - metrics['delayed_activities'] - metrics['at_risk_activities'],
            metrics['delayed_activities'],
            metrics['at_risk_activities']
        ]
    )]))

    # --- Presupuesto ---
    figs.append(go.Figure(data=[
        go.Bar(name='Planeado', x=['Presupuesto'], y=[metrics['total_budget']]),
        go.Bar(name='Real', x=['Presupuesto'], y=[metrics['actual_cost']])
    ]))

    # --- Curva S ---
    if activities:
        df = pd.DataFrame(activities).sort_values('start_date')
        df['cumulative_budget'] = df['budget_cost'].cumsum()
        df['cumulative_actual'] = df['actual_cost'].cumsum()

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df['cumulative_budget'], mode='lines', name='Planeado'))
        fig.add_trace(go.Scatter(y=df['cumulative_actual'], mode='lines', name='Real'))
        figs.append(fig)

    return figs


# ======================================================
# FUNCIÓN PRINCIPAL PDF
# ======================================================

def generate_advanced_pdf_report(project_name):
    try:
        pm = st.session_state.project_manager
        if project_name not in pm.projects:
            return None

        project = pm.projects[project_name]
        activities = project['activities']
        metrics = pm.calculate_project_metrics(project_name)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm
        )

        styles = getSampleStyleSheet()
        title = ParagraphStyle(
            'Title', parent=styles['Heading1'],
            alignment=TA_CENTER, fontSize=22, textColor=colors.HexColor('#1a365d')
        )
        subtitle = ParagraphStyle(
            'Sub', parent=styles['Normal'], alignment=TA_CENTER
        )
        section = ParagraphStyle(
            'Section', parent=styles['Heading2'], textColor=colors.HexColor('#2b6cb0')
        )

        story = []

        # --------------------------------------------------
        # PORTADA EJECUTIVA
        # --------------------------------------------------
        story.append(Spacer(1, 30))
        story.append(Paragraph('INFORME EJECUTIVO DE PROYECTO', title))
        story.append(Spacer(1, 10))
        story.append(Paragraph(project_name, title))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", subtitle))
        story.append(Spacer(1, 40))

        # --------------------------------------------------
        # MÉTRICAS
        # --------------------------------------------------
        story.append(Paragraph('Resumen Ejecutivo', section))
        table_data = [
            ['Métrica', 'Valor'],
            ['Progreso', f"{metrics['progress_percentage']:.1f}%"],
            ['Presupuesto', f"€{metrics['total_budget']:,.0f}"],
            ['Costo Real', f"€{metrics['actual_cost']:,.0f}"],
            ['Variación', f"€{metrics['cost_variance']:,.0f}"],
            ['Retraso Promedio', f"{metrics['average_delay_days']:.1f} días"]
        ]
        table = Table(table_data, colWidths=[70*mm, 60*mm])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # --------------------------------------------------
        # GRÁFICOS (PLOTLY → PNG)
        # --------------------------------------------------
        activities_json = json.dumps(activities, sort_keys=True)
        metrics_json = json.dumps(metrics, sort_keys=True)
        figs = generate_plotly_charts_cached(activities_json, metrics_json)

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, fig in enumerate(figs):
                img_path = os.path.join(tmpdir, f'chart_{i}.png')
                fig.write_image(img_path, width=1200, height=800, scale=2)
                story.append(Image(img_path, width=170*mm, height=95*mm))
                story.append(Spacer(1, 15))

            # --------------------------------------------------
            # TABLA ACTIVIDADES
            # --------------------------------------------------
            story.append(Paragraph('Detalle de Actividades', section))
            data = [['Actividad', 'Inicio', 'Fin', 'Progreso', 'Estado']]
            for a in activities:
                data.append([
                    a.get('name', ''), a.get('start_date', ''), a.get('end_date', ''),
                    f"{a.get('progress',0)}%", a.get('status','')
                ])

            t = Table(data, colWidths=[60*mm, 25*mm, 25*mm, 20*mm, 30*mm])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2b6cb0')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTSIZE', (0,0), (-1,-1), 8)
            ]))
            story.append(t)

            doc.build(story)
            buffer.seek(0)

        return buffer.getvalue()

    except Exception as e:
        st.error('Error al generar el informe PDF')
        st.exception(e)
        return None
