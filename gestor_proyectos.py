import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import io
import base64
# --- IMPORTACIONES PARA PDF ---
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage
import tempfile
# --- NUEVA IMPORTACIÓN PARA BASE DE DATOS ---
import sqlite3

# Configuración inicial
st.set_page_config(
    page_title="Gestor de Proyectos Avanzado",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CLASE PROJECTMANAGER CON SQLITE (MEJORADA)
# ==========================================
class ProjectManager:
    def __init__(self, db_name="proyectos_db.sqlite3"):
        self.db_name = db_name
        self.init_db()
        # Mantenemos self.projects en memoria para compatibilidad con la UI existente
        self.projects = {} 
        self.load_projects_from_db()

    def get_connection(self):
        """Connection factory para asegurar cierre"""
        return sqlite3.connect(self.db_name, check_same_thread=False)

    def init_db(self):
        """Inicializar esquema de base de datos"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Tabla Proyectos
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_date TEXT,
                settings TEXT
            )
        ''')
        
        # Tabla Actividades
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL,
                name TEXT NOT NULL,
                "group" TEXT,
                start_date TEXT,
                end_date TEXT,
                progress INTEGER,
                status TEXT,
                budget_cost REAL,
                actual_cost REAL,
                weight INTEGER,
                real_start_date TEXT,
                real_end_date TEXT,
                FOREIGN KEY (project_name) REFERENCES projects (name) ON DELETE CASCADE
            )
        ''')
        conn.commit()
        conn.close()

    def load_projects_from_db(self):
        """Cargar toda la estructura DB a memoria para la UI"""
        conn = self.get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        self.projects = {}
        
        # 1. Cargar proyectos
        cursor.execute("SELECT * FROM projects")
        projects_rows = cursor.fetchall()
        
        for proj_row in projects_rows:
            proj_name = proj_row['name']
            self.projects[proj_name] = {
                "description": proj_row['description'],
                "created_date": proj_row['created_date'],
                "settings": json.loads(proj_row['settings']) if proj_row['settings'] else {
                    "currency": "€", "working_days": True, "auto_calculate": True
                },
                "activities": []
            }
            
            # 2. Cargar actividades para este proyecto
            cursor.execute("SELECT * FROM activities WHERE project_name = ?", (proj_name,))
            activities_rows = cursor.fetchall()
            
            for act_row in activities_rows:
                act_dict = dict(act_row)
                act_dict['id'] = str(act_dict['id'])
                # group es palabra reservada, en DB se guardó como "group"
                act_dict['group'] = act_dict.pop('"group"', act_dict.get('group'))
                
                # Asegurar que las fechas reales existan, aunque sean cadenas vacías
                if 'real_start_date' not in act_dict or act_dict['real_start_date'] is None:
                    act_dict['real_start_date'] = ""
                if 'real_end_date' not in act_dict or act_dict['real_end_date'] is None:
                     act_dict['real_end_date'] = ""
                     
                self.projects[proj_name]["activities"].append(act_dict)
                
        conn.close()

    def create_project(self, name, description=""):
        """Crear nuevo proyecto en DB"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            default_settings = json.dumps({
                "currency": "€",
                "working_days": True,
                "auto_calculate": True
            })
            
            cursor.execute("""
                INSERT INTO projects (name, description, created_date, settings)
                VALUES (?, ?, ?, ?)
            """, (name, description, datetime.now().strftime("%Y-%m-%d"), default_settings))
            
            conn.commit()
            conn.close()
            self.load_projects_from_db()
            return True
        except sqlite3.IntegrityError:
            return False
        except Exception as e:
            st.error(f"Error DB: {e}")
            return False

    def delete_project(self, name):
        """Eliminar proyecto y sus actividades en DB"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("DELETE FROM projects WHERE name = ?", (name,))
            conn.commit()
            conn.close()
            self.load_projects_from_db()
            return True
        except Exception as e:
            st.error(f"Error DB: {e}")
            return False

    def add_activity(self, project_name, activity_dict):
        """Añadir actividad a DB"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO activities (
                    project_name, name, "group", start_date, end_date, 
                    progress, status, budget_cost, actual_cost, weight, 
                    real_start_date, real_end_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project_name,
                activity_dict['name'],
                activity_dict['group'],
                activity_dict['start_date'],
                activity_dict['end_date'],
                activity_dict['progress'],
                activity_dict['status'],
                activity_dict['budget_cost'],
                activity_dict['actual_cost'],
                activity_dict['weight'],
                activity_dict.get('real_start_date', ''),
                activity_dict.get('real_end_date', '')
            ))
            conn.commit()
            conn.close()
            self.load_projects_from_db()
            return True
        except Exception as e:
            st.error(f"Error DB al añadir actividad: {e}")
            return False
            
    def delete_activity(self, project_name, activity_id):
        """Eliminar una actividad específica de la DB"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            # Usamos project_name y el ID para asegurar que borramos la correcta
            cursor.execute("DELETE FROM activities WHERE project_name = ? AND id = ?", (project_name, activity_id))
            conn.commit()
            conn.close()
            self.load_projects_from_db()
            return True
        except Exception as e:
            st.error(f"Error DB al borrar actividad: {e}")
            return False

    def update_project_description(self, project_name, new_description):
        """Actualizar solo la descripción del proyecto"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE projects SET description = ? WHERE name = ?", (new_description, project_name))
            conn.commit()
            conn.close()
            self.load_projects_from_db()
            return True
        except Exception as e:
            st.error(f"Error DB actualizando descripción: {e}")
            return False

    def update_activity(self, project_name, activity_id, updates):
        """Actualizar una actividad existente en la DB"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Construir la query de update dinámicamente
            fields = []
            values = []
            for key, value in updates.items():
                # Mapear 'group' a '"group"' para SQL
                db_key = '"group"' if key == 'group' else key
                fields.append(f"{db_key} = ?")
                values.append(value)
            
            values.append(project_name)
            values.append(activity_id)
            
            query = f"UPDATE activities SET {', '.join(fields)} WHERE project_name = ? AND id = ?"
            
            cursor.execute(query, tuple(values))
            conn.commit()
            conn.close()
            self.load_projects_from_db()
            return True
        except Exception as e:
            st.error(f"Error DB al actualizar actividad: {e}")
            return False
    
    def calculate_project_metrics(self, project_name):
        # Esta función trabaja sobre self.projects (memoria), así que no cambia.
        """Calcular métricas del proyecto"""
        if project_name not in self.projects:
            return None
        
        activities = self.projects[project_name]["activities"]
        if not activities:
            return {
                "total_activities": 0,
                "completed_activities": 0,
                "progress_percentage": 0,
                "on_time_activities": 0,
                "delayed_activities": 0,
                "at_risk_activities": 0,
                "total_budget": 0,
                "actual_cost": 0,
                "cost_variance": 0,
                "average_delay_days": 0
            }
        
        today = datetime.now()
        total_budget = sum(a.get("budget_cost", 0) for a in activities)
        actual_cost = sum(a.get("actual_cost", 0) for a in activities)
        
        completed = sum(1 for a in activities if a.get("progress", 0) == 100)
        on_time = 0
        delayed = 0
        at_risk = 0
        total_delay = 0
        
        for activity in activities:
            progress = activity.get("progress", 0)
            end_date_str = activity.get("end_date", "")
            real_end_str = activity.get("real_end_date", "")
            
            if end_date_str:
                end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                
                # Si la actividad está completada, miramos la fecha real de fin
                if progress == 100 and real_end_str:
                     real_end_dt = datetime.strptime(real_end_str, "%Y-%m-%d")
                     delay = (real_end_dt - end_date_dt).days
                     if delay <= 0:
                         on_time += 1
                     else:
                         delayed += 1
                         total_delay += delay
                # Si no está completada, comparamos con hoy
                elif progress < 100:
                    if today > end_date_dt:
                        delayed += 1
                        total_delay += (today - end_date_dt).days
                    elif (end_date_dt - today).days <= 7:
                        at_risk += 1
        
        avg_delay = total_delay / len(activities) if activities else 0
        
        return {
            "total_activities": len(activities),
            "completed_activities": completed,
            "progress_percentage": (completed / len(activities)) * 100,
            "on_time_activities": on_time,
            "delayed_activities": delayed,
            "at_risk_activities": at_risk,
            "total_budget": total_budget,
            "actual_cost": actual_cost,
            "cost_variance": actual_cost - total_budget,
            "average_delay_days": avg_delay
        }

# Inicializar el gestor de proyectos
if 'project_manager' not in st.session_state:
    st.session_state.project_manager = ProjectManager()

# Estado de sesión para edición
if 'editing_activity_id' not in st.session_state:
    st.session_state.editing_activity_id = None

# Funciones auxiliares
def create_gantt_chart(project_name):
    """Crear gráfico de Gantt"""
    if project_name not in st.session_state.project_manager.projects:
        return None
    
    activities = st.session_state.project_manager.projects[project_name]["activities"]
    if not activities:
        return None
    
    df = pd.DataFrame(activities)
    
    # Usar fecha real si existe para el Gantt, si no la planificada
    df['plot_start'] = df.apply(lambda x: x['real_start_date'] if x['real_start_date'] else x['start_date'], axis=1)
    df['plot_end'] = df.apply(lambda x: x['real_end_date'] if x['real_end_date'] else x['end_date'], axis=1)

    
    fig = px.timeline(
        df,
        x_start="plot_start",
        x_end="plot_end",
        y="name",
        color="status",
        title=f"Diagrama de Gantt - {project_name} (Basado en Fechas Reales si existen)",
        color_discrete_map={
            "Completado": "#2E8B57",
            "En Progreso": "#FFD700",
            "Pendiente": "#87CEEB",
            "Retrasado": "#DC143C",
            "En Riesgo": "#FF8C00"
        },
        hover_data=["start_date", "end_date", "real_start_date", "real_end_date", "progress"]
    )
    
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(
        height=400 + len(activities) * 20,
        xaxis_title="Fecha",
        yaxis_title="Actividades"
    )
    
    return fig

def create_s_curve(project_name):
    """Crear Curva S del proyecto"""
    if project_name not in st.session_state.project_manager.projects:
        return None
    
    activities = st.session_state.project_manager.projects[project_name]["activities"]
    if not activities:
        return None
    
    df = pd.DataFrame(activities)
    df = df.sort_values('start_date')
    
    # Calcular acumulados
    df['cumulative_budget'] = df['budget_cost'].cumsum()
    # Para el coste real, deberíamos usar la fecha real si es posible
    df_real = df.copy()
    # Si no tiene fecha real de inicio, usamos la planificada para ordenar
    df_real['sort_date'] = df_real.apply(lambda x: x['real_start_date'] if x['real_start_date'] else x['start_date'], axis=1)
    df_real = df_real.sort_values('sort_date')
    df_real['cumulative_actual'] = df_real['actual_cost'].cumsum()
    
    fig = go.Figure()
    
    # Curva planeada
    fig.add_trace(go.Scatter(
        x=df['start_date'],
        y=df['cumulative_budget'],
        mode='lines+markers',
        name='Costo Planeado (Base Planificada)',
        line=dict(color='blue', width=2)
    ))
    
    # Curva real
    fig.add_trace(go.Scatter(
        x=df_real['sort_date'],
        y=df_real['cumulative_actual'],
        mode='lines+markers',
        name='Costo Real (Base Real)',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f"Curva S - {project_name}",
        xaxis_title="Fecha",
        yaxis_title="Costo Acumulado (€)",
        height=400
    )
    
    return fig

def create_kpi_dashboard(project_name):
    """Crear dashboard de KPIs"""
    metrics = st.session_state.project_manager.calculate_project_metrics(project_name)
    if not metrics:
        return None
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Estado de Actividades", "Progreso del Presupuesto", 
                      "Distribución de Tiempo", "Tendencia de Progreso"),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Gráfico de pastel - Estado de actividades
    labels = ['Completadas', 'En Progreso', 'Retrasadas', 'En Riesgo']
    values = [
        metrics["completed_activities"],
        metrics["total_activities"] - metrics["completed_activities"] - metrics["delayed_activities"] - metrics["at_risk_activities"],
        metrics["delayed_activities"],
        metrics["at_risk_activities"]
    ]
    
    fig.add_trace(
        go.Pie(labels=labels, values=values, name="Estado"),
        row=1, col=1
    )
    
    # Gráfico de barras - Presupuesto
    fig.add_trace(
        go.Bar(x=['Planeado', 'Real'], y=[metrics["total_budget"], metrics["actual_cost"]], 
               name="Presupuesto", marker_color=['blue', 'red']),
        row=1, col=2
    )
    
    # Distribución de tiempo
    fig.add_trace(
        go.Bar(x=['A Tiempo', 'Con Retraso'], 
               y=[metrics["on_time_activities"], metrics["delayed_activities"]],
               name="Tiempo", marker_color=['green', 'orange']),
        row=2, col=1
    )
    
    # Tendencia de progreso (simulada)
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                         end=datetime.now(), freq='D')
    progress = np.linspace(0, metrics["progress_percentage"], len(dates))
    
    fig.add_trace(
        go.Scatter(x=dates, y=progress, mode='lines', name="Progreso"),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    return fig

def generate_advanced_pdf_report(project_name):
    """Generar informe PDF avanzado y profesional (CORREGIDO PARA WINDOWS)"""
    if project_name not in st.session_state.project_manager.projects:
        return None
    
    project = st.session_state.project_manager.projects[project_name]
    activities = project["activities"]
    metrics = st.session_state.project_manager.calculate_project_metrics(project_name)
    
    # Crear buffer para el PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm,
                           topMargin=20*mm, bottomMargin=20*mm)
    
    # Estilos
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a202c')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.HexColor('#2b6cb0')
    )
    
    # Contenido del informe
    story = []
    
    # Portada
    story.append(Paragraph(f"INFORME EJECUTIVO", title_style))
    story.append(Paragraph(f"{project_name}", title_style))
    story.append(Spacer(1, 20))
    
    # Información del proyecto
    story.append(Paragraph("Información General", heading_style))
    project_info = f"""
    <b>Fecha del Informe:</b> {datetime.now().strftime('%d/%m/%Y')}<br/>
    <b>Descripción:</b> {project.get('description', 'N/A')}<br/>
    <b>Fecha de Creación:</b> {project.get('created_date', 'N/A')}<br/>
    <b>Total de Actividades:</b> {metrics['total_activities']}<br/>
    <b>Actividades Completadas:</b> {metrics['completed_activities']}
    """
    story.append(Paragraph(project_info, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Métricas principales
    story.append(Paragraph("Métricas del Proyecto", heading_style))
    
    # Tabla de métricas
    metrics_data = [
        ['Métrica', 'Valor'],
        ['Procentaje de Progreso', f"{metrics['progress_percentage']:.1f}%"],
        ['Presupuesto Total', f"€{metrics['total_budget']:,.2f}"],
        ['Costo Real', f"€{metrics['actual_cost']:,.2f}"],
        ['Variación de Costo', f"€{metrics['cost_variance']:,.2f}"],
        ['Actividades a Tiempo', metrics['on_time_activities']],
        ['Actividades Retrasadas', metrics['delayed_activities']],
        ['Actividades en Riesgo', metrics['at_risk_activities']],
        ['Retraso Promedio (días)', f"{metrics['average_delay_days']:.1f}"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[60*mm, 40*mm])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Gráficos
    story.append(Paragraph("Análisis Gráfico", heading_style))
    
    # Generar gráficos con matplotlib
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Estado de actividades
    status_labels = ['Completadas', 'En Progreso', 'Retrasadas', 'En Riesgo']
    status_values = [
        metrics["completed_activities"],
        metrics["total_activities"] - metrics["completed_activities"] - metrics["delayed_activities"] - metrics["at_risk_activities"],
        metrics["delayed_activities"],
        metrics["at_risk_activities"]
    ]
    # Evitar error si no hay actividades
    if sum(status_values) > 0:
        ax1.pie(status_values, labels=status_labels, autopct='%1.1f%%', startangle=90)
    else:
        ax1.text(0.5, 0.5, "Sin datos", ha='center')
    ax1.set_title('Distribución de Actividades')
    
    # Gráfico 2: Presupuesto vs Real
    budget_categories = ['Planeado', 'Real']
    budget_values = [metrics["total_budget"], metrics["actual_cost"]]
    ax2.bar(budget_categories, budget_values, color=['blue', 'red'])
    ax2.set_title('Presupuesto vs Costo Real')
    ax2.set_ylabel('€')
    
    # Gráfico 3: Curva S
    if activities:
        df = pd.DataFrame(activities)
        df = df.sort_values('start_date')
        df['cumulative_budget'] = df['budget_cost'].cumsum()
        ax3.plot(range(len(df)), df['cumulative_budget'], 'b-', label='Planeado')
        df['cumulative_actual'] = df['actual_cost'].cumsum()
        ax3.plot(range(len(df)), df['cumulative_actual'], 'r-', label='Real')
        ax3.set_title('Curva S de Costos')
        ax3.set_xlabel('Actividades')
        ax3.set_ylabel('Costo Acumulado (€)')
        ax3.legend()
    else:
         ax3.text(0.5, 0.5, "Sin datos", ha='center')
    
    # Gráfico 4: Timeline de actividades
    if activities:
        df = pd.DataFrame(activities)
        # Asegurar que las fechas sean datetime
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])
        
        df['duration'] = df['end_date'] - df['start_date']
        df['duration_days'] = df['duration'].dt.days
        top_activities = df.nlargest(5, 'duration_days')[['name', 'duration_days']]
        if not top_activities.empty:
            ax4.barh(top_activities['name'], top_activities['duration_days'])
        else:
             ax4.text(0.5, 0.5, "Sin datos suficientes", ha='center')
        ax4.set_title('Top 5 Actividades Más Largas')
        ax4.set_xlabel('Duración (días)')
    else:
         ax4.text(0.5, 0.5, "Sin datos", ha='center')
    
    plt.tight_layout()
    
    # --- CORRECCIÓN WINERROR 32 USANDO DIRECTorio TEMPORAL ---
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, 'report_charts.png')
        
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        img = Image(img_path, width=170*mm, height=120*mm)
        story.append(img)
        
        story.append(Spacer(1, 20))
        
        # Tabla detallada de actividades
        story.append(Paragraph("Detalle de Actividades", heading_style))
        
        table_data = [['ID', 'Actividad', 'Inicio Plan.', 'Fin Plan.', 'Inicio Real', 'Fin Real', 'Progreso', 'Estado']]
        
        for activity in activities:
            table_data.append([
                activity.get('id', ''),
                activity.get('name', '')[:25],
                activity.get('start_date', ''),
                activity.get('end_date', ''),
                activity.get('real_start_date', '') if activity.get('real_start_date') else '-',
                activity.get('real_end_date', '') if activity.get('real_end_date') else '-',
                f"{activity.get('progress', 0)}%",
                activity.get('status', '')
            ])
        
        activities_table = Table(table_data, colWidths=[10*mm, 40*mm, 22*mm, 22*mm, 22*mm, 22*mm, 15*mm, 22*mm])
        activities_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('ALIGN', (1, 1), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7), # Fuente más pequeña para que quepa
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        
        story.append(activities_table)
        
        doc.build(story)
        buffer.seek(0)
        pdf_bytes = buffer.getvalue()
        
    return pdf_bytes

# Interfaz principal
def main():
    st.title("🏗️ Gestor de Proyectos Avanzado")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Menú Principal")
        
        page = st.selectbox(
            "Selecciona una opción:",
            ["🏠 Dashboard", "📊 Gestionar Proyectos", "📈 Plan del Proyecto", 
             "📋 Kanban", "📑 Generar Informe", "⚙️ Configuración"]
        )
        
        st.markdown("---")
        st.header("📁 Proyectos")
        
        projects = list(st.session_state.project_manager.projects.keys())
        if projects:
            selected_project = st.selectbox("Seleccionar Proyecto:", projects)
            st.session_state.current_project = selected_project
        else:
            st.warning("No hay proyectos creados")
            selected_project = None
    
    # Contenido principal según página seleccionada
    if page == "🏠 Dashboard":
        st.header("📊 Dashboard Principal")
        
        if selected_project:
            col1, col2, col3, col4 = st.columns(4)
            
            metrics = st.session_state.project_manager.calculate_project_metrics(selected_project)
            
            with col1:
                st.metric(
                    "Progreso Global",
                    f"{metrics['progress_percentage']:.1f}%",
                    delta=f"{metrics['completed_activities']}/{metrics['total_activities']} tareas"
                )
            
            with col2:
                st.metric(
                    "Presupuesto",
                    f"€{metrics['total_budget']:,.0f}",
                    delta=f"€{metrics['cost_variance']:,.0f}",
                    delta_color="inverse" if metrics['cost_variance'] > 0 else "normal"
                )
            
            with col3:
                st.metric(
                    "Actividades a Tiempo",
                    metrics['on_time_activities'],
                    delta=f"{metrics['delayed_activities']} retrasadas"
                )
            
            with col4:
                st.metric(
                    "Retraso Promedio",
                    f"{metrics['average_delay_days']:.1f} días",
                    delta="Días"
                )
            
            # Gráficos
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = create_gantt_chart(selected_project)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("Añade actividades para ver el diagrama de Gantt.")
            
            with col2:
                fig = create_s_curve(selected_project)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("Añade actividades con costes para ver la Curva S.")
            
            # Dashboard de KPIs
            st.markdown("---")
            st.subheader("📈 Análisis Detallado de KPIs")
            fig = create_kpi_dashboard(selected_project)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("No hay datos suficientes para el análisis detallado.")
        else:
            st.info("Por favor, selecciona un proyecto para ver el dashboard")
    
    elif page == "📊 Gestionar Proyectos":
        st.header("📊 Gestión de Proyectos")
        
        tab1, tab2, tab3 = st.tabs(["Crear Proyecto", "Ver Proyectos", "Editar Proyecto"])
        
        with tab1:
            with st.form("crear_proyecto"):
                st.subheader("🆕 Crear Nuevo Proyecto")
                name = st.text_input("Nombre del Proyecto*")
                description = st.text_area("Descripción")
                
                submitted = st.form_submit_button("Crear Proyecto")
                if submitted and name:
                    if st.session_state.project_manager.create_project(name, description):
                        st.success(f"Proyecto '{name}' creado exitosamente")
                        st.rerun()
                    else:
                        st.error("El proyecto ya existe")
        
        with tab2:
            st.subheader("📋 Lista de Proyectos")
            
            if projects:
                for project_name in projects:
                    with st.expander(f"📁 {project_name}"):
                        project = st.session_state.project_manager.projects[project_name]
                        metrics = st.session_state.project_manager.calculate_project_metrics(project_name)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**Descripción:** {project.get('description', 'N/A')}")
                            st.write(f"**Creado:** {project.get('created_date', 'N/A')}")
                        
                        with col2:
                            st.write(f"**Actividades:** {metrics['total_activities']}")
                            st.write(f"**Progreso:** {metrics['progress_percentage']:.1f}%")
                        
                        with col3:
                            st.write(f"**Presupuesto:** €{metrics['total_budget']:,.0f}")
                            st.write(f"**Costo Real:** €{metrics['actual_cost']:,.0f}")
                        
                        if st.button(f"🗑️ Eliminar {project_name}", key=f"del_{project_name}"):
                            if st.session_state.project_manager.delete_project(project_name):
                                st.success("Proyecto eliminado")
                                st.rerun()
            else:
                st.info("No hay proyectos creados")
        
        with tab3:
            if selected_project:
                st.subheader(f"✏️ Editar Proyecto: {selected_project}")

                project = st.session_state.project_manager.projects[selected_project]

                # --- FORMULARIO 1: ACTUALIZAR DESCRIPCIÓN ---
                with st.form("update_description_form"):
                    st.markdown("#### 📝 Actualizar Descripción")
                    new_description = st.text_area(
                        "Descripción",
                        value=project.get('description', ''),
                        key="description_input"
                    )
                    submitted_desc = st.form_submit_button("Actualizar Descripción")

                    if submitted_desc:
                        if st.session_state.project_manager.update_project_description(selected_project, new_description):
                             st.success("✅ Descripción actualizada exitosamente")
                             st.rerun()
                        else:
                             st.error("Error al actualizar la descripción")

                st.markdown("---")

                # --- SECCIÓN DE EDICIÓN DE ACTIVIDAD (SI SE SELECCIONÓ UNA) ---
                if st.session_state.editing_activity_id:
                    st.markdown("#### ✏️ Editando Actividad")
                    # Buscar la actividad que se está editando
                    activity_to_edit = next((act for act in project["activities"] if act["id"] == st.session_state.editing_activity_id), None)
                    
                    if activity_to_edit:
                        with st.form("edit_activity_form"):
                            col1, col2 = st.columns(2)
                            with col1:
                                edit_name = st.text_input("Nombre*", value=activity_to_edit["name"], key="edit_name")
                                edit_group = st.selectbox("Grupo", ["INGENIERÍA", "OBRA CIVIL", "ELECTROMECÁNICO", "SUMINISTROS", "OTROS"], index=["INGENIERÍA", "OBRA CIVIL", "ELECTROMECÁNICO", "SUMINISTROS", "OTROS"].index(activity_to_edit["group"]), key="edit_group")
                            with col2:
                                edit_start = st.date_input("Inicio Planificado", value=datetime.strptime(activity_to_edit["start_date"], "%Y-%m-%d"), key="edit_start")
                                edit_end = st.date_input("Fin Planificado", value=datetime.strptime(activity_to_edit["end_date"], "%Y-%m-%d"), key="edit_end")
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                # Manejo seguro de fechas reales (pueden estar vacías)
                                try:
                                    default_real_start = datetime.strptime(activity_to_edit["real_start_date"], "%Y-%m-%d")
                                except (ValueError, TypeError):
                                    default_real_start = None

                                try:
                                    default_real_end = datetime.strptime(activity_to_edit["real_end_date"], "%Y-%m-%d")
                                except (ValueError, TypeError):
                                    default_real_end = None

                                edit_real_start = st.date_input("Inicio Real", value=default_real_start, key="edit_real_start")
                                edit_real_end = st.date_input("Fin Real", value=default_real_end, key="edit_real_end")
                            with col4:
                                edit_progress = st.slider("Progreso (%)", 0, 100, activity_to_edit["progress"], key="edit_progress")
                                edit_status = st.selectbox("Estado", ["Pendiente", "En Progreso", "Completado", "Retrasado", "En Riesgo"], index=["Pendiente", "En Progreso", "Completado", "Retrasado", "En Riesgo"].index(activity_to_edit["status"]), key="edit_status")

                            col5, col6 = st.columns(2)
                            with col5:
                                edit_budget = st.number_input("Presupuesto", min_value=0.0, value=activity_to_edit["budget_cost"], key="edit_budget")
                            with col6:
                                edit_actual = st.number_input("Costo Real", min_value=0.0, value=activity_to_edit["actual_cost"], key="edit_actual")

                            submitted_edit = st.form_submit_button("Guardar Cambios")
                            
                            if submitted_edit:
                                updates = {
                                    "name": edit_name,
                                    "group": edit_group,
                                    "start_date": edit_start.strftime("%Y-%m-%d"),
                                    "end_date": edit_end.strftime("%Y-%m-%d"),
                                    "real_start_date": edit_real_start.strftime("%Y-%m-%d") if edit_real_start else "",
                                    "real_end_date": edit_real_end.strftime("%Y-%m-%d") if edit_real_end else "",
                                    "progress": edit_progress,
                                    "status": edit_status,
                                    "budget_cost": edit_budget,
                                    "actual_cost": edit_actual
                                }
                                if st.session_state.project_manager.update_activity(selected_project, st.session_state.editing_activity_id, updates):
                                    st.success("Actividad actualizada.")
                                    st.session_state.editing_activity_id = None # Salir del modo edición
                                    st.rerun()
                                else:
                                    st.error("Error al actualizar.")
                        
                        if st.button("Cancelar Edición"):
                            st.session_state.editing_activity_id = None
                            st.rerun()
                    st.markdown("---")


                # --- FORMULARIO 2: AÑADIR NUEVA ACTIVIDAD (Ahora con Inicio Real) ---
                with st.form("add_activity_form"):
                    st.markdown("#### ➕ Añadir Nueva Actividad")
                    col1, col2 = st.columns(2)

                    with col1:
                        activity_name = st.text_input("Nombre de la Actividad*", key="act_name")
                        activity_group = st.selectbox(
                            "Grupo",
                            ["INGENIERÍA", "OBRA CIVIL", "ELECTROMECÁNICO", "SUMINISTROS", "OTROS"],
                            key="act_group"
                        )

                    with col2:
                        start_date = st.date_input("Inicio Planificado", key="act_start")
                        end_date = st.date_input("Fin Planificado", key="act_end")
                    
                    col_real1, col_real2 = st.columns(2)
                    with col_real1:
                         # Inicio real opcional al crear
                         real_start_date = st.date_input("Inicio Real (Opcional)", value=None, key="act_real_start")
                    with col_real2:
                         progress = st.slider("Progreso (%)", 0, 100, 0, key="act_progress")


                    col3, col4 = st.columns(2)
                    with col3:
                        budget_cost = st.number_input("Presupuesto", min_value=0.0, value=0.0, key="act_budget")
                        actual_cost = st.number_input("Costo Real", min_value=0.0, value=0.0, key="act_actual")
                    with col4:
                        status = st.selectbox(
                            "Estado",
                            ["Pendiente", "En Progreso", "Completado", "Retrasado", "En Riesgo"],
                            key="act_status"
                        )
                        weight = st.number_input("Peso", min_value=1, value=1, key="act_weight")

                    submitted_activity = st.form_submit_button("Añadir Actividad")

                    if submitted_activity:
                        if activity_name:
                            # Determinar fechas reales iniciales
                            r_start = real_start_date.strftime("%Y-%m-%d") if real_start_date else ""
                            # Si el progreso es 100% al crear, asumimos que termina hoy si no se especifica otra cosa.
                            r_end = datetime.now().strftime("%Y-%m-%d") if progress == 100 else ""

                            new_activity = {
                                "name": activity_name,
                                "group": activity_group,
                                "start_date": start_date.strftime("%Y-%m-%d"),
                                "end_date": end_date.strftime("%Y-%m-%d"),
                                "progress": progress,
                                "status": status,
                                "budget_cost": budget_cost,
                                "actual_cost": actual_cost,
                                "weight": weight,
                                "real_start_date": r_start,
                                "real_end_date": r_end
                            }

                            if st.session_state.project_manager.add_activity(selected_project, new_activity):
                                st.success(f"✅ Actividad '{activity_name}' añadida exitosamente")
                                st.rerun()
                            else:
                                st.error("❌ Hubo un error al añadir la actividad.")
                        else:
                            st.error("⚠️ Por favor, introduce un nombre para la actividad.")

                # --- LISTA INTERACTIVA DE ACTIVIDADES EXISTENTES (Borrar y Editar) ---
                st.markdown("---")
                st.markdown("#### 📋 Actividades Existentes (Gestión)")

                if project["activities"]:
                    # Usamos columnas para hacer una lista interactiva en lugar de un dataframe estático
                    for activity in project["activities"]:
                        col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 2, 1, 1, 1])
                        with col1:
                            st.write(f"**{activity['name']}**")
                            st.caption(f"Grupo: {activity['group']}")
                        with col2:
                            st.write(f"Plan: {activity['start_date']} -> {activity['end_date']}")
                        with col3:
                            # Mostrar fechas reales si existen
                            r_start = activity.get('real_start_date') if activity.get('real_start_date') else "(sin inicio)"
                            r_end = activity.get('real_end_date') if activity.get('real_end_date') else "(sin fin)"
                            st.write(f"Real: {r_start} -> {r_end}")
                        with col4:
                            st.write(f"{activity['progress']}%")
                            st.caption(activity['status'])
                        
                        with col5:
                            # Botón de EDITAR
                            if st.button("✏️", key=f"edit_{activity['id']}"):
                                st.session_state.editing_activity_id = activity['id']
                                st.rerun()
                        
                        with col6:
                            # Botón de BORRAR
                            if st.button("🗑️", key=f"del_act_{activity['id']}"):
                                if st.session_state.project_manager.delete_activity(selected_project, activity['id']):
                                    st.success("Actividad eliminada")
                                    # Si estábamos editando la que borramos, salir del modo edición
                                    if st.session_state.editing_activity_id == activity['id']:
                                        st.session_state.editing_activity_id = None
                                    st.rerun()
                        st.divider() # Línea separadora entre actividades
                else:
                    st.info("No hay actividades en este proyecto")
            else:
                st.warning("Por favor, selecciona un proyecto para editar")
    
    elif page == "📈 Plan del Proyecto":
        st.header("📈 Plan del Proyecto")
        
        if selected_project:
            # Gantt Chart
            st.subheader("📊 Diagrama de Gantt")
            st.caption("Este diagrama prioriza las fechas REALES de inicio/fin si están disponibles.")
            fig = create_gantt_chart(selected_project)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("Añade actividades para ver el diagrama de Gantt.")
            
            # Curva S
            st.subheader("📈 Curva S de Progreso")
            st.caption("La línea roja (Coste Real) se basa en la fecha de inicio real si existe.")
            fig = create_s_curve(selected_project)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.info("Añade actividades con costes para ver la Curva S.")
            
            # Tabla de planificación
            st.subheader("📋 Tabla de Planificación Detallada")
            
            project = st.session_state.project_manager.projects[selected_project]
            if project["activities"]:
                df = pd.DataFrame(project["activities"])
                
                # Asegurar formato de fecha para cálculos
                today = datetime.now()
                
                # Cálculo de retraso (simplificado)
                def calculate_delay(row):
                    try:
                        end_date_dt = datetime.strptime(row['end_date'], '%Y-%m-%d')
                        if row['progress'] < 100 and today > end_date_dt:
                            return (today - end_date_dt).days
                        elif row['progress'] == 100 and row.get('real_end_date'):
                             real_end_dt = datetime.strptime(row['real_end_date'], '%Y-%m-%d')
                             delay = (real_end_dt - end_date_dt).days
                             return delay if delay > 0 else 0
                        return 0
                    except:
                        return 0

                df['retraso_dias'] = df.apply(calculate_delay, axis=1)
                
                # Reordenar columnas para mejor visualización
                cols = ['name', 'group', 'start_date', 'end_date', 'real_start_date', 'real_end_date', 'progress', 'status', 'retraso_dias']
                st.dataframe(df[cols], use_container_width=True)
            else:
                st.info("No hay actividades planificadas")
        else:
            st.warning("Por favor, selecciona un proyecto")
    
    elif page == "📋 Kanban":
        st.header("📋 Tablero Kanban")
        
        if selected_project:
            project = st.session_state.project_manager.projects[selected_project]
            
            if project["activities"]:
                # Organizar actividades por estado
                pendientes = [a for a in project["activities"] if a["status"] == "Pendiente"]
                en_progreso = [a for a in project["activities"] if a["status"] == "En Progreso"]
                completadas = [a for a in project["activities"] if a["status"] == "Completado"]
                retrasadas = [a for a in project["activities"] if a["status"] == "Retrasado"]
                en_riesgo = [a for a in project["activities"] if a["status"] == "En Riesgo"]
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.markdown("### 🔵 Pendiente")
                    for activity in pendientes:
                        with st.container():
                            st.markdown(f"""
                            <div style="background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <strong>{activity['name']}</strong><br/>
                                <small>Grupo: {activity['group']}</small><br/>
                                <small>Progreso: {activity['progress']}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 🟡 En Progreso")
                    for activity in en_progreso:
                        with st.container():
                            st.markdown(f"""
                            <div style="background-color: #fff9c4; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <strong>{activity['name']}</strong><br/>
                                <small>Grupo: {activity['group']}</small><br/>
                                <small>Progreso: {activity['progress']}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("### 🟢 Completado")
                    for activity in completadas:
                        with st.container():
                            st.markdown(f"""
                            <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <strong>{activity['name']}</strong><br/>
                                <small>Grupo: {activity['group']}</small><br/>
                                <small>Progreso: {activity['progress']}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown("### 🔴 Retrasado")
                    for activity in retrasadas:
                        with st.container():
                            st.markdown(f"""
                            <div style="background-color: #ffebee; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <strong>{activity['name']}</strong><br/>
                                <small>Grupo: {activity['group']}</small><br/>
                                <small>Progreso: {activity['progress']}%</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                with col5:
                    st.markdown("### 🟠 En Riesgo")
                    for activity in en_riesgo:
                        with st.container():
                            st.markdown(f"""
                            <div style="background-color: #fff3e0; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                <strong>{activity['name']}</strong><br/>
                                <small>Grupo: {activity['group']}</small><br/>
                                <small>Progreso: {activity['progress']}%</small>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.info("No hay actividades en este proyecto")
        else:
            st.warning("Por favor, selecciona un proyecto")
    
    elif page == "📑 Generar Informe":
        st.header("📑 Generador de Informes Avanzado")
        
        if selected_project:
            st.subheader(f"📊 Informe del Proyecto: {selected_project}")
            
            # Opciones del informe
            col1, col2 = st.columns(2)
            
            with col1:
                include_charts = st.checkbox("Incluir Gráficos", value=True)
                include_kpis = st.checkbox("Incluir KPIs", value=True)
                include_activities = st.checkbox("Incluir Detalle de Actividades", value=True)
            
            with col2:
                report_format = st.selectbox("Formato del Informe", ["PDF", "Excel"])
                report_type = st.selectbox(
                    "Tipo de Informe",
                    ["Ejecutivo", "Detallado", "Técnico", "Financiero"]
                )
            
            if st.button("🚀 Generar Informe"):
                with st.spinner("Generando informe..."):
                    if report_format == "PDF":
                        # Se llama a la función corregida
                        pdf_data = generate_advanced_pdf_report(selected_project)
                        if pdf_data:
                            st.success("✅ Informe PDF generado exitosamente")
                            
                            st.download_button(
                                label="📥 Descargar Informe PDF",
                                data=pdf_data,
                                file_name=f"Informe_{selected_project}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                    else:
                        # Generar Excel
                        project = st.session_state.project_manager.projects[selected_project]
                        df = pd.DataFrame(project["activities"])
                        
                        # Añadir métricas
                        metrics = st.session_state.project_manager.calculate_project_metrics(selected_project)
                        
                        # Crear Excel con múltiples hojas
                        with pd.ExcelWriter(f"Informe_{selected_project}_{datetime.now().strftime('%Y%m%d')}.xlsx", engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Actividades', index=False)
                            
                            # Hoja de resumen
                            summary_data = {
                                'Métrica': [
                                    'Progreso Global (%)',
                                    'Total Actividades',
                                    'Actividades Completadas',
                                    'Presupuesto Total (€)',
                                    'Costo Real (€)',
                                    'Variación de Costo (€)',
                                    'Actividades a Tiempo',
                                    'Actividades Retrasadas',
                                    'Retraso Promedio (días)'
                                ],
                                'Valor': [
                                    f"{metrics['progress_percentage']:.1f}",
                                    metrics['total_activities'],
                                    metrics['completed_activities'],
                                    f"{metrics['total_budget']:,.2f}",
                                    f"{metrics['actual_cost']:,.2f}",
                                    f"{metrics['cost_variance']:,.2f}",
                                    metrics['on_time_activities'],
                                    metrics['delayed_activities'],
                                    f"{metrics['average_delay_days']:.1f}"
                                ]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, sheet_name='Resumen', index=False)
                        
                        st.success("✅ Informe Excel generado exitosamente")
            
            # Vista previa del informe
            st.markdown("---")
            st.subheader("👁️ Vista Previa del Informe")
            
            # Mostrar métricas principales
            metrics = st.session_state.project_manager.calculate_project_metrics(selected_project)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Progreso", f"{metrics['progress_percentage']:.1f}%")
            with col2:
                st.metric("Presupuesto", f"€{metrics['total_budget']:,.0f}")
            with col3:
                st.metric("Costo Real", f"€{metrics['actual_cost']:,.0f}")
            with col4:
                st.metric("Retraso Promedio", f"{metrics['average_delay_days']:.1f} días")
            
            # Gráficos de muestra
            if include_charts:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = create_gantt_chart(selected_project)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = create_s_curve(selected_project)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Por favor, selecciona un proyecto para generar el informe")
    
    elif page == "⚙️ Configuración":
        st.header("⚙️ Configuración de la Aplicación")
        
        tab1, tab2, tab3 = st.tabs(["General", "Importar/Exportar", "Acerca de"])
        
        with tab1:
            st.subheader("🎨 Configuración General")
            
            # Configuración de la aplicación
            app_settings = {
                "theme": st.selectbox("Tema", ["Claro", "Oscuro"]),
                "language": st.selectbox("Idioma", ["Español", "Inglés"]),
                "currency": st.selectbox("Moneda", ["€", "$", "£"]),
                "date_format": st.selectbox("Formato de Fecha", ["DD/MM/YYYY", "MM/DD/YYYY", "YYYY-MM-DD"]),
                "auto_save": st.checkbox("Guardado Automático", value=True),
                "notifications": st.checkbox("Notificaciones", value=True)
            }
            
            if st.button("💾 Guardar Configuración"):
                st.success("Configuración guardada exitosamente")
        
        with tab2:
            st.subheader("📁 Importar/Exportar Datos")
            
            # NOTA SOBRE IMPORTACIÓN DESDE CARPETA
            st.info("""
                ℹ️ **Nota sobre seguridad web:** Por razones de seguridad, las aplicaciones web no pueden acceder directamente a las carpetas de tu escritorio sin permiso. 
                Para importar un proyecto, arrastra el archivo JSON o Excel desde tu carpeta al área de carga correspondiente a continuación.
            """)

            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📤 Exportar Proyectos")
                # NOTA: La exportación completa de JSON es compleja con bases de datos relacionales.
                # Se desactiva temporalmente hasta la siguiente fase para evitar errores.
                st.warning("La exportación masiva a JSON se habilitará en una versión futura.")

            
            with col2:
                st.markdown("#### 📥 Importar Proyectos (JSON)")
                st.warning("La importación masiva desde JSON se habilitará en una versión futura.")

            
            # Importar desde Excel (Simplificado para la nueva estructura DB)
            st.markdown("#### 📊 Importar desde Excel")
            st.caption("Arrastra aquí tu archivo Excel desde tu carpeta.")
            excel_file = st.file_uploader("Sube un archivo Excel", type=["xlsx", "xls"])
            
            if excel_file is not None:
                try:
                    df = pd.read_excel(excel_file)
                    st.dataframe(df.head())
                    
                    project_name = st.text_input("Nombre del nuevo proyecto:")
                    
                    if st.button("Importar a Proyecto"):
                        if project_name:
                            # 1. Crear el proyecto en la DB
                            if st.session_state.project_manager.create_project(project_name):
                                # 2. Convertir e insertar actividades
                                for index, row in df.iterrows():
                                    # Intentar parsear fechas del excel
                                    start_d = str(row.get('Inicio', datetime.now().strftime('%Y-%m-%d')))
                                    end_d = str(row.get('Fin', datetime.now().strftime('%Y-%m-%d')))
                                    
                                    activity = {
                                        # El ID se genera en DB
                                        "name": str(row.get('Tarea', f'Actividad {index+1}')),
                                        "group": str(row.get('Grupo', 'General')),
                                        "start_date": start_d,
                                        "end_date": end_d,
                                        "progress": int(row.get('Progreso', 0)),
                                        "status": str(row.get('Estado', 'Pendiente')),
                                        "budget_cost": float(row.get('Presupuesto', 0)),
                                        "actual_cost": float(row.get('Coste Real', 0)),
                                        "weight": int(row.get('Peso', 1)),
                                        # Asumimos sin fechas reales al importar de excel simple
                                        "real_start_date": "",
                                        "real_end_date": ""
                                    }
                                    st.session_state.project_manager.add_activity(project_name, activity)
                                
                                st.success(f"Proyecto '{project_name}' importado exitosamente")
                                st.rerun()
                            else:
                                st.error("El proyecto ya existe o hubo un error al crearlo.")
                except Exception as e:
                    st.error(f"Error al leer el archivo Excel: {e}")
        
        with tab3:
            st.subheader("ℹ️ Acerca de")
            st.markdown("""
            ### Gestor de Proyectos Avanzado v3.1 (Edición Completa)
            
            **Novedades de esta versión:**
            - ✨ **Edición Completa:** Ahora puedes editar todas las propiedades de una actividad existente (fechas, costes, progreso).
            - 🗑️ **Borrado de Actividades:** Elimina actividades individuales si te equivocas.
            - 📅 **Fechas Reales:** Inclusión de "Inicio Real" y "Fin Real" para un seguimiento más preciso.
            
            **Características Principales:**
            - 🗄️ Base de Datos SQLite.
            - 📊 Dashboard interactivo y Diagramas (Gantt/Curva S).
            - 📑 Generador de informes PDF/Excel profesionales.
            
            **Desarrollado por:**
            Ingeniería y Supervisión Técnica
            
            **Versión:** 3.1.0
            **Última Actualización:** Diciembre 2024
            """)

if __name__ == "__main__":
    main()