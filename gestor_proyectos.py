import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import datetime, timedelta, date
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
# CLASE PROJECTMANAGER CON SQLITE (MEJORADA V4)
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
        """Inicializar esquema de base de datos y actualizar si es necesario"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Tabla Proyectos BASE
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT,
                created_date TEXT,
                settings TEXT
            )
        ''')
        
        # --- MIGRACIÓN: Intentar añadir columnas de fechas de proyecto si no existen ---
        try:
            cursor.execute("ALTER TABLE projects ADD COLUMN project_start_date TEXT")
            cursor.execute("ALTER TABLE projects ADD COLUMN project_end_date TEXT")
            conn.commit()
            print("Base de datos actualizada con columnas de fechas de proyecto.")
        except sqlite3.OperationalError:
            # Las columnas ya existen, ignorar el error
            pass
        
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
            # Manejo seguro de las nuevas fechas de proyecto (pueden ser NULL)
            p_start = proj_row['project_start_date'] if proj_row['project_start_date'] else ""
            p_end = proj_row['project_end_date'] if proj_row['project_end_date'] else ""

            self.projects[proj_name] = {
                "description": proj_row['description'],
                "created_date": proj_row['created_date'],
                "project_start_date": p_start,
                "project_end_date": p_end,
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
            
            # Inicializamos las fechas de proyecto como vacías
            cursor.execute("""
                INSERT INTO projects (name, description, created_date, settings, project_start_date, project_end_date)
                VALUES (?, ?, ?, ?, "", "")
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

    def update_project_details(self, project_name, new_description, new_start_date, new_end_date):
        """Actualizar descripción y fechas globales del proyecto"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            # Convertir fechas a string si no son None, si no cadena vacía
            s_date_str = new_start_date.strftime("%Y-%m-%d") if new_start_date else ""
            e_date_str = new_end_date.strftime("%Y-%m-%d") if new_end_date else ""

            cursor.execute("""
                UPDATE projects 
                SET description = ?, project_start_date = ?, project_end_date = ? 
                WHERE name = ?
            """, (new_description, s_date_str, e_date_str, project_name))
            conn.commit()
            conn.close()
            self.load_projects_from_db()
            return True
        except Exception as e:
            st.error(f"Error DB actualizando detalles del proyecto: {e}")
            return False

    def update_activity(self, project_name, activity_id, updates):
        """Actualizar una actividad existente en la DB"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            fields = []
            values = []
            for key, value in updates.items():
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
        """Calcular métricas del proyecto"""
        if project_name not in self.projects:
            return None
        
        activities = self.projects[project_name]["activities"]
        if not activities:
            return {
                "total_activities": 0, "completed_activities": 0, "progress_percentage": 0,
                "on_time_activities": 0, "delayed_activities": 0, "at_risk_activities": 0,
                "total_budget": 0, "actual_cost": 0, "cost_variance": 0, "average_delay_days": 0
            }
        
        today = datetime.now()
        total_budget = sum(a.get("budget_cost", 0) for a in activities)
        actual_cost = sum(a.get("actual_cost", 0) for a in activities)
        completed = sum(1 for a in activities if a.get("progress", 0) == 100)
        on_time, delayed, at_risk, total_delay = 0, 0, 0, 0
        
        for activity in activities:
            progress = activity.get("progress", 0)
            end_date_str = activity.get("end_date", "")
            real_end_str = activity.get("real_end_date", "")
            
            if end_date_str:
                try:
                    end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
                    if progress == 100 and real_end_str:
                         real_end_dt = datetime.strptime(real_end_str, "%Y-%m-%d")
                         delay = (real_end_dt - end_date_dt).days
                         if delay <= 0: on_time += 1
                         else: delayed += 1; total_delay += delay
                    elif progress < 100:
                        if today > end_date_dt:
                            delayed += 1; total_delay += (today - end_date_dt).days
                        elif (end_date_dt - today).days <= 7: at_risk += 1
                except ValueError: pass # Ignorar fechas inválidas

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

# --- FUNCIÓN AUXILIAR PARA DETERMINAR ESTADO AUTOMÁTICO ---
def determine_status(progress):
    if progress == 0: return "Pendiente"
    elif progress == 100: return "Completado"
    else: return "En Progreso"

# Funciones auxiliares de Gráficos (Sin cambios mayores, solo robustez)
def create_gantt_chart(project_name):
    if project_name not in st.session_state.project_manager.projects: return None
    activities = st.session_state.project_manager.projects[project_name]["activities"]
    if not activities: return None
    df = pd.DataFrame(activities)
    # Usar fecha real si existe para el Gantt, si no la planificada. Validar que existan.
    df['plot_start'] = df.apply(lambda x: x.get('real_start_date') if x.get('real_start_date') else x.get('start_date'), axis=1)
    df['plot_end'] = df.apply(lambda x: x.get('real_end_date') if x.get('real_end_date') else x.get('end_date'), axis=1)
    # Filtrar filas sin fechas válidas
    df = df[df['plot_start'].astype(bool) & df['plot_end'].astype(bool)]
    if df.empty: return None

    fig = px.timeline(df, x_start="plot_start", x_end="plot_end", y="name", color="status",
        title=f"Diagrama de Gantt - {project_name} (Prioriza Fechas Reales)",
        color_discrete_map={"Completado": "#2E8B57", "En Progreso": "#FFD700", "Pendiente": "#87CEEB", "Retrasado": "#DC143C", "En Riesgo": "#FF8C00"})
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(height=400 + len(activities) * 20, xaxis_title="Fecha", yaxis_title="Actividades")
    return fig

def create_s_curve(project_name):
    if project_name not in st.session_state.project_manager.projects: return None
    activities = st.session_state.project_manager.projects[project_name]["activities"]
    if not activities: return None
    df = pd.DataFrame(activities)
    # Filtrar actividades sin fecha de inicio planificada
    df = df[df['start_date'].astype(bool)].copy()
    if df.empty: return None

    df = df.sort_values('start_date')
    df['cumulative_budget'] = df['budget_cost'].cumsum()
    
    df_real = df.copy()
    # Usar fecha real de inicio si existe para ordenar la curva real
    df_real['sort_date'] = df_real.apply(lambda x: x.get('real_start_date') if x.get('real_start_date') else x.get('start_date'), axis=1)
    df_real = df_real.sort_values('sort_date')
    df_real['cumulative_actual'] = df_real['actual_cost'].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['start_date'], y=df['cumulative_budget'], mode='lines+markers', name='Costo Planeado (Base Planificada)', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=df_real['sort_date'], y=df_real['cumulative_actual'], mode='lines+markers', name='Costo Real (Base Real)', line=dict(color='red', width=2)))
    fig.update_layout(title=f"Curva S - {project_name}", xaxis_title="Fecha", yaxis_title="Costo Acumulado (€)", height=400)
    return fig

def create_kpi_dashboard(project_name):
    metrics = st.session_state.project_manager.calculate_project_metrics(project_name)
    if not metrics: return None
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Estado de Actividades", "Progreso del Presupuesto", "Distribución de Tiempo", "Tendencia de Progreso"),
        specs=[[{"type": "pie"}, {"type": "bar"}], [{"type": "bar"}, {"type": "scatter"}]])
    
    labels = ['Completadas', 'En Progreso', 'Retrasadas', 'En Riesgo']
    values = [metrics["completed_activities"], metrics["total_activities"] - metrics["completed_activities"] - metrics["delayed_activities"] - metrics["at_risk_activities"], metrics["delayed_activities"], metrics["at_risk_activities"]]
    fig.add_trace(go.Pie(labels=labels, values=values, name="Estado"), row=1, col=1)
    fig.add_trace(go.Bar(x=['Planeado', 'Real'], y=[metrics["total_budget"], metrics["actual_cost"]], name="Presupuesto", marker_color=['blue', 'red']), row=1, col=2)
    fig.add_trace(go.Bar(x=['A Tiempo', 'Con Retraso'], y=[metrics["on_time_activities"], metrics["delayed_activities"]], name="Tiempo", marker_color=['green', 'orange']), row=2, col=1)
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    progress = np.linspace(0, metrics["progress_percentage"], len(dates))
    fig.add_trace(go.Scatter(x=dates, y=progress, mode='lines', name="Progreso"), row=2, col=2)
    fig.update_layout(height=600, showlegend=False)
    return fig

def generate_advanced_pdf_report(project_name):
    """Generar informe PDF avanzado (CORREGIDO PARA WINDOWS Y ERRORES)"""
    try:
        if project_name not in st.session_state.project_manager.projects: return None
        project = st.session_state.project_manager.projects[project_name]
        activities = project["activities"]
        metrics = st.session_state.project_manager.calculate_project_metrics(project_name)
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=24, spaceAfter=30, alignment=TA_CENTER, textColor=colors.HexColor('#1a202c'))
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=16, spaceAfter=12, textColor=colors.HexColor('#2b6cb0'))
        
        story = []
        story.append(Paragraph(f"INFORME EJECUTIVO", title_style))
        story.append(Paragraph(f"{project_name}", title_style))
        story.append(Spacer(1, 20))
        
        # Obtener fechas del proyecto de forma segura
        p_start = project.get('project_start_date', 'N/D')
        p_end = project.get('project_end_date', 'N/D')

        story.append(Paragraph("Información General", heading_style))
        project_info = f"""
        <b>Fecha del Informe:</b> {datetime.now().strftime('%d/%m/%Y')}<br/>
        <b>Descripción:</b> {project.get('description', 'N/A')}<br/>
        <b>Inicio Proyecto:</b> {p_start}<br/>
        <b>Fin Proyecto:</b> {p_end}<br/>
        <b>Total Actividades:</b> {metrics['total_activities']}<br/>
        <b>Actividades Completadas:</b> {metrics['completed_activities']}
        """
        story.append(Paragraph(project_info, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Métricas (sin cambios)
        story.append(Paragraph("Métricas del Proyecto", heading_style))
        metrics_data = [['Métrica', 'Valor'], ['Procentaje de Progreso', f"{metrics['progress_percentage']:.1f}%"], ['Presupuesto Total', f"€{metrics['total_budget']:,.2f}"], ['Costo Real', f"€{metrics['actual_cost']:,.2f}"], ['Variación de Costo', f"€{metrics['cost_variance']:,.2f}"], ['Actividades a Tiempo', metrics['on_time_activities']], ['Actividades Retrasadas', metrics['delayed_activities']], ['Actividades en Riesgo', metrics['at_risk_activities']], ['Retraso Promedio (días)', f"{metrics['average_delay_days']:.1f}"]]
        metrics_table = Table(metrics_data, colWidths=[60*mm, 40*mm])
        metrics_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'LEFT'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 10), ('BOTTOMPADDING', (0, 0), (-1, 0), 12), ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')), ('GRID', (0, 0), (-1, -1), 1, colors.grey)]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Gráficos (Matplotlib) - Robusto contra datos vacíos
        story.append(Paragraph("Análisis Gráfico", heading_style))
        fig_mpl, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        status_values = [metrics["completed_activities"], metrics["total_activities"] - metrics["completed_activities"] - metrics["delayed_activities"] - metrics["at_risk_activities"], metrics["delayed_activities"], metrics["at_risk_activities"]]
        if sum(status_values) > 0: ax1.pie(status_values, labels=['Completadas', 'En Progreso', 'Retrasadas', 'En Riesgo'], autopct='%1.1f%%', startangle=90)
        else: ax1.text(0.5, 0.5, "Sin datos", ha='center')
        ax1.set_title('Distribución de Actividades')
        
        ax2.bar(['Planeado', 'Real'], [metrics["total_budget"], metrics["actual_cost"]], color=['blue', 'red'])
        ax2.set_title('Presupuesto vs Costo Real (€)')
        
        if activities:
            df = pd.DataFrame(activities)
            df = df[df['start_date'].astype(bool)].sort_values('start_date')
            if not df.empty:
                df['cumulative_budget'] = df['budget_cost'].cumsum()
                ax3.plot(range(len(df)), df['cumulative_budget'], 'b-', label='Planeado')
                df['cumulative_actual'] = df['actual_cost'].cumsum()
                ax3.plot(range(len(df)), df['cumulative_actual'], 'r-', label='Real')
                ax3.legend()
            else: ax3.text(0.5, 0.5, "Sin fechas válidas", ha='center')
        else: ax3.text(0.5, 0.5, "Sin datos", ha='center')
        ax3.set_title('Curva S de Costos')
        
        if activities:
            df = pd.DataFrame(activities)
            df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
            df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')
            df = df.dropna(subset=['start_date', 'end_date'])
            if not df.empty:
                df['duration_days'] = (df['end_date'] - df['start_date']).dt.days
                top_activities = df.nlargest(5, 'duration_days')
                ax4.barh(top_activities['name'], top_activities['duration_days'])
            else: ax4.text(0.5, 0.5, "Sin fechas válidas", ha='center')
        else: ax4.text(0.5, 0.5, "Sin datos", ha='center')
        ax4.set_title('Top 5 Actividades Más Largas (Días)')
        
        plt.tight_layout()
        
        # Uso seguro de directorio temporal para Windows
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, 'report_charts.png')
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            plt.close(fig_mpl) # Cerrar la figura explícitamente
            
            img = Image(img_path, width=170*mm, height=120*mm)
            story.append(img)
            story.append(Spacer(1, 20))
            
            # Tabla detallada
            story.append(Paragraph("Detalle de Actividades", heading_style))
            table_data = [['ID', 'Actividad', 'Inicio Plan.', 'Fin Plan.', 'Inicio Real', 'Fin Real', 'Progreso', 'Estado']]
            for activity in activities:
                table_data.append([
                    activity.get('id', ''), activity.get('name', '')[:25],
                    activity.get('start_date', '') if activity.get('start_date') else '-',
                    activity.get('end_date', '') if activity.get('end_date') else '-',
                    activity.get('real_start_date', '') if activity.get('real_start_date') else '-',
                    activity.get('real_end_date', '') if activity.get('real_end_date') else '-',
                    f"{activity.get('progress', 0)}%", activity.get('status', 'Pendiente')
                ])
            
            activities_table = Table(table_data, colWidths=[10*mm, 40*mm, 22*mm, 22*mm, 22*mm, 22*mm, 15*mm, 22*mm])
            activities_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('ALIGN', (1, 1), (1, -1), 'LEFT'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, 0), 8), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white]), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))
            story.append(activities_table)
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
            
    except Exception as e:
        st.error(f"Error crítico generando el PDF: {str(e)}")
        return None

# Interfaz principal
def main():
    st.title("🏗️ Gestor de Proyectos Avanzado V4")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Menú Principal")
        page = st.selectbox("Selecciona una opción:", ["🏠 Dashboard", "📊 Gestionar Proyectos", "📈 Plan del Proyecto", "📋 Kanban", "📑 Generar Informe", "⚙️ Configuración"])
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
            with col1: st.metric("Progreso Global", f"{metrics['progress_percentage']:.1f}%", delta=f"{metrics['completed_activities']}/{metrics['total_activities']} tareas")
            with col2: st.metric("Presupuesto", f"€{metrics['total_budget']:,.0f}", delta=f"€{metrics['cost_variance']:,.0f}", delta_color="inverse" if metrics['cost_variance'] > 0 else "normal")
            with col3: st.metric("Actividades a Tiempo", metrics['on_time_activities'], delta=f"{metrics['delayed_activities']} retrasadas")
            with col4: st.metric("Retraso Promedio", f"{metrics['average_delay_days']:.1f} días", delta="Días")
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                fig = create_gantt_chart(selected_project)
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.info("Añade actividades con fechas válidas para ver el diagrama de Gantt.")
            with col2:
                fig = create_s_curve(selected_project)
                if fig: st.plotly_chart(fig, use_container_width=True)
                else: st.info("Añade actividades con fechas y costes para ver la Curva S.")
            
            st.markdown("---")
            st.subheader("📈 Análisis Detallado de KPIs")
            fig = create_kpi_dashboard(selected_project)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("No hay datos suficientes para el análisis detallado.")
        else: st.info("Por favor, selecciona un proyecto para ver el dashboard")
    
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
                    else: st.error("El proyecto ya existe")
        
        with tab2:
            st.subheader("📋 Lista de Proyectos")
            if projects:
                for project_name in projects:
                    with st.expander(f"📁 {project_name}"):
                        project = st.session_state.project_manager.projects[project_name]
                        metrics = st.session_state.project_manager.calculate_project_metrics(project_name)
                        col1, col2, col3 = st.columns(3)
                        # Mostrar fechas del proyecto si existen
                        p_start = project.get('project_start_date', 'N/D')
                        p_end = project.get('project_end_date', 'N/D')
                        with col1:
                            st.write(f"**Descripción:** {project.get('description', 'N/A')}")
                            st.caption(f"Inicio: {p_start} | Fin: {p_end}")
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
            else: st.info("No hay proyectos creados")
        
        with tab3:
            if selected_project:
                st.subheader(f"✏️ Editar Proyecto: {selected_project}")
                project = st.session_state.project_manager.projects[selected_project]

                # --- FORMULARIO 1: ACTUALIZAR DETALLES DEL PROYECTO (Descripción y Fechas Globales) ---
                with st.form("update_project_details_form"):
                    st.markdown("#### 📝 Detalles Generales del Proyecto")
                    new_description = st.text_area("Descripción", value=project.get('description', ''), key="desc_input")
                    
                    # Preparar fechas por defecto (manejar si están vacías en DB)
                    try: def_start = datetime.strptime(project.get('project_start_date'), "%Y-%m-%d")
                    except: def_start = None
                    try: def_end = datetime.strptime(project.get('project_end_date'), "%Y-%m-%d")
                    except: def_end = None

                    col_p1, col_p2 = st.columns(2)
                    with col_p1: new_project_start = st.date_input("Fecha Inicio Proyecto", value=def_start, key="p_start_input")
                    with col_p2: new_project_end = st.date_input("Fecha Fin Proyecto", value=def_end, key="p_end_input")

                    submitted_details = st.form_submit_button("Actualizar Detalles del Proyecto")

                    if submitted_details:
                        if st.session_state.project_manager.update_project_details(selected_project, new_description, new_project_start, new_project_end):
                             st.success("✅ Detalles del proyecto actualizados")
                             st.rerun()
                        else: st.error("Error al actualizar los detalles")

                st.markdown("---")

                # --- SECCIÓN DE EDICIÓN DE ACTIVIDAD ---
                if st.session_state.editing_activity_id:
                    st.markdown("#### ✏️ Editando Actividad")
                    activity_to_edit = next((act for act in project["activities"] if act["id"] == st.session_state.editing_activity_id), None)
                    
                    if activity_to_edit:
                        with st.form("edit_activity_form"):
                            col1, col2 = st.columns(2)
                            with col1:
                                edit_name = st.text_input("Nombre*", value=activity_to_edit["name"], key="edit_name")
                                edit_group = st.selectbox("Grupo", ["INGENIERÍA", "OBRA CIVIL", "ELECTROMECÁNICO", "SUMINISTROS", "OTROS"], index=["INGENIERÍA", "OBRA CIVIL", "ELECTROMECÁNICO", "SUMINISTROS", "OTROS"].index(activity_to_edit["group"]), key="edit_group")
                            with col2:
                                try: edit_start = st.date_input("Inicio Planificado", value=datetime.strptime(activity_to_edit["start_date"], "%Y-%m-%d"), key="edit_start")
                                except: edit_start = st.date_input("Inicio Planificado", key="edit_start")
                                try: edit_end = st.date_input("Fin Planificado", value=datetime.strptime(activity_to_edit["end_date"], "%Y-%m-%d"), key="edit_end")
                                except: edit_end = st.date_input("Fin Planificado", key="edit_end")
                            
                            col3, col4 = st.columns(2)
                            with col3:
                                try: def_r_start = datetime.strptime(activity_to_edit["real_start_date"], "%Y-%m-%d")
                                except: def_r_start = None
                                try: def_r_end = datetime.strptime(activity_to_edit["real_end_date"], "%Y-%m-%d")
                                except: def_r_end = None
                                edit_real_start = st.date_input("Inicio Real", value=def_r_start, key="edit_real_start")
                                edit_real_end = st.date_input("Fin Real", value=def_r_end, key="edit_real_end")
                            with col4:
                                # Slider de Progreso - Determina el estado automáticamente
                                edit_progress = st.slider("Progreso (%)", 0, 100, activity_to_edit["progress"], key="edit_progress")
                                # Se calcula el nuevo estado basado en el slider
                                new_auto_status = determine_status(edit_progress)
                                st.info(f"Estado automático: **{new_auto_status}**")

                            col5, col6 = st.columns(2)
                            with col5: edit_budget = st.number_input("Presupuesto", min_value=0.0, value=activity_to_edit["budget_cost"], key="edit_budget")
                            with col6: edit_actual = st.number_input("Costo Real", min_value=0.0, value=activity_to_edit["actual_cost"], key="edit_actual")

                            submitted_edit = st.form_submit_button("Guardar Cambios")
                            
                            if submitted_edit:
                                updates = {
                                    "name": edit_name, "group": edit_group,
                                    "start_date": edit_start.strftime("%Y-%m-%d"), "end_date": edit_end.strftime("%Y-%m-%d"),
                                    "real_start_date": edit_real_start.strftime("%Y-%m-%d") if edit_real_start else "",
                                    "real_end_date": edit_real_end.strftime("%Y-%m-%d") if edit_real_end else "",
                                    "progress": edit_progress,
                                    "status": new_auto_status, # Usamos el estado calculado automáticamente
                                    "budget_cost": edit_budget, "actual_cost": edit_actual
                                }
                                if st.session_state.project_manager.update_activity(selected_project, st.session_state.editing_activity_id, updates):
                                    st.success("Actividad actualizada.")
                                    st.session_state.editing_activity_id = None
                                    st.rerun()
                                else: st.error("Error al actualizar.")
                        
                        if st.button("Cancelar Edición"):
                            st.session_state.editing_activity_id = None
                            st.rerun()
                    st.markdown("---")

                # --- FORMULARIO 2: AÑADIR NUEVA ACTIVIDAD (Estado Automático) ---
                with st.form("add_activity_form"):
                    st.markdown("#### ➕ Añadir Nueva Actividad")
                    col1, col2 = st.columns(2)
                    with col1:
                        activity_name = st.text_input("Nombre de la Actividad*", key="act_name")
                        activity_group = st.selectbox("Grupo", ["INGENIERÍA", "OBRA CIVIL", "ELECTROMECÁNICO", "SUMINISTROS", "OTROS"], key="act_group")
                    with col2:
                        start_date = st.date_input("Inicio Planificado", key="act_start")
                        end_date = st.date_input("Fin Planificado", key="act_end")
                    
                    col_real1, col_real2 = st.columns(2)
                    with col_real1: real_start_date = st.date_input("Inicio Real (Opcional)", value=None, key="act_real_start")
                    with col_real2: 
                        # Slider de Progreso
                        progress = st.slider("Progreso (%)", 0, 100, 0, key="act_progress")
                        # Determinar estado automáticamente
                        auto_status = determine_status(progress)
                        st.info(f"Estado inicial: **{auto_status}**")

                    col3, col4 = st.columns(2)
                    with col3:
                        budget_cost = st.number_input("Presupuesto", min_value=0.0, value=0.0, key="act_budget")
                        actual_cost = st.number_input("Costo Real", min_value=0.0, value=0.0, key="act_actual")
                    with col4:
                        weight = st.number_input("Peso", min_value=1, value=1, key="act_weight")

                    submitted_activity = st.form_submit_button("Añadir Actividad")

                    if submitted_activity:
                        if activity_name:
                            r_start = real_start_date.strftime("%Y-%m-%d") if real_start_date else ""
                            r_end = datetime.now().strftime("%Y-%m-%d") if progress == 100 else ""
                            
                            new_activity = {
                                "name": activity_name, "group": activity_group,
                                "start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d"),
                                "progress": progress,
                                "status": auto_status, # Usamos el estado calculado
                                "budget_cost": budget_cost, "actual_cost": actual_cost, "weight": weight,
                                "real_start_date": r_start, "real_end_date": r_end
                            }
                            if st.session_state.project_manager.add_activity(selected_project, new_activity):
                                st.success(f"✅ Actividad '{activity_name}' añadida exitosamente")
                                st.rerun()
                            else: st.error("❌ Hubo un error al añadir la actividad.")
                        else: st.error("⚠️ Por favor, introduce un nombre para la actividad.")

                # --- LISTA DE ACTIVIDADES EXISTENTES ---
                st.markdown("---")
                st.markdown("#### 📋 Actividades Existentes (Gestión)")
                if project["activities"]:
                    for activity in project["activities"]:
                        col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 2, 1, 1, 1])
                        with col1:
                            st.write(f"**{activity['name']}**")
                            st.caption(f"Grupo: {activity['group']}")
                        with col2: st.write(f"Plan: {activity['start_date']} -> {activity['end_date']}")
                        with col3:
                            r_start = activity.get('real_start_date') if activity.get('real_start_date') else "(sin inicio)"
                            r_end = activity.get('real_end_date') if activity.get('real_end_date') else "(sin fin)"
                            st.write(f"Real: {r_start} -> {r_end}")
                        with col4:
                            st.write(f"{activity['progress']}%")
                            st.caption(activity['status'])
                        with col5:
                            if st.button("✏️", key=f"edit_{activity['id']}"):
                                st.session_state.editing_activity_id = activity['id']
                                st.rerun()
                        with col6:
                            if st.button("🗑️", key=f"del_act_{activity['id']}"):
                                if st.session_state.project_manager.delete_activity(selected_project, activity['id']):
                                    st.success("Actividad eliminada")
                                    if st.session_state.editing_activity_id == activity['id']:
                                        st.session_state.editing_activity_id = None
                                    st.rerun()
                        st.divider()
                else: st.info("No hay actividades en este proyecto")
            else: st.warning("Por favor, selecciona un proyecto para editar")
    
    elif page == "📈 Plan del Proyecto":
        st.header("📈 Plan del Proyecto")
        if selected_project:
            st.subheader("📊 Diagrama de Gantt")
            st.caption("Prioriza fechas REALES si existen.")
            fig = create_gantt_chart(selected_project)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Añade actividades con fechas válidas.")
            
            st.subheader("📈 Curva S de Progreso")
            fig = create_s_curve(selected_project)
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.info("Añade actividades con fechas planificadas y costes.")
            
            st.subheader("📋 Tabla de Planificación Detallada")
            project = st.session_state.project_manager.projects[selected_project]
            if project["activities"]:
                df = pd.DataFrame(project["activities"])
                # Lógica de retraso simplificada para visualización
                today = datetime.now()
                def calculate_delay_status(row):
                    try:
                        end_dt = datetime.strptime(row['end_date'], '%Y-%m-%d')
                        if row['progress'] < 100 and today > end_dt: return "Retrasado"
                        return row['status']
                    except: return row['status']
                df['estado_calc'] = df.apply(calculate_delay_status, axis=1)
                cols = ['name', 'group', 'start_date', 'end_date', 'real_start_date', 'real_end_date', 'progress', 'estado_calc']
                st.dataframe(df[cols].rename(columns={'estado_calc': 'Estado (Tiempo)'}), use_container_width=True)
            else: st.info("No hay actividades planificadas")
        else: st.warning("Por favor, selecciona un proyecto")
    
    elif page == "📋 Kanban":
        st.header("📋 Tablero Kanban")
        if selected_project:
            project = st.session_state.project_manager.projects[selected_project]
            if project["activities"]:
                # Clasificación simple basada en el estado guardado
                states = {"Pendiente": [], "En Progreso": [], "Completado": []}
                # Estados adicionales calculados (simplificado)
                today = datetime.now()
                retrasadas, en_riesgo = [], []

                for a in project["activities"]:
                    # Clasificación base
                    if a["status"] in states: states[a["status"]].append(a)
                    
                    # Cálculos de riesgo/retraso si no está completada
                    if a["progress"] < 100 and a["end_date"]:
                        try:
                            end_dt = datetime.strptime(a["end_date"], '%Y-%m-%d')
                            if today > end_dt: retrasadas.append(a)
                            elif (end_dt - today).days <= 7: en_riesgo.append(a)
                        except: pass

                col1, col2, col3, col4, col5 = st.columns(5)
                def render_card(activity, bg_color):
                     st.markdown(f"""<div style="background-color: {bg_color}; padding: 10px; border-radius: 5px; margin: 5px 0;"><strong>{activity['name']}</strong><br/><small>{activity['group']} | {activity['progress']}%</small></div>""", unsafe_allow_html=True)

                with col1:
                    st.markdown("### 🔵 Pendiente"); [render_card(a, "#e3f2fd") for a in states["Pendiente"]]
                with col2:
                    st.markdown("### 🟡 En Progreso"); [render_card(a, "#fff9c4") for a in states["En Progreso"]]
                with col3:
                    st.markdown("### 🟢 Completado"); [render_card(a, "#e8f5e9") for a in states["Completado"]]
                with col4:
                    st.markdown("### 🔴 Retrasado (Calc.)"); [render_card(a, "#ffebee") for a in retrasadas]
                with col5:
                    st.markdown("### 🟠 En Riesgo (<7 días)"); [render_card(a, "#fff3e0") for a in en_riesgo]
            else: st.info("No hay actividades en este proyecto")
        else: st.warning("Por favor, selecciona un proyecto")
    
    elif page == "📑 Generar Informe":
        st.header("📑 Generador de Informes")
        if selected_project:
            st.subheader(f"📊 Informe: {selected_project}")
            report_format = st.selectbox("Formato", ["PDF", "Excel"])
            
            if st.button("🚀 Generar Informe"):
                with st.spinner("Generando..."):
                    if report_format == "PDF":
                        pdf_data = generate_advanced_pdf_report(selected_project)
                        if pdf_data:
                            st.success("✅ PDF generado")
                            st.download_button("📥 Descargar PDF", data=pdf_data, file_name=f"Informe_{selected_project}_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf")
                    else:
                        # Generar Excel (Simplificado)
                        project = st.session_state.project_manager.projects[selected_project]
                        df = pd.DataFrame(project["activities"])
                        buffer = io.BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            df.to_excel(writer, sheet_name='Actividades', index=False)
                        st.success("✅ Excel generado")
                        st.download_button("📥 Descargar Excel", data=buffer.getvalue(), file_name=f"Informe_{selected_project}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else: st.warning("Selecciona un proyecto")
    
    elif page == "⚙️ Configuración":
        st.header("⚙️ Configuración y Datos")
        tab1, tab2 = st.tabs(["Importar Excel", "Acerca de"])
        with tab1:
            st.markdown("#### 📊 Importar Actividades desde Excel")
            st.caption("Arrastra tu archivo Excel. Debe tener columnas como 'Tarea', 'Inicio', 'Fin', 'Progreso'.")
            excel_file = st.file_uploader("Archivo Excel", type=["xlsx", "xls"])
            if excel_file and selected_project:
                if st.button(f"Importar a '{selected_project}'"):
                    try:
                        df = pd.read_excel(excel_file)
                        count = 0
                        for index, row in df.iterrows():
                            start_d = str(row.get('Inicio', datetime.now().strftime('%Y-%m-%d')))[:10]
                            end_d = str(row.get('Fin', datetime.now().strftime('%Y-%m-%d')))[:10]
                            progress_val = int(row.get('Progreso', 0))
                            activity = {
                                "name": str(row.get('Tarea', f'Actividad {index+1}')),
                                "group": str(row.get('Grupo', 'General')),
                                "start_date": start_d, "end_date": end_d,
                                "progress": progress_val,
                                "status": determine_status(progress_val),
                                "budget_cost": float(row.get('Presupuesto', 0)), "actual_cost": float(row.get('Coste Real', 0)), "weight": int(row.get('Peso', 1)),
                                "real_start_date": "", "real_end_date": ""
                            }
                            if st.session_state.project_manager.add_activity(selected_project, activity): count +=1
                        st.success(f"✅ Se importaron {count} actividades a '{selected_project}'.")
                        st.rerun()
                    except Exception as e: st.error(f"Error en importación: {e}")
            elif excel_file and not selected_project: st.warning("Primero selecciona un proyecto en el menú lateral para importar.")
        with tab3: st.markdown("### Gestor de Proyectos V4 (Final) \n Base de datos SQLite y generación robusta de informes.")

if __name__ == "__main__":
    main()