#!/usr/bin/env python3
"""
Generate Yatra Architecture Diagram
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors
COLOR_FRONTEND = '#4A90E2'
COLOR_BACKEND = '#50C878'
COLOR_AGENT = '#F5A623'
COLOR_DATA = '#BD10E0'
COLOR_EXTERNAL = '#E74C3C'
COLOR_TOOLS = '#9B59B6'

def draw_box(ax, x, y, width, height, text, color, text_size=10, subtext=''):
    """Draw a rounded box with text"""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2,
        alpha=0.8
    )
    ax.add_patch(box)

    # Main text
    ax.text(x + width/2, y + height/2 + 0.15, text,
            ha='center', va='center',
            fontsize=text_size, fontweight='bold',
            color='white')

    # Subtext
    if subtext:
        ax.text(x + width/2, y + height/2 - 0.25, subtext,
                ha='center', va='center',
                fontsize=8,
                color='white', style='italic')

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow with optional label"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color='black',
        linewidth=2,
        mutation_scale=20
    )
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label,
                ha='center', va='bottom',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Title
ax.text(8, 11.5, 'YATRA TRAVEL AGENT - ARCHITECTURE',
        ha='center', va='top',
        fontsize=18, fontweight='bold')
ax.text(8, 11, 'Hybrid AI Travel Recommendation System',
        ha='center', va='top',
        fontsize=12, style='italic', color='gray')

# ==================== USER LAYER ====================
draw_box(ax, 6.5, 9.5, 3, 0.8, 'User', COLOR_FRONTEND, 11, 'Web Browser')

# ==================== FRONTEND LAYER ====================
draw_box(ax, 5.5, 7.8, 5, 1.2, 'Frontend - Web Interface', COLOR_FRONTEND, 11,
         'index.html | JavaScript | jsPDF')

# Frontend components
draw_box(ax, 5.7, 7.0, 2.2, 0.5, 'Form Input', COLOR_FRONTEND, 9)
draw_box(ax, 8.1, 7.0, 2.2, 0.5, 'Results Display', COLOR_FRONTEND, 9)

# ==================== BACKEND LAYER ====================
draw_box(ax, 5.5, 5.2, 5, 1.2, 'Backend - REST API', COLOR_BACKEND, 11,
         'web_server.py | Flask | CORS')

# Backend components
draw_box(ax, 5.7, 4.4, 1.5, 0.5, 'Session Mgmt', COLOR_BACKEND, 8)
draw_box(ax, 7.4, 4.4, 1.5, 0.5, 'Pagination', COLOR_BACKEND, 8)
draw_box(ax, 9.1, 4.4, 1.2, 0.5, 'Cache', COLOR_BACKEND, 8)

# ==================== AGENT CORE ====================
draw_box(ax, 3, 2.2, 5.5, 1.5, 'AI Agent Core', COLOR_AGENT, 12,
         'yatra_agent.py | DestinationSuggester')

# Agent pipeline stages
draw_box(ax, 3.2, 1.3, 2.4, 0.6, 'Quantitative Filter', COLOR_AGENT, 9,
         'Type, Rating, Budget, Duration')
draw_box(ax, 5.8, 1.3, 2.4, 0.6, 'Qualitative Filter', COLOR_AGENT, 9,
         'City, Family-Friendly')

# ==================== TOOLS LAYER ====================
draw_box(ax, 9, 2.2, 4, 1.5, 'Agent Tools', COLOR_TOOLS, 11,
         'tools.py | TravelTools')

# Tool components
draw_box(ax, 9.2, 1.3, 1.7, 0.6, 'City Filter', COLOR_TOOLS, 8,
         'LLM + Search')
draw_box(ax, 11.1, 1.3, 1.7, 0.6, 'Family Filter', COLOR_TOOLS, 8,
         'LLM + Search')

# ==================== DATA LAYER ====================
draw_box(ax, 0.5, 2.2, 2, 1.5, 'Data Handler', COLOR_DATA, 10,
         'dataset_handler.py')

draw_box(ax, 0.7, 0.8, 1.6, 0.8, 'Dataset', COLOR_DATA, 9,
         'destinations.csv\n10K+ Places')

# ==================== EXTERNAL SERVICES ====================
draw_box(ax, 11, 5.2, 2, 1.2, 'External Services', COLOR_EXTERNAL, 10)

# External service components
draw_box(ax, 11.2, 4.0, 1.6, 0.5, 'Google Gemini', COLOR_EXTERNAL, 8,
         '2.5 Flash Lite')
draw_box(ax, 11.2, 3.3, 1.6, 0.5, 'Google Search', COLOR_EXTERNAL, 8,
         'Grounding')

# ==================== SYSTEM PROMPTS ====================
draw_box(ax, 13.5, 2.2, 2, 1.0, 'System Prompts', COLOR_TOOLS, 9,
         'system_prompts.py\nLLM Instructions')

# ==================== ARROWS - DATA FLOW ====================

# User to Frontend
draw_arrow(ax, 8, 9.5, 8, 9.0, 'Input Query')

# Frontend to Backend
draw_arrow(ax, 8, 7.8, 8, 6.4, 'POST /api/suggest')
draw_arrow(ax, 8, 6.4, 8, 7.8, 'JSON Response', '<-')

# Backend to Agent
draw_arrow(ax, 8, 5.2, 6.5, 3.7, 'Process Request')
draw_arrow(ax, 6.5, 3.7, 8, 5.2, 'Filtered Results', '<-')

# Agent to Data Handler
draw_arrow(ax, 3, 2.9, 1.5, 3.3, 'Query Dataset', '<->')

# Data Handler to Dataset
draw_arrow(ax, 1.3, 2.2, 1.3, 1.6, 'Pandas Filter', '<->')

# Agent to Tools
draw_arrow(ax, 8.5, 2.9, 9, 2.9, 'Execute Tools')

# Tools to External Services
draw_arrow(ax, 11, 3.0, 11, 4.0, 'API Calls', '<->')

# Tools to System Prompts
draw_arrow(ax, 13, 2.7, 13.5, 2.7, 'Get Prompts', '<-')

# Agent pipeline internal flow
draw_arrow(ax, 5.6, 1.6, 5.8, 1.6, '')

# ==================== ANNOTATIONS ====================

# Pipeline flow annotation
ax.text(5.5, 0.5, 'FILTERING PIPELINE', fontsize=9, fontweight='bold')
ax.text(5.5, 0.2, '1. Quantitative (Pandas) → 2. Qualitative (LLM+Search) → 3. Recommendations',
        fontsize=7, style='italic')

# Key features
features_text = '''KEY FEATURES:
• Hybrid Intelligence (Deterministic + LLM)
• Web-Grounded Validation
• Session Management & Caching
• Cancellation Support
• PDF Export'''

ax.text(0.5, 5.5, features_text, fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Tech stack
tech_text = '''TECH STACK:
• Python 3.x
• Flask REST API
• Google Gemini 2.5
• Pandas/NumPy
• Google Search API'''

ax.text(13.5, 5.5, tech_text, fontsize=8,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# Data flow key
flow_text = '''DATA FLOW:
User Input → Frontend Validation →
Backend Session → Agent Pipeline →
Pandas Filter + LLM Analysis →
Grounded Results → Pagination →
Frontend Display'''

ax.text(0.5, 9.8, flow_text, fontsize=7,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=COLOR_FRONTEND, label='Frontend Layer'),
    mpatches.Patch(facecolor=COLOR_BACKEND, label='Backend Layer'),
    mpatches.Patch(facecolor=COLOR_AGENT, label='AI Agent Core'),
    mpatches.Patch(facecolor=COLOR_TOOLS, label='Tools & Prompts'),
    mpatches.Patch(facecolor=COLOR_DATA, label='Data Layer'),
    mpatches.Patch(facecolor=COLOR_EXTERNAL, label='External Services')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig('/home/user/Yatra/yatra_architecture_diagram.png', dpi=300, bbox_inches='tight')
print("Architecture diagram generated: yatra_architecture_diagram.png")
