"""
Templates Module for ASU Tapeout Agent
=====================================

Provides pre-validated RTL templates, SDC templates, and ORFS configurations.
"""

from pathlib import Path

# Template directories
TEMPLATE_DIR = Path(__file__).parent
RTL_TEMPLATES_DIR = TEMPLATE_DIR / "rtl_templates"
SDC_TEMPLATES_DIR = TEMPLATE_DIR / "sdc_templates"
ORFS_CONFIGS_DIR = TEMPLATE_DIR / "orfs_configs"

# Template categories
RTL_TEMPLATE_CATEGORIES = {
    'state_machines': RTL_TEMPLATES_DIR / 'state_machines',
    'arithmetic': RTL_TEMPLATES_DIR / 'arithmetic',
    'dsp': RTL_TEMPLATES_DIR / 'dsp',
    'fixed_point': RTL_TEMPLATES_DIR / 'fixed_point'
}

__all__ = [
    'TEMPLATE_DIR',
    'RTL_TEMPLATES_DIR',
    'SDC_TEMPLATES_DIR',
    'ORFS_CONFIGS_DIR',
    'RTL_TEMPLATE_CATEGORIES'
] 