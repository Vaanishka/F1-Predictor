"""
Config file for F1 Predictor scripts
Save this as: script/config.py
"""

# ============================================================================
# COLORS
# ============================================================================

COLORS = {
    'background': '#0A0A0A',
    'primary': '#E10600',
    'card_bg': '#1A1A1A',
    'glass_bg': 'rgba(26, 26, 26, 0.6)',
    'text': '#FFFFFF',
    'text_muted': 'rgba(255,255,255,0.6)',
    'success': '#00D26A',
    'warning': '#FFA500',
    'danger': '#E10600',
    'border': 'rgba(255,255,255,0.1)',
}

# ============================================================================
# TEAM COLORS (2024)
# ============================================================================

TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'RB': '#6692FF',
    'Kick Sauber': '#52E252',
    'Haas F1 Team': '#B6BABD',
    'Sauber': '#52E252',
    'AlphaTauri': '#6692FF',
    'Alfa Romeo': '#52E252',
}

# ============================================================================
# DRIVER TEAMS 2024
# ============================================================================

DRIVER_TEAMS_2024 = {
    'VER': 'Red Bull Racing',
    'PER': 'Red Bull Racing',
    'LEC': 'Ferrari',
    'SAI': 'Ferrari',
    'HAM': 'Mercedes',
    'RUS': 'Mercedes',
    'NOR': 'McLaren',
    'PIA': 'McLaren',
    'ALO': 'Aston Martin',
    'STR': 'Aston Martin',
    'GAS': 'Alpine',
    'OCO': 'Alpine',
    'ALB': 'Williams',
    'SAR': 'Williams',
    'TSU': 'RB',
    'RIC': 'RB',
    'BOT': 'Kick Sauber',
    'ZHO': 'Kick Sauber',
    'MAG': 'Haas F1 Team',
    'HUL': 'Haas F1 Team',
    'BEA': 'Haas F1 Team',
    'LAW': 'RB',
    'COL': 'Williams',
}

# ============================================================================
# DRIVER TEAMS 2025
# ============================================================================

DRIVER_TEAMS_2025 = {
    'VER': 'Red Bull Racing',
    'PER': 'Red Bull Racing',
    'LEC': 'Ferrari',
    'HAM': 'Ferrari',
    'RUS': 'Mercedes',
    'ANT': 'Mercedes',
    'NOR': 'McLaren',
    'PIA': 'McLaren',
    'ALO': 'Aston Martin',
    'STR': 'Aston Martin',
    'GAS': 'Alpine',
    'DRU': 'Alpine',
    'ALB': 'Williams',
    'SAI': 'Williams',
    'TSU': 'RB',
    'LAW': 'RB',
    'BOT': 'Kick Sauber',
    'HUL': 'Kick Sauber',
    'OCO': 'Haas F1 Team',
    'BEA': 'Haas F1 Team',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_driver_team(driver_code, year=2024):
    """Get team for a driver in a specific year"""
    if year == 2024:
        return DRIVER_TEAMS_2024.get(driver_code, 'Haas F1 Team')
    elif year == 2025:
        return DRIVER_TEAMS_2025.get(driver_code, 'Haas F1 Team')
    else:
        return DRIVER_TEAMS_2024.get(driver_code, 'Haas F1 Team')

def get_team_color(team_name):
    """Get color for a team"""
    return TEAM_COLORS.get(team_name, '#B6BABD')