"""
Generate thesis figures for dataset visualization and system understanding.

Figures:
1. System-level relationship diagram (conceptual)
2. Time-series alignment (load vs PV vs battery) - 1-2 weeks
3. PV feature relationships (scatter plots, correlation heatmap)
4. Load behavior by family type (average daily profiles)
5. Battery operation logic (constraint-based diagram)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import seaborn as sns
from pathlib import Path
from typing import Optional

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def figure1_system_diagram(output_path: Path):
    """
    Figure 1: System-level relationship diagram showing forecast → optimization → MPC flow.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    data_color = '#4A90E2'
    process_color = '#50C878'
    control_color = '#FF6B6B'
    battery_color = '#FFD93D'
    
    # Exogenous inputs (left side)
    ax.text(1, 8.5, 'Exogenous Inputs', fontsize=14, fontweight='bold', ha='center')
    
    load_box = FancyBboxPatch((0.3, 7), 1.4, 0.8, boxstyle="round,pad=0.1", 
                               facecolor=data_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(load_box)
    ax.text(1, 7.4, 'Load\nProfiles\n(LPG)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    pv_box = FancyBboxPatch((0.3, 5.5), 1.4, 0.8, boxstyle="round,pad=0.1", 
                            facecolor=data_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(pv_box)
    ax.text(1, 5.9, 'PV\nGeneration\n(PVGIS)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    weather_box = FancyBboxPatch((0.3, 4), 1.4, 0.8, boxstyle="round,pad=0.1", 
                                  facecolor=data_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(weather_box)
    ax.text(1, 4.4, 'Weather\n(OpenWeather)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Forecasting block (center-left)
    forecast_box = FancyBboxPatch((3.5, 5.5), 1.8, 1.5, boxstyle="round,pad=0.15", 
                                  facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(forecast_box)
    ax.text(4.4, 6.5, 'Forecasting\nModels', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(4.4, 6.1, '• Load forecast\n• PV forecast', ha='center', va='center', fontsize=9)
    
    # Optimization block (center)
    opt_box = FancyBboxPatch((6, 5.5), 1.8, 1.5, boxstyle="round,pad=0.15", 
                              facecolor=process_color, edgecolor='black', linewidth=2)
    ax.add_patch(opt_box)
    ax.text(6.9, 6.5, 'Day-ahead\nOptimization', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(6.9, 6.1, '• Cost minimization\n• Battery dispatch', ha='center', va='center', fontsize=9)
    
    # MPC block (center-right)
    mpc_box = FancyBboxPatch((8.5, 5.5), 1.2, 1.5, boxstyle="round,pad=0.15", 
                              facecolor=control_color, edgecolor='black', linewidth=2)
    ax.add_patch(mpc_box)
    ax.text(9.1, 6.5, 'MPC\nControl', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(9.1, 6.1, 'Real-time\nadjustment', ha='center', va='center', fontsize=9)
    
    # Battery (right side)
    battery_box = FancyBboxPatch((8.2, 2.5), 1.8, 1.2, boxstyle="round,pad=0.15", 
                                 facecolor=battery_color, edgecolor='black', linewidth=2)
    ax.add_patch(battery_box)
    ax.text(9.1, 3.2, 'Shared\nBattery', ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(9.1, 2.8, 'SoC, P_ch, P_dis', ha='center', va='center', fontsize=9)
    
    # Grid (bottom)
    grid_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, boxstyle="round,pad=0.1", 
                              facecolor='#E0E0E0', edgecolor='black', linewidth=1.5)
    ax.add_patch(grid_box)
    ax.text(5, 0.9, 'Grid Exchange (P_grid)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows: Data → Forecasting
    ax.arrow(1.7, 7.4, 1.6, -0.2, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)
    ax.arrow(1.7, 5.9, 1.6, 0.2, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)
    ax.arrow(1.7, 4.4, 1.6, 1.1, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)
    
    # Arrow: Forecasting → Optimization
    ax.arrow(5.3, 6.25, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Arrow: Optimization → MPC
    ax.arrow(7.8, 6.25, 0.5, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Arrow: MPC → Battery
    ax.arrow(9.1, 5.5, 0, -0.6, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)
    
    # Arrow: Battery → Grid
    ax.arrow(8.2, 2.5, -1.2, -1.2, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)
    ax.text(6.5, 1.2, 'P_grid (kW)', ha='center', fontsize=9, style='italic')
    
    # Arrow: Optimization → Grid
    ax.arrow(6.9, 5.5, -1.4, -4.2, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=1.5)
    
    # Feedback arrow: Battery → MPC (dashed)
    ax.arrow(8.2, 3.1, -0.5, 2.2, head_width=0.15, head_length=0.1, 
             fc='black', ec='black', linewidth=1.5, linestyle='--', alpha=0.6)
    ax.text(7.5, 4.5, 'SoC\nfeedback', ha='center', fontsize=8, style='italic')
    
    # Title
    ax.text(5, 9.5, 'Energy Management System Framework', 
            ha='center', fontsize=16, fontweight='bold')
    ax.text(5, 9.1, 'Forecast-based Optimization and MPC Control', 
            ha='center', fontsize=12, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ok] Figure 1 saved: {output_path}")


def figure2_timeseries_alignment(df: pd.DataFrame, output_path: Path, 
                                  start_date: Optional[str] = None,
                                  weeks: int = 2):
    """
    Figure 2: Time-series alignment showing load, PV, and net load over 1-2 weeks.
    Uses ONLY real data from the dataset (no simulated values).
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    
    # Select period (summer week for high PV)
    if start_date is None:
        # Find a summer week with good PV
        summer_mask = (df.index.month >= 6) & (df.index.month <= 8)
        summer_df = df[summer_mask]
        if len(summer_df) > 0:
            # Find day with high PV
            high_pv_day = summer_df['PV_true'].idxmax()
            start_date = high_pv_day.strftime('%Y-%m-%d')
        else:
            start_date = df.index[0].strftime('%Y-%m-%d')
    
    start = pd.to_datetime(start_date)
    end = start + pd.Timedelta(weeks=weeks)
    plot_df = df.loc[start:end].copy()
    
    if len(plot_df) == 0:
        print(f"[warn] No data for period {start_date} to {end}, using first available period")
        plot_df = df.iloc[:weeks*7*24].copy()
    
    # Calculate net load (real data: load - PV)
    plot_df['net_load'] = plot_df['total_load'] - plot_df['PV_true']
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Total Load (real data from LPG)
    ax1.plot(plot_df.index, plot_df['total_load'], color='#FF6B6B', linewidth=2, label='Total Load')
    ax1.fill_between(plot_df.index, 0, plot_df['total_load'], alpha=0.3, color='#FF6B6B')
    ax1.set_ylabel('Load (kWh/h)', fontsize=12, fontweight='bold')
    ax1.set_title('Building Total Load', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # Plot 2: PV Generation (real data from PVGIS)
    ax2.plot(plot_df.index, plot_df['PV_true'], color='#4ECDC4', linewidth=2, label='PV Generation')
    ax2.fill_between(plot_df.index, 0, plot_df['PV_true'], alpha=0.3, color='#4ECDC4')
    ax2.set_ylabel('PV Power (kW)', fontsize=12, fontweight='bold')
    ax2.set_title('PV Generation', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Plot 3: Net Load (real calculation: load - PV)
    ax3.plot(plot_df.index, plot_df['net_load'], color='#9B59B6', linewidth=2, label='Net Load (Load - PV)')
    ax3.fill_between(plot_df.index, 0, plot_df['net_load'], alpha=0.3, color='#9B59B6', where=(plot_df['net_load'] >= 0))
    ax3.fill_between(plot_df.index, 0, plot_df['net_load'], alpha=0.3, color='#4ECDC4', where=(plot_df['net_load'] < 0))
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Net Load (kWh/h)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax3.set_title('Net Load (Positive = Deficit, Negative = Excess PV)', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # Format x-axis
    ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    fig.suptitle(f'Time-Series Alignment: Load, PV, and Net Load ({weeks} weeks)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ok] Figure 2 saved: {output_path}")


def figure3_pv_feature_relationships(df: pd.DataFrame, output_path_dir: Path):
    """
    Figure 3: PV feature relationships (scatter plots + correlation heatmap).
    Creates multiple subplots.
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    
    # Filter to meaningful cases: daytime hours with actual PV generation
    # This removes misleading night-time zeros and focuses on real relationships
    pv_df = df[(df['PV_true'] > 0.1) & (df['hour'] >= 6) & (df['hour'] <= 18)].copy()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Scatter 1: PV vs Gb (direct irradiance) - REAL DATA ONLY
    # This is the PRIMARY driver of PV output (strong correlation)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(pv_df['Gb'], pv_df['PV_true'], alpha=0.5, s=10, color='#4A90E2')
    ax1.set_xlabel('Direct Irradiance Gb (W/m²)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('PV Power (kW)', fontsize=11, fontweight='bold')
    ax1.set_title('PV vs Direct Irradiance\n(Primary driver)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    # Add correlation
    corr = pv_df[['Gb', 'PV_true']].corr().iloc[0, 1]
    ax1.text(0.05, 0.95, f'ρ = {corr:.3f}\n(strong)', transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter 2: PV vs Gd (diffuse irradiance) - REAL DATA ONLY
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(pv_df['Gd'], pv_df['PV_true'], alpha=0.5, s=10, color='#50C878')
    ax2.set_xlabel('Diffuse Irradiance Gd (W/m²)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('PV Power (kW)', fontsize=11, fontweight='bold')
    ax2.set_title('PV vs Diffuse Irradiance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    corr = pv_df[['Gd', 'PV_true']].corr().iloc[0, 1]
    ax2.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax2.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter 3: PV vs Temperature - REAL DATA ONLY
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(pv_df['T2m'], pv_df['PV_true'], alpha=0.5, s=10, color='#FF6B6B')
    ax3.set_xlabel('Temperature T2m (°C)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('PV Power (kW)', fontsize=11, fontweight='bold')
    ax3.set_title('PV vs Temperature', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    corr = pv_df[['T2m', 'PV_true']].corr().iloc[0, 1]
    ax3.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax3.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter 4: PV vs Clouds - REAL DATA ONLY
    # Note: Cloud cover % is an indirect proxy (measures sky coverage, not irradiance blocking)
    # Thin/broken clouds can still allow high irradiance, explaining weak correlation
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(pv_df['clouds'], pv_df['PV_true'], alpha=0.5, s=10, color='#9B59B6')
    ax4.set_xlabel('Cloud Cover (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('PV Power (kW)', fontsize=11, fontweight='bold')
    ax4.set_title('PV vs Cloud Cover\n(Indirect proxy - see Gb for direct driver)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    corr = pv_df[['clouds', 'PV_true']].corr().iloc[0, 1]
    ax4.text(0.05, 0.95, f'ρ = {corr:.3f}\n(weak: indirect)', transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Scatter 5: PV vs H_sun - REAL DATA ONLY
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(pv_df['H_sun'], pv_df['PV_true'], alpha=0.5, s=10, color='#F39C12')
    ax5.set_xlabel('Sun Height H_sun (°)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('PV Power (kW)', fontsize=11, fontweight='bold')
    ax5.set_title('PV vs Sun Height', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    corr = pv_df[['H_sun', 'PV_true']].corr().iloc[0, 1]
    ax5.text(0.05, 0.95, f'ρ = {corr:.3f}', transform=ax5.transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Correlation heatmap (spans 2x2)
    ax6 = fig.add_subplot(gs[1, 2])
    pv_features = ['PV_true', 'Gb', 'Gd', 'Gr', 'H_sun', 'T2m', 'WS10m', 'clouds', 'temp']
    corr_matrix = df[pv_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax6)
    ax6.set_title('PV Feature Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Overall title
    fig.suptitle('PV Feature Relationships and Correlations', fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(output_path_dir / 'figure3_pv_relationships.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ok] Figure 3 saved: {output_path_dir / 'figure3_pv_relationships.png'}")


def figure4_load_by_family_type(df: pd.DataFrame, output_path: Path):
    """
    Figure 4: Average daily load profile by family type.
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time').sort_index()
    
    # Map apartments to family types (based on assignment)
    # ap1,5,9,13,17 = retired
    # ap2,6,10,14,18 = working
    # ap3,7,11,15,19 = one_child
    # ap4,8,12,16,20 = two_children
    family_map = {
        'retired': ['load_ap1', 'load_ap5', 'load_ap9', 'load_ap13', 'load_ap17'],
        'working': ['load_ap2', 'load_ap6', 'load_ap10', 'load_ap14', 'load_ap18'],
        'one_child': ['load_ap3', 'load_ap7', 'load_ap11', 'load_ap15', 'load_ap19'],
        'two_children': ['load_ap4', 'load_ap8', 'load_ap12', 'load_ap16', 'load_ap20'],
    }
    
    # Compute average daily profile for each family type
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    
    colors = {'retired': '#4A90E2', 'working': '#50C878', 
              'one_child': '#FF6B6B', 'two_children': '#FFD93D'}
    labels = {'retired': 'Retired Couple', 'working': 'Working Couple',
              'one_child': 'Family (1 child)', 'two_children': 'Family (2 children)'}
    
    for family_type, ap_cols in family_map.items():
        # Average across apartments of this type
        family_load = df[ap_cols].mean(axis=1)
        
        # Group by hour of day
        hourly_avg = family_load.groupby(df.index.hour).mean()
        
        ax.plot(hourly_avg.index, hourly_avg.values, 
                linewidth=2.5, label=labels[family_type], color=colors[family_type], marker='o', markersize=4)
        ax.fill_between(hourly_avg.index, 0, hourly_avg.values, 
                        alpha=0.2, color=colors[family_type])
    
    ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Load (kWh/h)', fontsize=12, fontweight='bold')
    ax.set_title('Average Daily Load Profile by Household Type', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(0, 23)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ok] Figure 4 saved: {output_path}")


def figure5_battery_operation_logic(output_path: Path):
    """
    Figure 5: Battery operation logic diagram (constraints and energy balance).
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Battery box (center)
    battery_box = FancyBboxPatch((3.5, 4), 3, 2, boxstyle="round,pad=0.2", 
                                 facecolor='#FFD93D', edgecolor='black', linewidth=2.5)
    ax.add_patch(battery_box)
    ax.text(5, 5.8, 'Battery Energy Storage', fontsize=14, fontweight='bold', ha='center')
    ax.text(5, 5.3, 'Capacity: E_max (kWh)', fontsize=11, ha='center')
    ax.text(5, 4.8, 'Max Power: P_max (kW)', fontsize=11, ha='center')
    ax.text(5, 4.3, 'Efficiency: η_ch, η_dis (-)', fontsize=11, ha='center')
    
    # SoC indicator (inside battery)
    soc_bar = Rectangle((4, 4.5), 2, 1, facecolor='#4ECDC4', edgecolor='black', linewidth=1.5)
    ax.add_patch(soc_bar)
    ax.text(5, 5, 'SoC', fontsize=10, fontweight='bold', ha='center', va='center')
    ax.text(3.2, 5, 'SoC_min', fontsize=9, ha='right', va='center')
    ax.text(6.8, 5, 'SoC_max', fontsize=9, ha='left', va='center')
    
    # Constraints box (left)
    constraints_box = FancyBboxPatch((0.5, 1.5), 2.5, 2.5, boxstyle="round,pad=0.15", 
                                     facecolor='#FF6B6B', edgecolor='black', linewidth=2)
    ax.add_patch(constraints_box)
    ax.text(1.75, 3.5, 'Constraints', fontsize=12, fontweight='bold', ha='center')
    ax.text(1.75, 3.1, 'SoC_min ≤ SoC(t) ≤ SoC_max', fontsize=9, ha='center')
    ax.text(1.75, 2.7, '0 ≤ P_ch(t) ≤ P_max (kW)', fontsize=9, ha='center')
    ax.text(1.75, 2.3, '0 ≤ P_dis(t) ≤ P_max (kW)', fontsize=9, ha='center')
    ax.text(1.75, 1.9, 'P_ch(t) · P_dis(t) = 0', fontsize=9, ha='center')
    
    # Energy balance box (right)
    balance_box = FancyBboxPatch((7, 1.5), 2.5, 2.5, boxstyle="round,pad=0.15", 
                                 facecolor='#50C878', edgecolor='black', linewidth=2)
    ax.add_patch(balance_box)
    ax.text(8.25, 3.5, 'Energy Balance', fontsize=12, fontweight='bold', ha='center')
    ax.text(8.25, 3.1, 'SoC(t+1) = SoC(t)', fontsize=9, ha='center')
    ax.text(8.25, 2.8, '+ η_ch · P_ch(t) · Δt', fontsize=9, ha='center')
    ax.text(8.25, 2.5, '- (1/η_dis) · P_dis(t) · Δt', fontsize=9, ha='center')
    ax.text(8.25, 2.2, '[SoC: -, P: kW, Δt: h]', fontsize=8, ha='center', style='italic')
    
    # Inputs (top)
    pv_input = FancyBboxPatch((1, 7.5), 1.5, 0.6, boxstyle="round,pad=0.1", 
                              facecolor='#4ECDC4', edgecolor='black', linewidth=1.5)
    ax.add_patch(pv_input)
    ax.text(1.75, 7.8, 'PV(t) (kW)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    load_input = FancyBboxPatch((7.5, 7.5), 1.5, 0.6, boxstyle="round,pad=0.1", 
                                facecolor='#FF6B6B', edgecolor='black', linewidth=1.5)
    ax.add_patch(load_input)
    ax.text(8.25, 7.8, 'Load(t) (kWh/h)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows: PV → Battery
    ax.arrow(1.75, 7.5, 1.75, -1.3, head_width=0.2, head_length=0.15, 
             fc='black', ec='black', linewidth=2)
    ax.text(2.5, 6.5, 'Excess PV\n→ Charge', ha='center', fontsize=9, style='italic')
    
    # Arrows: Battery → Load
    ax.arrow(6.5, 5, 1.5, 2.3, head_width=0.2, head_length=0.15, 
             fc='black', ec='black', linewidth=2)
    ax.text(7.5, 6.5, 'Deficit\n→ Discharge', ha='center', fontsize=9, style='italic')
    
    # Charge/Discharge indicators
    charge_arrow = FancyArrowPatch((4.5, 4), (4.5, 3.2), 
                                    arrowstyle='->', mutation_scale=20, 
                                    color='green', linewidth=2.5)
    ax.add_patch(charge_arrow)
    ax.text(4.5, 3, 'P_ch (kW)', ha='center', fontsize=10, fontweight='bold', color='green')
    
    discharge_arrow = FancyArrowPatch((5.5, 3.2), (5.5, 4), 
                                       arrowstyle='->', mutation_scale=20, 
                                       color='red', linewidth=2.5)
    ax.add_patch(discharge_arrow)
    ax.text(5.5, 3, 'P_dis (kW)', ha='center', fontsize=10, fontweight='bold', color='red')
    
    # Title
    ax.text(5, 9.5, 'Battery Operation Logic and Constraints', 
            ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[ok] Figure 5 saved: {output_path}")


def generate_all_figures(master_csv_path: str, output_dir: str):
    """
    Generate all 5 thesis figures.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading master dataset...")
    df = pd.read_csv(master_csv_path, parse_dates=['time'])
    
    print("\nGenerating thesis figures...")
    
    # Figure 1: System diagram
    figure1_system_diagram(output_path / 'figure1_system_diagram.png')
    
    # Figure 2: Time-series alignment
    figure2_timeseries_alignment(df, output_path / 'figure2_timeseries_alignment.png', weeks=2)
    
    # Figure 3: PV feature relationships
    figure3_pv_feature_relationships(df, output_path)
    
    # Figure 4: Load by family type
    figure4_load_by_family_type(df, output_path / 'figure4_load_by_family_type.png')
    
    # Figure 5: Battery operation logic
    figure5_battery_operation_logic(output_path / 'figure5_battery_operation_logic.png')
    
    print(f"\n[ok] All figures saved to: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate thesis figures')
    parser.add_argument('--input', type=str, 
                       default='/Users/mariabigonah/Desktop/thesis/code/outputs/MASTER_20_APARTMENTS_2022_2023.csv',
                       help='Path to master dataset CSV')
    parser.add_argument('--output', type=str,
                       default='/Users/mariabigonah/Desktop/thesis/code/outputs/figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    generate_all_figures(args.input, args.output)

