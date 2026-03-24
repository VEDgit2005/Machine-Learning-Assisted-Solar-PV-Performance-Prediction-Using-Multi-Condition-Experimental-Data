"""
SIMPLE VISUALIZATION SNIPPETS - Solar PV Dataset
Easy to understand, modify, and run independently
Perfect for quick customizations before your review
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load your cleaned dataset
df = pd.read_csv('SOLAR_PV_CLEANED_DATASET.csv')
df['date'] = pd.to_datetime(df['date'])

print("Dataset loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# =============================================================================
# SNIPPET 1: Simple Line Chart - Daily Generation
# =============================================================================
print("\n📊 Creating Snippet 1: Daily Generation Line Chart...")

plt.figure(figsize=(14, 6))
plt.plot(df['date'], df['day_gen_kwh'], linewidth=2, color='#2ecc71', alpha=0.8)
plt.fill_between(df['date'], df['day_gen_kwh'], alpha=0.3, color='#2ecc71')

# Add average line
avg = df['day_gen_kwh'].mean()
plt.axhline(avg, color='red', linestyle='--', linewidth=2, label=f'Average: {avg:.0f} kWh')

plt.title('Daily Solar Energy Generation', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Generation (kWh)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('simple_01_daily_generation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_01_daily_generation.png")

# =============================================================================
# SNIPPET 2: Bar Chart - Monthly Generation
# =============================================================================
print("\n📊 Creating Snippet 2: Monthly Generation Bar Chart...")

monthly_gen = df.groupby('month')['day_gen_kwh'].sum()

plt.figure(figsize=(12, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(monthly_gen)))
bars = plt.bar(monthly_gen.index, monthly_gen.values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, val in zip(bars, monthly_gen.values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Total Monthly Generation', fontsize=16, fontweight='bold')
plt.xlabel('Month', fontsize=12)
plt.ylabel('Total Generation (kWh)', fontsize=12)
plt.xticks(range(1, 13))
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('simple_02_monthly_bars.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_02_monthly_bars.png")

# =============================================================================
# SNIPPET 3: Scatter Plot - Radiation vs Generation
# =============================================================================
print("\n📊 Creating Snippet 3: Radiation vs Generation Scatter...")

plt.figure(figsize=(10, 8))
scatter = plt.scatter(df['radiation_kwh_m2_day'], df['day_gen_kwh'], 
                     c=df['performance_ratio_pct'], cmap='RdYlGn', 
                     s=80, alpha=0.6, edgecolors='black', linewidth=0.5)

# Add trend line
z = np.polyfit(df['radiation_kwh_m2_day'], df['day_gen_kwh'], 1)
p = np.poly1d(z)
plt.plot(df['radiation_kwh_m2_day'], p(df['radiation_kwh_m2_day']), 
        "r--", linewidth=3, alpha=0.8, label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')

# Calculate correlation
corr = df['radiation_kwh_m2_day'].corr(df['day_gen_kwh'])

plt.title('Solar Radiation vs Energy Generation', fontsize=16, fontweight='bold')
plt.xlabel('Solar Radiation (kWh/m²/day)', fontsize=12)
plt.ylabel('Generation (kWh)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add correlation text
plt.text(0.05, 0.95, f'Correlation: {corr:.3f}\n(Strong positive relationship)', 
        transform=plt.gca().transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

cbar = plt.colorbar(scatter)
cbar.set_label('Performance Ratio (%)', fontsize=11)

plt.tight_layout()
plt.savefig('simple_03_scatter_radiation.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_03_scatter_radiation.png")

# =============================================================================
# SNIPPET 4: Box Plot - Seasonal Comparison
# =============================================================================
print("\n📊 Creating Snippet 4: Seasonal Box Plot...")

plt.figure(figsize=(10, 7))

# Prepare data
season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
available_seasons = [s for s in season_order if s in df['season'].unique()]
df['season'] = pd.Categorical(df['season'], categories=season_order, ordered=True)
df_sorted = df.sort_values('season')

# Create box plot
bp = df_sorted.boxplot(column='day_gen_kwh', by='season', 
                       figsize=(10, 7), patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2.5))

plt.suptitle('')  # Remove default title
plt.title('Generation Distribution by Season', fontsize=16, fontweight='bold')
plt.xlabel('Season', fontsize=12)
plt.ylabel('Daily Generation (kWh)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_04_seasonal_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_04_seasonal_boxplot.png")

# =============================================================================
# SNIPPET 5: Histogram - Generation Distribution
# =============================================================================
print("\n📊 Creating Snippet 5: Generation Distribution Histogram...")

plt.figure(figsize=(10, 7))

plt.hist(df['day_gen_kwh'], bins=40, color='#3498db', edgecolor='black', alpha=0.7)

# Add mean and median lines
mean_val = df['day_gen_kwh'].mean()
median_val = df['day_gen_kwh'].median()

plt.axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
           label=f'Mean: {mean_val:.0f} kWh')
plt.axvline(median_val, color='orange', linestyle='--', linewidth=2.5, 
           label=f'Median: {median_val:.0f} kWh')

plt.title('Distribution of Daily Generation', fontsize=16, fontweight='bold')
plt.xlabel('Generation (kWh)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('simple_05_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_05_histogram.png")

# =============================================================================
# SNIPPET 6: Pie Chart - Seasonal Contribution
# =============================================================================
print("\n📊 Creating Snippet 6: Seasonal Contribution Pie Chart...")

seasonal_total = df.groupby('season')['day_gen_kwh'].sum()

plt.figure(figsize=(10, 10))
colors_pie = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
explode = [0.05] * len(seasonal_total)  # Slight separation

wedges, texts, autotexts = plt.pie(seasonal_total.values, 
                                    labels=seasonal_total.index,
                                    autopct='%1.1f%%',
                                    colors=colors_pie[:len(seasonal_total)],
                                    explode=explode,
                                    startangle=90,
                                    textprops={'fontsize': 12, 'fontweight': 'bold'})

# Make percentage text bold and larger
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')

plt.title('Total Generation by Season', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('simple_06_seasonal_pie.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_06_seasonal_pie.png")

# =============================================================================
# SNIPPET 7: Heatmap - Correlation Matrix
# =============================================================================
print("\n📊 Creating Snippet 7: Correlation Heatmap...")

# Select numeric columns
numeric_cols = ['day_gen_kwh', 'radiation_kwh_m2_day', 'performance_ratio_pct',
               'cuf_on_ac_capacity_%', 'cuf_on_dc_capacity_%']
corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
           square=True, linewidths=2, cbar_kws={'shrink': 0.8},
           vmin=-1, vmax=1, center=0,
           annot_kws={'fontsize': 11, 'fontweight': 'bold'})

plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig('simple_07_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_07_correlation_heatmap.png")

# =============================================================================
# SNIPPET 8: Dual Axis - Generation & Performance Ratio
# =============================================================================
print("\n📊 Creating Snippet 8: Monthly Generation & PR (Dual Axis)...")

monthly_stats = df.groupby('month').agg({
    'day_gen_kwh': 'mean',
    'performance_ratio_pct': 'mean'
}).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot generation on left axis
color1 = '#3498db'
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Average Generation (kWh)', fontsize=12, color=color1)
line1 = ax1.plot(monthly_stats['month'], monthly_stats['day_gen_kwh'], 
                color=color1, marker='o', linewidth=3, markersize=10, label='Generation')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3, linestyle='--')

# Plot performance ratio on right axis
ax2 = ax1.twinx()
color2 = '#e74c3c'
ax2.set_ylabel('Performance Ratio (%)', fontsize=12, color=color2)
line2 = ax2.plot(monthly_stats['month'], monthly_stats['performance_ratio_pct'], 
                color=color2, marker='s', linewidth=3, markersize=10, label='Performance Ratio')
ax2.tick_params(axis='y', labelcolor=color2)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', fontsize=11)

plt.title('Monthly Generation vs Performance Ratio', fontsize=16, fontweight='bold')
ax1.set_xticks(range(1, 13))
plt.tight_layout()
plt.savefig('simple_08_dual_axis.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_08_dual_axis.png")

# =============================================================================
# SNIPPET 9: Stacked Area Chart - Cumulative Generation
# =============================================================================
print("\n📊 Creating Snippet 9: Cumulative Generation...")

df_sorted = df.sort_values('date')
cumulative = df_sorted['day_gen_kwh'].cumsum()

plt.figure(figsize=(14, 7))
plt.plot(df_sorted['date'], cumulative, linewidth=3, color='#2ecc71')
plt.fill_between(df_sorted['date'], cumulative, alpha=0.4, color='#2ecc71')

# Add milestone markers
total = cumulative.iloc[-1]
milestones = [total * 0.25, total * 0.5, total * 0.75]
for milestone in milestones:
    idx = (cumulative - milestone).abs().idxmin()
    date_milestone = df_sorted.loc[idx, 'date']
    plt.plot(date_milestone, milestone, 'ro', markersize=10)
    plt.text(date_milestone, milestone, f' {milestone/1000:.0f}k kWh', 
            fontsize=10, fontweight='bold', va='bottom')

plt.title('Cumulative Energy Generation', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Generation (kWh)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Add total text box
plt.text(0.02, 0.98, f'TOTAL: {total:,.0f} kWh', 
        transform=plt.gca().transAxes, fontsize=14, fontweight='bold',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

plt.tight_layout()
plt.savefig('simple_09_cumulative.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_09_cumulative.png")

# =============================================================================
# SNIPPET 10: Weekday vs Weekend Comparison
# =============================================================================
print("\n📊 Creating Snippet 10: Weekday vs Weekend...")

weekday_avg = df[df['is_weekend'] == 0]['day_gen_kwh'].mean()
weekend_avg = df[df['is_weekend'] == 1]['day_gen_kwh'].mean()

plt.figure(figsize=(10, 7))

categories = ['Weekday', 'Weekend']
values = [weekday_avg, weekend_avg]
colors_we = ['#3498db', '#e74c3c']

bars = plt.bar(categories, values, color=colors_we, edgecolor='black', 
              linewidth=2, width=0.6, alpha=0.8)

# Add value labels
for bar, val in zip(bars, values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f} kWh',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add percentage difference
diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
plt.text(0.5, 0.95, f'Difference: {abs(diff_pct):.1f}% {"higher" if diff_pct > 0 else "lower"} on weekends', 
        transform=plt.gca().transAxes, ha='center', va='top',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.title('Average Generation: Weekday vs Weekend', fontsize=16, fontweight='bold')
plt.ylabel('Average Generation (kWh)', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.ylim(0, max(values) * 1.2)

plt.tight_layout()
plt.savefig('simple_10_weekday_weekend.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✓ Saved: simple_10_weekday_weekend.png")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print(" ALL 10 SIMPLE VISUALIZATIONS CREATED!")
print("=" * 80)
print("\nGenerated Files:")
print("   1. simple_01_daily_generation.png      - Line chart of daily generation")
print("   2. simple_02_monthly_bars.png          - Bar chart of monthly totals")
print("   3. simple_03_scatter_radiation.png     - Scatter: radiation vs generation")
print("   4. simple_04_seasonal_boxplot.png      - Box plot by season")
print("   5. simple_05_histogram.png             - Distribution histogram")
print("   6. simple_06_seasonal_pie.png          - Pie chart of seasonal share")
print("   7. simple_07_correlation_heatmap.png   - Correlation matrix")
print("   8. simple_08_dual_axis.png             - Generation & PR dual axis")
print("   9. simple_09_cumulative.png            - Cumulative generation")
print("  10. simple_10_weekday_weekend.png       - Weekday vs weekend comparison")
print("\n" + "=" * 80)
print(" Ready to customize or present!")
print("=" * 80)