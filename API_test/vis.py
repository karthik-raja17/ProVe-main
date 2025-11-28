import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# 1. RAW DATA (Transcribed from your Image & JSON results)
# ---------------------------------------------------------
# We use the raw item data to ensure calculations are 100% consistent
items = [
    # History
    {"Domain": "History", "Score": 0.0, "Sup": 0, "Ref": 0, "NEI": 4, "Status": "SUCCESS"},
    # Politics
    {"Domain": "Politics", "Score": 0.2203, "Sup": 19, "Ref": 6, "NEI": 12, "Status": "SUCCESS"}, # Obama
    {"Domain": "Politics", "Score": 0.2037, "Sup": 18, "Ref": 7, "NEI": 20, "Status": "SUCCESS"}, # Zelenskyy
    {"Domain": "Politics", "Score": 0.2647, "Sup": 11, "Ref": 2, "NEI": 17, "Status": "SUCCESS"}, # Washington
    {"Domain": "Politics", "Score": -0.4459, "Sup": 0, "Ref": 165, "NEI": 205, "Status": "SUCCESS"}, # Thatcher
    {"Domain": "Politics", "Score": 0.2692, "Sup": 14, "Ref": 0, "NEI": 18, "Status": "SUCCESS"}, # Lenin
    {"Domain": "Politics", "Score": 0.1228, "Sup": 13, "Ref": 6, "NEI": 33, "Status": "SUCCESS"}, # De Gaulle
    # Science
    {"Domain": "Science", "Score": 0.2105, "Sup": 5, "Ref": 1, "NEI": 13, "Status": "SUCCESS"},
    # Culture
    {"Domain": "Culture", "Score": 0.2235, "Sup": 25, "Ref": 6, "NEI": 34, "Status": "SUCCESS"}, # LOTR
    {"Domain": "Culture", "Score": 0.0, "Sup": 2, "Ref": 2, "NEI": 4, "Status": "SUCCESS"}, # Mozart
    # Geography
    {"Domain": "Geography", "Score": 0.2278, "Sup": 19, "Ref": 1, "NEI": 38, "Status": "SUCCESS"}, # USA
    {"Domain": "Geography", "Score": 0.2549, "Sup": 13, "Ref": 0, "NEI": 24, "Status": "SUCCESS"}, # UK
    {"Domain": "Geography", "Score": 0.0722, "Sup": 34, "Ref": 0, "NEI": 211, "Status": "SUCCESS"}, # Germany
    {"Domain": "Geography", "Score": 0.0588, "Sup": 1, "Ref": 0, "NEI": 14, "Status": "SUCCESS"}, # London
    {"Domain": "Geography", "Score": -0.1935, "Sup": 2, "Ref": 8, "NEI": 14, "Status": "SUCCESS"}, # NYC
    # Organizations
    {"Domain": "Organizations", "Score": 0.1765, "Sup": 7, "Ref": 1, "NEI": 16, "Status": "SUCCESS"}, # Google
    # Concepts
    {"Domain": "Concepts", "Score": -0.0526, "Sup": 8, "Ref": 9, "NEI": 2, "Status": "SUCCESS"}, # Democracy
    # Countries
    {"Domain": "Countries", "Score": 0.2692, "Sup": 14, "Ref": 0, "NEI": 23, "Status": "SUCCESS"}, # Canada
    {"Domain": "Countries", "Score": -0.1522, "Sup": 4, "Ref": 25, "NEI": 109, "Status": "SUCCESS"}, # Austria
]

df = pd.DataFrame(items)

# ---------------------------------------------------------
# CHART 1: Overall Label Distribution (Pie Chart)
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))
total_sup = df['Sup'].sum()
total_ref = df['Ref'].sum()
total_nei = df['NEI'].sum()

labels = ['Supports', 'Refutes', 'Not Enough Info']
sizes = [total_sup, total_ref, total_nei]
colors = ['#66b3ff', '#ff9999', '#99ff99'] # Blue, Red, Green
explode = (0.05, 0.05, 0)  # Explode slices for emphasis

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 12})
plt.title('Overall Distribution of Verified Claims', fontsize=16, fontweight='bold')
plt.axis('equal')  
plt.tight_layout()
plt.savefig('chart_label_distribution.png')
print("Generated 'chart_label_distribution.png'")

# ---------------------------------------------------------
# CHART 2: Average Reference Score by Domain (Bar Chart)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
domain_stats = df.groupby('Domain')['Score'].mean().sort_values()

# Color logic: Red for negative, Blue for positive
colors = ['#ff9999' if x < 0 else '#66b3ff' for x in domain_stats.values]

bars = plt.bar(domain_stats.index, domain_stats.values, color=colors, edgecolor='black')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + (0.01 if yval>0 else -0.02), 
             f'{yval:.3f}', ha='center', va='bottom' if yval>0 else 'top', fontweight='bold')

plt.axhline(0, color='black', linewidth=0.8)
plt.title('Average Reference Score by Domain', fontsize=16, fontweight='bold')
plt.ylabel('Avg Score (-1.0 to 1.0)', fontsize=12)
plt.xlabel('Domain', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('chart_domain_performance.png')
print("Generated 'chart_domain_performance.png'")

# ---------------------------------------------------------
# CHART 3: Domain Verification Breakdown (Stacked Bar)
# ---------------------------------------------------------
# This helps explain WHY Politics/Concepts have lower scores (high Refutes?)
plt.figure(figsize=(12, 6))
domain_counts = df.groupby('Domain')[['Sup', 'Ref', 'NEI']].sum()

# Normalize to percentage to make domains comparable? 
# Or keep raw counts? Let's do raw counts but maybe log scale if variances are huge?
# Let's do 100% Stacked Bar to compare the "composition" of truth per domain
domain_pct = domain_counts.div(domain_counts.sum(1), axis=0) * 100

ax = domain_pct.plot(kind='bar', stacked=True, color=['#66b3ff', '#ff9999', '#99ff99'], 
                     figsize=(12, 6), edgecolor='black')

plt.title('Distribution of Labels per Domain (Normalized %)', fontsize=16, fontweight='bold')
plt.ylabel('Percentage of Claims (%)', fontsize=12)
plt.xlabel('Domain', fontsize=12)
plt.legend(['Supports', 'Refutes', 'Not Enough Info'], loc='upper right')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save using the figure object from pandas plot
plt.savefig('chart_domain_breakdown.png')
print("Generated 'chart_domain_breakdown.png'")