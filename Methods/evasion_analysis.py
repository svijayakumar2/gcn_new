import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON file
with open('../security_analysis/security_summary.json', 'r') as f:
    data = json.load(f)

# Extract data for plotting
techniques = []
evasion_rates = []
confidence_drops = []
detection_scores = []

for technique, metrics in data['vulnerability_ranking']:
    techniques.append('\n'.join(technique.replace('_', ' ').split()))
    evasion_rates.append(metrics['evasion_success_rate'] * 100)
    confidence_drops.append(metrics['avg_confidence_drop'] * 100)
    detection_scores.append(metrics['avg_detection_score'] * 100)

# Set up the plot
plt.figure(figsize=(10, 6))
width = 0.25
x = np.arange(len(techniques))

# Create bars
plt.bar(x - width, evasion_rates, width, label='Evasion Success Rate', color='#4C72B0')
plt.bar(x, confidence_drops, width, label='Confidence Drop', color='#55A868')
plt.bar(x + width, detection_scores, width, label='Detection Score', color='#C44E52')

# Customize the plot
plt.xlabel('Evasion Technique', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.title('Model Robustness Against Various Evasion Techniques', fontsize=14, pad=20)
plt.xticks(x, techniques, rotation=45, ha='right')
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='center', ncol=3)

# Add grid
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('evasion_analysis.pdf', bbox_inches='tight', dpi=300)
plt.savefig('evasion_analysis.png', bbox_inches='tight', dpi=300)