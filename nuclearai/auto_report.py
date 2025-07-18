import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template

REPORT_TEMPLATE = """
<html>
<head><title>NuclearAI Auto Report</title></head>
<body>
<h1>🚀 NuclearAI Experiment Report</h1>
<h2>Executive Summary</h2>
<p>{{ summary }}</p>
<h2>Results Table</h2>
{{ table_html|safe }}
<h2>Plots</h2>
{% for plot in plots %}
<img src="{{ plot }}" width="600"><br>
{% endfor %}
</body>
</html>
"""

def generate_report(results_dir, output_file, summary="Auto-generated by NuclearAI."):
    """
    Generate a PDF/HTML report for all results in results_dir.
    """
    # Collect results
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.result.json')]
    records = []
    for rf in result_files:
        with open(os.path.join(results_dir, rf)) as f:
            records.append(json.load(f))
    df = pd.DataFrame(records)
    table_html = df.to_html(index=False)
    # Plot summary
    plot_path = os.path.join(results_dir, 'summary_plot.png')
    plt.figure(figsize=(8,4))
    plt.bar(df['scenario'], df['cost'])
    plt.ylabel('Cost')
    plt.title('Scenario Cost Summary')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path)
    # Render HTML
    template = Template(REPORT_TEMPLATE)
    html = template.render(summary=summary, table_html=table_html, plots=[plot_path])
    with open(output_file, 'w') as f:
        f.write(html)
    print(f"Report generated: {output_file}") 