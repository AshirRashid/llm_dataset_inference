import os
import json
from datetime import datetime

# Limit multi-threading to avoid RLIMIT_NPROC issues on the cluster
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# --- Configuration Constants ---
INPUT_FILE = "/scratch/ar7789/llm_dataset_inference/scripts/dataset_stratified_nlp_x_quant_x_mia/all_models/analysis_results.json"
OUTPUT_DIR = "/scratch/ar7789/llm_dataset_inference/scripts/dashboard"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dashboard_dataset_stratified_nlp_x_quant_x_mia.html")

def flatten_causality_data(json_data):
    """
    Flattens the nested JSON structure into a list of records for easier filtering.
    """
    records = []
    for model_name, datasets in json_data.items():
        for dataset_name, attributes in datasets.items():
            for attr_name, quartiles in attributes.items():
                for q_data in quartiles:
                    records.append({
                        "model": model_name,
                        "dataset": dataset_name,
                        "attribute": attr_name,
                        "quartile": q_data["quartile"],
                        "avg_attr": q_data["avg_attr"],
                        "best_metric": q_data["best_metric"],
                        "cohens_d": q_data["cohens_d"]
                    })
    return records

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multivariate Stratification Dashboard</title>
    
    <!-- Design System: MRI Spec (Dark Mode + Glassmorphism) -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --accent-color: #38bdf8;
            --accent-gradient: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
            --text-color: #f1f5f9;
            --text-muted: #94a3b8;
            --glass-border: rgba(255, 255, 255, 0.1);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.4;
            min-height: 100vh;
            padding: 1rem;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 1.5rem;
            text-align: center;
        }

        h1 {
            font-size: 1.8rem;
            font-weight: 600;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 0.75rem;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.3);
        }

        .filters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
        }

        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }

        .filter-group label {
            font-weight: 600;
            color: var(--accent-color);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            color: var(--text-color);
            padding: 0.5rem;
            border-radius: 0.4rem;
            font-family: 'Outfit', sans-serif;
            font-size: 0.85rem;
            cursor: pointer;
            outline: none;
        }

        #plot-container {
            height: 500px;
            width: 100%;
        }

        .grid-view {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .small-chart {
            height: 250px;
            padding: 0.75rem;
        }

        .footer {
            text-align: center;
            margin-top: 2rem;
            color: var(--text-muted);
            font-size: 0.75rem;
        }

    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Stratified NLP vs. Quantization Dashboard</h1>
            <p class="subtitle">Cohen's d (MIA Vulnerability) across Text Attribute Quartiles</p>
        </header>

        <section class="glass-card">
            <div class="filters">
                <div class="filter-group">
                    <label for="filter-model">Model</label>
                    <select id="filter-model"></select>
                </div>
                <div class="filter-group">
                    <label for="filter-dataset">Dataset</label>
                    <select id="filter-dataset"></select>
                </div>
                <div class="filter-group">
                    <label for="filter-attribute">Attribute</label>
                    <select id="filter-attribute">
                        <option value="ALL">Show All (Grid)</option>
                    </select>
                </div>
            </div>
        </section>

        <div id="main-view" class="glass-card">
            <div id="plot-container"></div>
        </div>

        <div id="grid-view" class="grid-view" style="display: none;">
            <!-- Small charts injected here -->
        </div>

        <footer class="footer">
            Generated: <span id="gen-date"></span> | Dataset Stratification
        </footer>
    </div>

    <script>
        const rawData = [DATA_JSON];
        
        const filters = {
            model: document.getElementById('filter-model'),
            dataset: document.getElementById('filter-dataset'),
            attribute: document.getElementById('filter-attribute')
        };

        function initFilters() {
            const getUnique = (key) => [...new Set(rawData.map(d => d[key]))].sort();
            const populate = (el, items) => {
                el.innerHTML += items.map(it => `<option value="${it}">${it}</option>`).join('');
            };

            populate(filters.model, getUnique('model'));
            populate(filters.dataset, getUnique('dataset'));
            populate(filters.attribute, getUnique('attribute'));

            [filters.model, filters.dataset, filters.attribute].forEach(el => {
                el.addEventListener('change', updateDashboard);
            });
            document.getElementById('gen-date').innerText = new Date().toLocaleString();
        }

        function updateDashboard() {
            const model = filters.model.value;
            const dataset = filters.dataset.value;
            const attribute = filters.attribute.value;

            if (attribute === "ALL") {
                document.getElementById('main-view').style.display = 'none';
                document.getElementById('grid-view').style.display = 'grid';
                requestAnimationFrame(() => renderGridView(model, dataset));
            } else {
                document.getElementById('main-view').style.display = 'block';
                document.getElementById('grid-view').style.display = 'none';
                requestAnimationFrame(() => renderMainPlot(model, dataset, attribute));
            }
        }

        function renderMainPlot(model, dataset, attribute) {
            const filtered = rawData.filter(d => d.model === model && d.dataset === dataset && d.attribute === attribute);
            const container = document.getElementById('plot-container');
            const rect = container.getBoundingClientRect();
            
            const trace = {
                x: filtered.map(d => `Q${d.quartile}`),
                y: filtered.map(d => d.cohens_d),
                text: filtered.map(d => `Avg ${attribute}: ${d.avg_attr.toFixed(3)}<br>Best Metric: ${d.best_metric}`),
                mode: 'lines+markers',
                type: 'scatter',
                line: { color: '#38bdf8', width: 3, shape: 'spline' },
                marker: { size: 10, color: '#f1f5f9', bordercolor: '#38bdf8', borderwidth: 2 },
                hoverinfo: 'y+text'
            };

            const layout = {
                title: { text: `${attribute} Impact`, font: { size: 14, color: '#38bdf8' }, y: 0.95 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Outfit, sans-serif', color: '#f1f5f9', size: 10 },
                yaxis: { title: "Cohen's d", gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                xaxis: { title: "Quartile", gridcolor: 'rgba(255,255,255,0.05)' },
                margin: { t: 30, b: 40, l: 40, r: 10 },
                width: rect.width,
                height: 500,
                autosize: true
            };

            Plotly.newPlot(container, [trace], layout, { responsive: true, displayModeBar: false });
        }

        function renderGridView(model, dataset) {
            const grid = document.getElementById('grid-view');
            grid.innerHTML = '';
            const attributes = [...new Set(rawData.map(d => d.attribute))].sort();
            
            attributes.forEach(attr => {
                const card = document.createElement('div');
                card.className = 'glass-card small-chart';
                const id = 'chart-' + attr.replace(/\\\\s+/g, '-').toLowerCase();
                card.id = id;
                grid.appendChild(card);
                
                // Use setTimeout to ensure the card is in the DOM and has a width
                setTimeout(() => {
                    const rect = card.getBoundingClientRect();
                    const filtered = rawData.filter(d => d.model === model && d.dataset === dataset && d.attribute === attr);
                    
                    const trace = {
                        x: filtered.map(d => `Q${d.quartile}`),
                        y: filtered.map(d => d.cohens_d),
                        mode: 'lines+markers',
                        type: 'scatter',
                        line: { color: '#818cf8', width: 2, shape: 'spline' },
                        marker: { size: 6 }
                    };
                    
                    const layout = {
                        title: { text: attr, font: { size: 11, color: '#94a3b8' }, y: 0.98 },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: { family: 'Outfit, sans-serif', color: '#f1f5f9', size: 9 },
                        margin: { t: 25, b: 25, l: 30, r: 10 },
                        yaxis: { gridcolor: 'rgba(255,255,255,0.03)', zeroline: false },
                        xaxis: { gridcolor: 'rgba(255,255,255,0.03)' },
                        width: rect.width,
                        height: 250,
                        autosize: true
                    };
                    Plotly.newPlot(id, [trace], layout, { displayModeBar: false, responsive: true });
                }, 0);
            });
        }

        initFilters();
        updateDashboard();
    </script>
</body>
</html>
"""

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)

    records = flatten_causality_data(data)
    
    # Inject data into template
    json_str = json.dumps(records)
    final_html = HTML_TEMPLATE.replace("[DATA_JSON]", json_str)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(final_html)
        
    print(f"Dashboard generated successfully: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
