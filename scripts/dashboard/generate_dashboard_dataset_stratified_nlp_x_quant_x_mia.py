import os
import json
from datetime import datetime

import re

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

def parse_model_info(model_dir_name):
    """
    Parses model directory name to extract family, quantization type, bit-width, and config.
    Example: pythia-12b-deduped-gptq-b4-gs128-da1
    """
    # Default values
    family = model_dir_name
    quant_type = "Original"
    bit_width = "fp16"
    config = "Default"

    # Check for quantization patterns
    families = ["pythia-70m", "pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-1b", "pythia-2.8b", "pythia-6.9b", "pythia-12b"]
    for f in families:
        if f in model_dir_name:
            family = f
            break
    
    # Extract bit-width, quant type, and config
    bit_match = re.search(r'-b(\d+)(?:-(.+))?$', model_dir_name)
    if bit_match:
        bits = int(bit_match.group(1))
        bit_width = f"b{bits}"
        bit_sort_key = bits
        if bit_match.group(2):
            config = bit_match.group(2)
    else:
        bit_sort_key = 99 # fp16/Original
    
    if "gptq" in model_dir_name.lower():
        quant_type = "GPTQ"
    elif "awq" in model_dir_name.lower():
        quant_type = "AWQ"
    elif "static" in model_dir_name.lower():
        quant_type = "Static"
    elif "bnb" in model_dir_name.lower():
        quant_type = "BnB"
    
    return family, quant_type, bit_width, bit_sort_key, config

def flatten_causality_data(json_data):
    """
    Flattens the nested JSON structure into a list of records for easier filtering.
    """
    records = []
    for model_name, datasets in json_data.items():
        family, quant_type, bit_width, bit_sort_key, config = parse_model_info(model_name)
        for dataset_name, attributes in datasets.items():
            for attr_name, quartiles in attributes.items():
                for q_data in quartiles:
                    records.append({
                        "model": model_name,
                        "family": family,
                        "quant_type": quant_type,
                        "bit_width": bit_width,
                        "bit_sort_key": bit_sort_key,
                        "config": config,
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
            overflow: hidden;
        }

        .filters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .filter-group label {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .filter-group select {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            color: var(--text-color);
            padding: 0.5rem;
            border-radius: 0.5rem;
            font-family: inherit;
            font-size: 0.85rem;
            outline: none;
            transition: all 0.2s;
        }

        .filter-group select[multiple] {
            height: 100px;
        }

        .filter-group select:focus {
            border-color: var(--accent-color);
            background: rgba(255, 255, 255, 0.1);
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
                    <label>Family</label>
                    <select id="filter-family" multiple></select>
                </div>
                <div class="filter-group">
                    <label>Quant Type</label>
                    <select id="filter-quant" multiple></select>
                </div>
                <div class="filter-group">
                    <label>Bit Width</label>
                    <select id="filter-bits" multiple></select>
                </div>
                <div class="filter-group">
                    <label>Config</label>
                    <select id="filter-config" multiple></select>
                </div>
                <div class="filter-group">
                    <label>Dataset</label>
                    <select id="filter-dataset" multiple></select>
                </div>
                <div class="filter-group">
                    <label>Attribute</label>
                    <select id="filter-attribute"></select>
                </div>
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1.5rem; border-top: 1px solid var(--glass-border); padding-top: 1rem;">
                <div class="filter-group" style="flex-direction: row; align-items: center; gap: 1rem;">
                    <label style="margin:0;">Export Format</label>
                    <select id="export-format" style="height: auto;">
                        <option value="svg">SVG (Vector)</option>
                        <option value="png">PNG (300DPI)</option>
                    </select>
                    <button id="btn-export-main" style="background: var(--accent-gradient); border: none; color: white; padding: 0.5rem 1.5rem; border-radius: 0.4rem; cursor: pointer; font-weight: 600;">Export Figure</button>
                </div>
                <button id="reset-filters" style="background: none; border: 1px solid var(--accent-color); color: var(--accent-color); padding: 0.4rem 1rem; border-radius: 0.4rem; cursor: pointer;">Reset Filters</button>
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
            family: document.getElementById('filter-family'),
            quant: document.getElementById('filter-quant'),
            bits: document.getElementById('filter-bits'),
            config: document.getElementById('filter-config'),
            dataset: document.getElementById('filter-dataset'),
            attribute: document.getElementById('filter-attribute')
        };

        function getSelectedValues(el) {
            return Array.from(el.selectedOptions).map(opt => opt.value);
        }

        function initFilters() {
            const getUnique = (key) => [...new Set(rawData.map(d => d[key]))].sort();
            const populate = (el, items) => {
                el.innerHTML = items.map(it => `<option value="${it}" selected>${it}</option>`).join('');
            };

            populate(filters.family, getUnique('family'));
            populate(filters.quant, getUnique('quant_type'));
            populate(filters.bits, getUnique('bit_width'));
            populate(filters.config, getUnique('config'));
            populate(filters.dataset, getUnique('dataset'));
            
            // For attribute, we want a single select with "ALL" option
            const attrs = getUnique('attribute');
            filters.attribute.innerHTML = '<option value="ALL">Grid View (All Attributes)</option>' + 
                attrs.map(it => `<option value="${it}">${it}</option>`).join('');

            [filters.family, filters.quant, filters.bits, filters.config, filters.dataset, filters.attribute].forEach(el => {
                el.addEventListener('change', updateDashboard);
            });

            document.getElementById('reset-filters').addEventListener('click', () => {
                [filters.family, filters.quant, filters.bits, filters.config, filters.dataset].forEach(el => {
                    Array.from(el.options).forEach(opt => opt.selected = true);
                });
                filters.attribute.value = "ALL";
                updateDashboard();
            });

            document.getElementById('gen-date').innerText = new Date().toLocaleString();
        }

        function updateDashboard() {
            const selected = {
                family: getSelectedValues(filters.family),
                quant: getSelectedValues(filters.quant),
                bits: getSelectedValues(filters.bits),
                config: getSelectedValues(filters.config),
                dataset: getSelectedValues(filters.dataset),
                attribute: filters.attribute.value
            };

            const filteredData = rawData.filter(d => 
                selected.family.includes(d.family) &&
                selected.quant.includes(d.quant_type) &&
                selected.bits.includes(d.bit_width) &&
                selected.config.includes(d.config) &&
                selected.dataset.includes(d.dataset)
            );

            if (selected.attribute === "ALL") {
                document.getElementById('main-view').style.display = 'none';
                document.getElementById('grid-view').style.display = 'grid';
                requestAnimationFrame(() => renderGridView(filteredData));
            } else {
                document.getElementById('main-view').style.display = 'block';
                document.getElementById('grid-view').style.display = 'none';
                requestAnimationFrame(() => renderMainPlot(filteredData, selected.attribute));
            }
        }

        function renderMainPlot(data, attribute) {
            const filtered = data.filter(d => d.attribute === attribute);
            const rect = document.getElementById('plot-container').getBoundingClientRect();
            
            // Group by quartile and average the Cohen's d across selected models/datasets
            const agg = {};
            filtered.forEach(d => {
                if (!agg[d.quartile]) agg[d.quartile] = { sum: 0, count: 0, sum_attr: 0 };
                agg[d.quartile].sum += Math.abs(d.cohens_d);
                agg[d.quartile].count++;
                agg[d.quartile].sum_attr += d.avg_attr;
            });

            const x = Object.keys(agg).sort((a,b) => a-b).map(q => `Q${q}`);
            const y = Object.keys(agg).sort((a,b) => a-b).map(q => agg[q].sum / agg[q].count);

            const trace = {
                x: x,
                y: y,
                text: x.map((q, i) => `Avg ${attribute}: ${(agg[i+1].sum_attr / agg[i+1].count).toFixed(3)}<br>Models: ${agg[i+1].count}`),
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
                yaxis: { title: "|Cohen's d|", gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
                xaxis: { title: attribute, gridcolor: 'rgba(255,255,255,0.05)' },
                margin: { t: 30, b: 40, l: 40, r: 10 },
                autosize: true
            };

            Plotly.newPlot(document.getElementById('plot-container'), [trace], layout, { responsive: true, displayModeBar: false });
        }

        function renderGridView(data) {
            const grid = document.getElementById('grid-view');
            grid.innerHTML = '';
            
            const attributes = [...new Set(data.map(d => d.attribute))].sort();
            
            attributes.forEach(attr => {
                const attrData = data.filter(d => d.attribute === attr);
                if (attrData.length === 0) return;

                const card = document.createElement('div');
                card.className = 'glass-card small-chart';
                const chartId = `chart-${attr.replace(/\s+/g, '-')}`;
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-size: 0.7rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">${attr}</span>
                        <button class="download-btn" data-chart="${chartId}" style="background: none; border: 1px solid var(--glass-border); color: var(--text-muted); padding: 2px 6px; border-radius: 4px; cursor: pointer; font-size: 10px;">SVG</button>
                    </div>
                    <div id="${chartId}" style="height: 180px; width: 100%;"></div>
                `;
                grid.appendChild(card);

                // Group by quartile and average
                const agg = {};
                attrData.forEach(d => {
                    if (!agg[d.quartile]) agg[d.quartile] = { sum: 0, count: 0 };
                    agg[d.quartile].sum += Math.abs(d.cohens_d);
                    agg[d.quartile].count++;
                });

                const x = Object.keys(agg).sort((a,b) => a-b).map(q => `Q${q}`);
                const y = Object.keys(agg).sort((a,b) => a-b).map(q => agg[q].sum / agg[q].count);

                const trace = {
                    x: x,
                    y: y,
                    mode: 'lines+markers',
                    type: 'scatter',
                    line: { color: '#818cf8', width: 2, shape: 'spline' },
                    marker: { size: 6 }
                };

                const layout = {
                    paper_bgcolor: 'rgba(0,0,0,0)',
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    font: { family: 'Outfit, sans-serif', color: '#f1f5f9', size: 9 },
                    margin: { t: 10, b: 25, l: 30, r: 20 },
                    yaxis: { gridcolor: 'rgba(255,255,255,0.03)', zeroline: false },
                    xaxis: { title: { text: attr, font: { size: 8 } }, gridcolor: 'rgba(255,255,255,0.03)' },
                    showlegend: false,
                    autosize: true
                };

                // Use requestAnimationFrame to ensure the container is fully sized by the browser
                requestAnimationFrame(() => {
                    Plotly.newPlot(chartId, [trace], layout, { displayModeBar: false, responsive: true });
                });
            });
        }

        // --- Export Logic ---
        document.getElementById('btn-export-main').addEventListener('click', () => {
            if (filters.attribute.value === "ALL") {
                alert("Please select a specific attribute to export the high-res main plot, or use the download icons on individual grid charts.");
                return;
            }
            const format = document.getElementById('export-format').value;
            const plotEl = document.getElementById('plot-container');
            
            // Construct a descriptive filename
            const family = getSelectedValues(filters.family)[0] || 'all';
            const filename = `stratified_${filters.attribute.value}_${family}_${getSelectedValues(filters.dataset)[0] || 'all'}`;

            const btn = document.getElementById('btn-export-main');
            btn.innerText = '...';

            const paperStyle = {
                'font.color': '#000000',
                'paper_bgcolor': '#ffffff',
                'plot_bgcolor': '#ffffff',
                'xaxis.title.font.color': '#000000',
                'xaxis.tickfont.color': '#000000',
                'xaxis.gridcolor': '#e2e8f0',
                'yaxis.title.font.color': '#000000',
                'yaxis.tickfont.color': '#000000',
                'yaxis.gridcolor': '#e2e8f0',
                'margin': { t: 50, b: 100, l: 80, r: 50 }
            };

            const originalLayout = JSON.parse(JSON.stringify(plotEl.layout));
            
            Plotly.relayout(plotEl, paperStyle).then(() => {
                Plotly.downloadImage(plotEl, {
                    format: format,
                    width: 1000,
                    height: 600,
                    scale: 3,
                    filename: filename
                }).then(() => {
                    Plotly.relayout(plotEl, originalLayout);
                    btn.innerText = 'Export Figure';
                });
            });
        });

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
