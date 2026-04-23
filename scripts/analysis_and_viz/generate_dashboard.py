import os
import csv
import json
import re
import pandas as pd
from datetime import datetime

# --- Configuration Constants ---
DEFAULT_START_DIR = "/scratch/ar7789/llm_dataset_inference/aggregated_results/p_values/mean+p-value-outliers"
OUTPUT_DIR = "/scratch/ar7789/llm_dataset_inference/aggregated_results/p_values/agg_plots"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dashboard.html")

def parse_model_info(model_dir_name):
    """
    Parses model directory name to extract family, quantization type, bit-width, and config.
    Example: _scratch_ar7789_llm_quant_saved_qmodels_tmp_EleutherAI_pythia-12b-deduped-gptq-b4-gs128-da1
    """
    # Default values
    family = model_dir_name
    quant_type = "Original"
    bit_width = "fp16"
    config = "Default"

    # Check for quantization patterns
    # Look for common families
    families = ["pythia-160m", "pythia-410m", "pythia-1b", "pythia-2.8b", "pythia-6.9b", "pythia-12b"]
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
    
    return family, quant_type, bit_width, bit_sort_key, config

def aggregate_data(start_dir):
    """
    Aggregates p-values from all CSV files in the directory structure.
    """
    data_points = []
    
    for root, dirs, files in os.walk(start_dir):
        for file in files:
            if file.endswith(".csv"):
                dataset_name = os.path.splitext(file)[0]
                file_path = os.path.join(root, file)
                model_subdir = os.path.basename(root)
                
                # Metadata
                is_false_positive = "false_positive" in root
                run_type = "False Positive" if is_false_positive else "Regular"
                family, quant_type, bit_width, bit_sort_key, config = parse_model_info(model_subdir)
                
                try:
                    df = pd.read_csv(file_path)
                    if df.empty or 'p_500' not in df.columns:
                        continue
                    
                    df['p_500'] = pd.to_numeric(df['p_500'], errors='coerce')
                    
                    # Look for cohens_d
                    d_path = file_path.replace('p_values', 'cohens_d')
                    if os.path.exists(d_path):
                        df_d = pd.read_csv(d_path)
                        if 'seed' in df.columns and 'seed' in df_d.columns:
                            df = df.merge(df_d, on='seed', how='left')
                        else:
                            df['d_500'] = df_d['d_500'] if 'd_500' in df_d.columns else None
                            
                    if 'd_500' in df.columns:
                        df['d_500'] = pd.to_numeric(df['d_500'], errors='coerce')
                    else:
                        df['d_500'] = None

                    for _, row in df.iterrows():
                        p = row['p_500']
                        d = row['d_500']
                        seed = row['seed'] if 'seed' in df.columns else 'unknown'
                        
                        if pd.isna(p): continue
                        
                        data_points.append({
                            "family": family,
                            "quant_type": quant_type,
                            "bit_width": bit_width,
                            "bit_sort_key": bit_sort_key,
                            "config": config,
                            "dataset": dataset_name,
                            "run_type": run_type,
                            "p_value": p,
                            "cohens_d": d if not pd.isna(d) else None,
                            "seed": seed,
                            "model_name": model_subdir
                        })
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    
    return data_points

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>P-Value / Cohen's d Dashboard</title>
    
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
            line-height: 1.6;
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 3rem;
            text-align: center;
        }

        h1 {
            font-size: 3rem;
            font-weight: 600;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        /* Glassmorphism Card Style */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 1rem;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        /* Filter Controls */
        .filters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 1rem;
        }

        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .filter-group label {
            font-weight: 600;
            color: var(--accent-color);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            color: var(--text-color);
            padding: 0.6rem;
            border-radius: 0.5rem;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            outline: none;
            transition: border-color 0.2s;
        }

        select:focus {
            border-color: var(--accent-color);
        }

        .filters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .toggle-group {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid var(--glass-border);
            grid-column: 1 / -1;
        }

        .filter-group label {
            font-weight: 600;
            color: var(--accent-color);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            color: var(--text-color);
            padding: 0.6rem;
            border-radius: 0.5rem;
            font-family: 'Outfit', sans-serif;
            cursor: pointer;
            outline: none;
            transition: border-color 0.2s;
        }

        select:focus {
            border-color: var(--accent-color);
        }

        /* Toggle Switch Styling */
        .switch {
            position: relative;
            display: inline-block;
            width: 44px;
            height: 24px;
        }

        .switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #334155;
            transition: .4s;
            border-radius: 24px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }

        input:checked + .slider {
            background-color: var(--accent-color);
        }

        input:checked + .slider:before {
            transform: translateX(20px);
        }

        .toggle-item {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
            font-weight: 500;
        }

        /* Plot Area */
        #plot-container {
            min-height: 600px;
            width: 100%;
        }

        .stats-summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .stat-item {
            text-align: center;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 0.5rem;
        }

        .stat-value {
            display: block;
            font-size: 1.5rem;
            font-weight: 600;
            color: var(--accent-color);
        }

        .stat-label {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        /* Multiselect Styling (Simulated with size) */
        select[multiple] {
            height: 120px;
        }

        .footer {
            text-align: center;
            margin-top: 4rem;
            color: var(--text-muted);
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>P-Value & Cohen's d Dashboard</h1>
            <p class="subtitle">Interactive distribution analysis of model outputs (p_500 & d_500)</p>
        </header>

        <section class="glass-card">
            <div class="filters">
                <div class="filter-group">
                    <label for="filter-family">Model Family</label>
                    <select id="filter-family" multiple></select>
                </div>
                <div class="filter-group">
                    <label for="filter-quant">Quant Type</label>
                    <select id="filter-quant" multiple></select>
                </div>
                <div class="filter-group">
                    <label for="filter-bits">Bit Width</label>
                    <select id="filter-bits" multiple></select>
                </div>
                <div class="filter-group">
                    <label for="filter-config">Config</label>
                    <select id="filter-config" multiple></select>
                </div>
                <div class="filter-group">
                    <label for="filter-dataset">Dataset</label>
                    <select id="filter-dataset" multiple></select>
                </div>
                <div class="filter-group">
                    <label for="x-axis-select">Group By (X-Axis)</label>
                    <select id="x-axis-select">
                        <option value="model" selected>Model Variant (Default)</option>
                        <option value="family">Model Family</option>
                        <option value="quant_type">Quant Type</option>
                        <option value="bit_width">Bit Width</option>
                        <option value="dataset">Dataset</option>
                    </select>
                </div>

                <div class="toggle-group">
                    <div class="filter-group" style="flex-grow: 1;">
                        <label for="filter-runtype">Run Type</label>
                        <select id="filter-runtype" multiple>
                            <option value="Regular" selected>Regular (TP)</option>
                            <option value="False Positive" selected>False Positive (FP)</option>
                        </select>
                    </div>
                    <div class="toggle-item" style="margin-left: 2rem;">
                        <span>Box Plot</span>
                        <label class="switch">
                            <input type="checkbox" id="toggle-plot-type">
                            <span class="slider"></span>
                        </label>
                        <span>Scatter</span>
                    </div>
                    <div class="toggle-item" style="margin-left: 2rem;">
                        <span>P-Value</span>
                        <label class="switch">
                            <input type="checkbox" id="toggle-metric-type">
                            <span class="slider"></span>
                        </label>
                        <span>Cohen's d</span>
                    </div>
                </div>
            </div>
            <div style="text-align: right; margin-top: 1rem;">
                <button id="reset-filters" style="background: none; border: 1px solid var(--accent-color); color: var(--accent-color); padding: 0.4rem 1rem; border-radius: 0.4rem; cursor: pointer;">Reset All</button>
            </div>
        </section>

        <section class="glass-card">
            <div id="plot-container"></div>
            <div class="stats-summary" id="stats-summary">
                <!-- Data will be populated by JS -->
            </div>
        </section>

        <footer class="footer">
            Generated on: <span id="gen-date"></span> | Dataset Inference Pipeline
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
            runType: document.getElementById('filter-runtype'),
            xAxisSelect: document.getElementById('x-axis-select'),
            togglePlotType: document.getElementById('toggle-plot-type'),
            toggleMetricType: document.getElementById('toggle-metric-type')
        };

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

            filters.quant.addEventListener('change', () => {
                const selectedQuants = getSelectedValues(filters.quant);
                const relevantConfigs = [...new Set(rawData
                    .filter(d => selectedQuants.includes(d.quant_type))
                    .map(d => d.config))].sort();
                
                const prevSelected = getSelectedValues(filters.config);
                const prevAvailable = Array.from(filters.config.options).map(opt => opt.value);
                
                filters.config.innerHTML = relevantConfigs.map(it => {
                    let isSelected = true;
                    if (prevAvailable.includes(it)) {
                        isSelected = prevSelected.includes(it);
                    }
                    return `<option value="${it}" ${isSelected ? 'selected' : ''}>${it}</option>`;
                }).join('');
                
                updateDashboard();
            });

            ['family', 'bits', 'config', 'dataset', 'runType', 'togglePlotType', 'toggleMetricType', 'xAxisSelect'].forEach(key => {
                if(filters[key]) filters[key].addEventListener('change', updateDashboard);
            });

            document.getElementById('reset-filters').addEventListener('click', () => {
                Object.values(filters).forEach(el => {
                    if (el.type === 'checkbox') {
                        el.checked = false;
                    } else if (el.multiple) {
                        Array.from(el.options).forEach(opt => opt.selected = true);
                    } else if (el.id === 'x-axis-select') {
                        el.value = 'model';
                    }
                });
                populate(filters.config, getUnique('config'));
                updateDashboard();
            });
            
            document.getElementById('gen-date').innerText = new Date().toLocaleString();
        }

        function getSelectedValues(el) {
            return Array.from(el.selectedOptions).map(opt => opt.value);
        }

        function updateDashboard() {
            const selected = {
                family: getSelectedValues(filters.family),
                quant: getSelectedValues(filters.quant),
                bits: getSelectedValues(filters.bits),
                config: getSelectedValues(filters.config),
                dataset: getSelectedValues(filters.dataset),
                runType: getSelectedValues(filters.runType),
                isScatter: filters.togglePlotType.checked,
                isMetricD: filters.toggleMetricType.checked,
                xAxisChoice: filters.xAxisSelect.value
            };

            const filteredData = rawData.filter(d => 
                selected.family.includes(d.family) &&
                selected.quant.includes(d.quant_type) &&
                selected.bits.includes(d.bit_width) &&
                selected.config.includes(d.config) &&
                selected.dataset.includes(d.dataset) &&
                selected.runType.includes(d.run_type)
            );

            renderPlot(filteredData, selected.isScatter, selected.isMetricD, selected.xAxisChoice);
            updateStats(filteredData, selected.isMetricD);
        }

        function renderPlot(data, isScatter, isMetricD, xAxisChoice) {
            const metricKey = isMetricD ? 'cohens_d' : 'p_value';
            const validData = data.filter(d => d[metricKey] !== null && d[metricKey] !== undefined);

            if (validData.length === 0) {
                document.getElementById('plot-container').innerHTML = '<div style="display: flex; height: 100%; align-items: center; justify-content: center; color: var(--text-muted);">No data matches current filters</div>';
                return;
            }

            const regular = validData.filter(d => d.run_type === 'Regular');
            const fp = validData.filter(d => d.run_type === 'False Positive');

            const traces = [];

            const commonProps = (name, color, subset) => ({
                name: name,
                type: 'box',
                marker: { color: color, opacity: 0.8, size: isScatter ? 5 : 6 },
                boxpoints: isScatter ? 'all' : 'outliers',
                jitter: 0.5,
                pointpos: isScatter ? 0 : -1.8,
                fillcolor: isScatter ? 'rgba(0,0,0,0)' : color + '33',
                line: { 
                    color: isScatter ? 'rgba(0,0,0,0)' : color, 
                    width: isScatter ? 0 : 2 
                },
                whiskerwidth: isScatter ? 0 : 0.5,
                notched: false,
                hoverinfo: 'y+text',
                text: subset.map(d => `Seed: ${d.seed}`)
            });

            // Sort data by family and bit_sort_key for consistent x-axis order
            const sortedData = [...validData].sort((a, b) => {
                if (a.family !== b.family) return a.family.localeCompare(b.family);
                if (a.bit_sort_key !== b.bit_sort_key) return a.bit_sort_key - b.bit_sort_key;
                return a.config.localeCompare(b.config);
            });

            const getXLabel = (d) => {
                if (xAxisChoice === 'family') return d.family;
                if (xAxisChoice === 'quant_type') return d.quant_type;
                if (xAxisChoice === 'bit_width') return d.bit_width;
                if (xAxisChoice === 'dataset') return d.dataset;
                return d.config === 'Default' ? `${d.family}\\n${d.quant_type}-${d.bit_width}` : `${d.family}\\n${d.quant_type}-${d.bit_width}\\n${d.config}`;
            };
            
            let uniqueConfigs = Array.from(new Set(sortedData.map(getXLabel)));
            if (xAxisChoice !== 'model') {
                uniqueConfigs.sort();
            }
            
            if (regular.length > 0) {
                const tr = commonProps('Regular (TP)', '#38bdf8', regular);
                tr.y = regular.map(d => d[metricKey]);
                tr.x = regular.map(getXLabel);
                traces.push(tr);
            }

            if (fp.length > 0) {
                const tr = commonProps('False Positive (FP)', '#f87171', fp);
                tr.y = fp.map(d => d[metricKey]);
                tr.x = fp.map(getXLabel);
                traces.push(tr);
            }

            const shapes = isMetricD ? [] : [
                { type: 'line', y0: 0.1, y1: 0.1, x0: 0, x1: 1, xref: 'paper', line: { color: '#94a3b8', width: 2, dash: 'dot' } },
                { type: 'line', y0: 0.5, y1: 0.5, x0: 0, x1: 1, xref: 'paper', line: { color: '#94a3b8', width: 2, dash: 'dot' } }
            ];

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Outfit, sans-serif', color: '#f1f5f9' },
                yaxis: { 
                    title: isMetricD ? "Cohen's d" : 'P-Value', 
                    gridcolor: 'rgba(255,255,255,0.1)', 
                    range: isMetricD ? null : [-0.05, 1.1] 
                },
                xaxis: { 
                    gridcolor: 'rgba(255,255,255,0.1)', 
                    tickangle: -45,
                    categoryorder: 'array',
                    categoryarray: uniqueConfigs
                },
                boxmode: 'group',
                margin: { t: 40, b: 120, l: 60, r: 20 },
                legend: { orientation: 'h', y: 1.1 },
                shapes: shapes
            };

            Plotly.newPlot('plot-container', traces, layout, { responsive: true });
        }

        function updateStats(data, isMetricD) {
            const metricKey = isMetricD ? 'cohens_d' : 'p_value';
            const validData = data.filter(d => d[metricKey] !== null && d[metricKey] !== undefined);

            const summary = document.getElementById('stats-summary');
            const total = validData.length;
            const regCount = validData.filter(d => d.run_type === 'Regular').length;
            const fpCount = validData.filter(d => d.run_type === 'False Positive').length;
            
            let extraStat = '';
            if (!isMetricD) {
                const outliers = validData.filter(d => d.p_value < 0.1).length;
                extraStat = `
                    <div class="stat-item">
                        <span class="stat-value">${((outliers/total || 0) * 100).toFixed(1)}%</span>
                        <span class="stat-label">Significant (p < 0.1)</span>
                    </div>
                `;
            } else {
                const highEffect = validData.filter(d => Math.abs(d.cohens_d) > 0.8).length;
                extraStat = `
                    <div class="stat-item">
                        <span class="stat-value">${((highEffect/total || 0) * 100).toFixed(1)}%</span>
                        <span class="stat-label">Large Effect (|d| > 0.8)</span>
                    </div>
                `;
            }

            summary.innerHTML = `
                <div class="stat-item">
                    <span class="stat-value">${total}</span>
                    <span class="stat-label">Total Points</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${regCount}</span>
                    <span class="stat-label">Regular Runs</span>
                </div>
                <div class="stat-item">
                    <span class="stat-value">${fpCount}</span>
                    <span class="stat-label">False Positives</span>
                </div>
                ${extraStat}
            `;
        }

        initFilters();
        updateDashboard();
    </script>
</body>
</html>
"""

def main():
    print(f"Aggregating data from: {DEFAULT_START_DIR}")
    data = aggregate_data(DEFAULT_START_DIR)
    
    if not data:
        print("No data found!")
        return

    print(f"Found {len(data)} data points. Generating dashboard...")
    
    # Inject data into template
    json_data = json.dumps(data)
    final_html = HTML_TEMPLATE.replace("[DATA_JSON]", json_data)
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(final_html)
        
    print(f"Dashboard generated successfully: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
