import os
import json
import re
import math
from datetime import datetime

# --- Configuration Constants ---
RESULTS_DIR = "/scratch/ar7789/llm_dataset_inference/results"
OUTPUT_DIR = "/scratch/ar7789/llm_dataset_inference/aggregated_results/raw_metrics"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dashboard.html")

def calculate_percentile(data, percentile):
    if not data:
        return 0
    data_sorted = sorted(data)
    n = len(data_sorted)
    if n == 1:
        return data_sorted[0]
    
    index = (n - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return data_sorted[lower]
    
    return data_sorted[lower] * (upper - index) + data_sorted[upper] * (index - lower)

def parse_model_info(model_dir_name):
    """
    Parses model directory name to extract family, quantization type, and bit-width.
    Example: _scratch_ar7789_llm_quant_saved_qmodels_tmp_EleutherAI_pythia-12b-deduped-gptq-b4-gs128-da1
    """
    family = model_dir_name
    quant_type = "Original"
    bit_width = "fp16"
    bit_sort_key = 99

    # Look for common families
    families = ["pythia-160m", "pythia-410m", "pythia-1.4b", "pythia-1b", "pythia-2.8b", "pythia-6.9b", "pythia-12b"]
    for f in families:
        if f in model_dir_name:
            family = f
            break
    
    # Extract bit-width and quant type
    bit_match = re.search(r'-b(\d+)', model_dir_name)
    if bit_match:
        bits = int(bit_match.group(1))
        bit_width = f"b{bits}"
        bit_sort_key = bits
        if "gptq" in model_dir_name.lower():
            quant_type = "GPTQ"
        elif "awq" in model_dir_name.lower():
            quant_type = "AWQ"
    
    return family, quant_type, bit_width, bit_sort_key

def aggregate_raw_metrics(results_dir):
    """
    Aggregates stats from JSON metric files.
    """
    data_points = []
    
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_metrics.json"):
                parts = file.replace("_metrics.json", "").split("_")
                if len(parts) < 2:
                    continue
                
                split = parts[-1] # train or val
                dataset_name = "_".join(parts[:-1])
                
                file_path = os.path.join(root, file)
                model_subdir = os.path.basename(root)
                
                family, quant_type, bit_width, bit_sort_key = parse_model_info(model_subdir)
                
                try:
                    with open(file_path, 'r') as f:
                        metrics_dict = json.load(f)
                    
                    for metric_name, values in metrics_dict.items():
                        if not isinstance(values, list) or not values:
                            continue
                        
                        # Use manual stats calculation
                        v_min = float(min(values))
                        v_max = float(max(values))
                        v_mean = float(sum(values) / len(values))
                        v_q5 = calculate_percentile(values, 5)
                        v_q25 = calculate_percentile(values, 25)
                        v_median = calculate_percentile(values, 50)
                        v_q75 = calculate_percentile(values, 75)
                        v_q95 = calculate_percentile(values, 95)

                        stats = {
                            "min": v_min,
                            "q5": v_q5,
                            "q25": v_q25,
                            "median": v_median,
                            "q75": v_q75,
                            "q95": v_q95,
                            "max": v_max,
                            "mean": v_mean,
                        }
                        
                        data_points.append({
                            "family": family,
                            "quant_type": quant_type,
                            "bit_width": bit_width,
                            "bit_sort_key": bit_sort_key,
                            "dataset": dataset_name,
                            "split": split,
                            "metric": metric_name,
                            **stats,
                            "model_name": model_subdir
                        })
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
    return data_points

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raw Metrics Dashboard</title>
    
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

        * { box-sizing: border-box; margin: 0; padding: 0; }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            padding: 2rem;
        }

        .container { max-width: 1400px; margin: 0 auto; }

        header { margin-bottom: 2rem; text-align: center; }

        h1 {
            font-size: 2.5rem;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            border-radius: 1rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }

        .filters {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
        }

        .filter-group { display: flex; flex-direction: column; gap: 0.3rem; }
        .filter-group label { font-weight: 600; color: var(--accent-color); font-size: 0.8rem; text-transform: uppercase; }

        select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            color: var(--text-color);
            padding: 0.5rem;
            border-radius: 0.4rem;
            font-family: 'Outfit', sans-serif;
        }

        #plot-container { min-height: 600px; width: 100%; }

        .footer { text-align: center; margin-top: 2rem; color: var(--text-muted); font-size: 0.8rem; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Raw Metrics Analysis Dashboard</h1>
            <p style="color: var(--text-muted)">Visualizing distributions of Membership Inference Metrics</p>
        </header>

        <section class="glass-card">
            <div class="filters">
                <div class="filter-group">
                    <label>Metric</label>
                    <select id="filter-metric"></select>
                </div>
                <div class="filter-group">
                    <label>Dataset</label>
                    <select id="filter-dataset"></select>
                </div>
                <div class="filter-group">
                    <label>Split</label>
                    <select id="filter-split">
                        <option value="both" selected>Both (Compare)</option>
                        <option value="train">Train only</option>
                        <option value="val">Val only</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Model Family</label>
                    <select id="filter-family" multiple size="4"></select>
                </div>
                <div class="filter-group">
                    <label>Quant Type</label>
                    <select id="filter-quant" multiple size="4"></select>
                </div>
            </div>
            <div style="text-align: right; margin-top: 1rem;">
                <button onclick="resetFilters()" style="background:none; border:1px solid var(--accent-color); color:var(--accent-color); padding: 0.3rem 0.8rem; border-radius:0.4rem; cursor:pointer;">Reset</button>
            </div>
        </section>

        <section class="glass-card">
            <div id="plot-container"></div>
        </section>

        <footer class="footer">
            Generated on: <span id="gen-date"></span>
        </footer>
    </div>

    <script>
        const rawData = [DATA_JSON];
        
        const ui = {
            metric: document.getElementById('filter-metric'),
            dataset: document.getElementById('filter-dataset'),
            split: document.getElementById('filter-split'),
            family: document.getElementById('filter-family'),
            quant: document.getElementById('filter-quant')
        };

        function init() {
            const getUnique = (key) => [...new Set(rawData.map(d => d[key]))].sort();
            
            const populate = (el, items, multi=false) => {
                el.innerHTML = items.map(it => `<option value="${it}" ${multi?'selected':''}>${it}</option>`).join('');
                if(!multi && items.length > 0) el.selectedIndex = 0;
            };

            populate(ui.metric, getUnique('metric'));
            populate(ui.dataset, getUnique('dataset'));
            populate(ui.family, getUnique('family'), true);
            populate(ui.quant, getUnique('quant_type'), true);

            Object.values(ui).forEach(el => el.addEventListener('change', update));
            document.getElementById('gen-date').innerText = new Date().toLocaleString();
            update();
        }

        function getSelected(el) {
            return Array.from(el.selectedOptions).map(o => o.value);
        }

        function resetFilters() {
            Array.from(ui.family.options).forEach(o => o.selected = true);
            Array.from(ui.quant.options).forEach(o => o.selected = true);
            ui.split.value = 'both';
            update();
        }

        function update() {
            const metric = ui.metric.value;
            const dataset = ui.dataset.value;
            const splitMode = ui.split.value;
            const families = getSelected(ui.family);
            const quants = getSelected(ui.quant);

            let filtered = rawData.filter(d => 
                d.metric === metric && 
                d.dataset === dataset && 
                families.includes(d.family) &&
                quants.includes(d.quant_type)
            );

            if (splitMode !== 'both') {
                filtered = filtered.filter(d => d.split === splitMode);
            }

            render(filtered, splitMode === 'both');
        }

        function render(data, compareSplits) {
            if (data.length === 0) {
                document.getElementById('plot-container').innerHTML = '<div style="text-align:center; padding:100px; color:var(--text-muted)">No data matches filters</div>';
                return;
            }

            // Sort by family and bit_sort_key
            data.sort((a,b) => {
                if(a.family !== b.family) return a.family.localeCompare(b.family);
                return a.bit_sort_key - b.bit_sort_key;
            });

            const traces = [];
            const splits = compareSplits ? ['train', 'val'] : [data[0].split];
            const colors = { 'train': '#38bdf8', 'val': '#f87171' };

            splits.forEach(s => {
                const subset = data.filter(d => d.split === s);
                if (subset.length === 0) return;

                traces.push({
                    name: s.toUpperCase(),
                    type: 'box',
                    x: subset.map(d => `${d.family}<br>${d.quant_type}-${d.bit_width}`),
                    q1: subset.map(d => d.q25),
                    median: subset.map(d => d.median),
                    q3: subset.map(d => d.q75),
                    lowerfence: subset.map(d => d.min),
                    upperfence: subset.map(d => d.max),
                    mean: subset.map(d => d.mean),
                    marker: { color: colors[s] },
                    boxmean: true
                });
            });

            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: { family: 'Outfit, sans-serif', color: '#f1f5f9' },
                xaxis: { tickangle: -45, gridcolor: 'rgba(255,255,255,0.1)' },
                yaxis: { title: ui.metric.value, gridcolor: 'rgba(255,255,255,0.1)' },
                boxmode: 'group',
                margin: { t: 20, b: 150, l: 80, r: 20 },
                legend: { orientation: 'h', y: 1.1 }
            };

            Plotly.newPlot('plot-container', traces, layout, { responsive: true });
        }

        init();
    </script>
</body>
</html>
"""

def main():
    print(f"Aggregating raw metrics from: {RESULTS_DIR}")
    data = aggregate_raw_metrics(RESULTS_DIR)
    
    if not data:
        print("No metrics data found in results directory!")
        return

    print(f"Computed stats for {len(data)} metric distributions. Generating dashboard...")
    
    json_data = json.dumps(data)
    final_html = HTML_TEMPLATE.replace("[DATA_JSON]", json_data)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(final_html)
        
    print(f"Raw metrics dashboard generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
