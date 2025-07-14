import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
st.set_page_config(page_title='AI-Guided Nuclear Waste Transmutation Dashboard', layout='wide')
from nuclearai import data, gnn, optimization, visualization
from nuclearai.visualization import get_3d_network_figure  # Explicit import for linter
import torch
import numpy as np
import yaml
import matplotlib.pyplot as plt
import io
import json

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

st.title('🧬 AI-Guided Nuclear Waste Transmutation Dashboard')

# Sidebar: Help toggle
show_help = st.sidebar.checkbox('Show Help', value=False)
if show_help:
    st.sidebar.info("""
    **How to use this dashboard:**
    - Select or upload a scenario file (YAML/JSON) in the sidebar.
    - Explore the tabs: Network, GNN Output, Optimization, Visualizations.
    - Use the 2D/3D visualization buttons for interactive exploration.
    - For custom data, upload your own scenario file.
    - Edit the scenario in-browser and export your changes.
    """)

# Sidebar: Scenario selection and file uploader
scenario_dir = st.sidebar.text_input('Scenario directory', 'nuclearai/scenario_templates')
try:
    scenarios = [f for f in os.listdir(scenario_dir) if f.endswith('.yaml') or f.endswith('.json')]
except Exception as e:
    st.sidebar.error(f"Could not list scenarios: {e}")
    scenarios = []
scenario_file = st.sidebar.selectbox('Select scenario', scenarios)

uploaded_file = st.sidebar.file_uploader('Or upload your own scenario (YAML/JSON)', type=['yaml', 'yml', 'json'])

# Scenario editing and state
if 'scenario_text' not in st.session_state:
    st.session_state['scenario_text'] = ''

if uploaded_file is not None:
    if uploaded_file.name.endswith('.yaml') or uploaded_file.name.endswith('.yml'):
        scenario_text = uploaded_file.read().decode('utf-8') or ''
        scenario = yaml.safe_load(scenario_text)
    else:
        scenario_text = uploaded_file.read().decode('utf-8') or ''
        scenario = json.loads(scenario_text)
    st.session_state['scenario_text'] = scenario_text
    st.success(f'Loaded custom scenario: {uploaded_file.name}')
elif scenario_file:
    with open(os.path.join(scenario_dir, scenario_file)) as f:
        scenario_text = f.read() or ''
        scenario = yaml.safe_load(scenario_text)
    st.session_state['scenario_text'] = scenario_text
    st.success(f'Loaded scenario: {scenario_file}')
else:
    scenario = None
    scenario_text = ''

# Scenario editor
if scenario:
    st.sidebar.markdown('---')
    st.sidebar.markdown('**Edit Scenario YAML/JSON**')
    edited_text = st.sidebar.text_area('Scenario Editor', st.session_state['scenario_text'] or '', height=300)
    if edited_text != (st.session_state['scenario_text'] or ''):
        try:
            if scenario_file and scenario_file.endswith('.json'):
                scenario = json.loads(edited_text or '')
            else:
                scenario = yaml.safe_load(edited_text or '')
            st.session_state['scenario_text'] = edited_text or ''
            st.sidebar.success('Scenario updated!')
        except Exception as e:
            st.sidebar.error(f'Invalid YAML/JSON: {e}')
    # Download/export scenario
    st.sidebar.markdown('---')
    st.sidebar.download_button('Download Scenario', edited_text or '', file_name='edited_scenario.yaml')

if scenario:
    # Build database and graph
    db = data.NuclearDatabase.from_dicts(scenario['isotopes'], scenario['reactions'])
    G = db.build_graph()

    # Tabs for UX
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Network", "Isotopes", "GNN Output", "Optimization", "Visualizations"])

    with tab1:
        st.subheader('Isotope Network')
        if show_help:
            st.info("""
            **Network Tab:**
            Visualizes the isotope transmutation network as a 2D graph. Nodes are isotopes, edges are reactions.
            """)
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = visualization.nx.spring_layout(G)
        node_colors = ['red' if n == scenario['isotopes'][0]['name'] else 'skyblue' for n in G.nodes]
        visualization.nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=800, ax=ax)
        st.pyplot(fig)
        st.subheader('3D Network Visualization')
        try:
            fig3d = get_3d_network_figure(G, title="3D Isotope Network")
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.warning(f"3D plot error: {e}")

    with tab2:
        st.subheader('Isotope Table')
        if show_help:
            st.info("""
            **Isotopes Tab:**
            View and filter all isotopes in the scenario. You can search, sort, and export the table.
            """)
        iso_df = None
        try:
            iso_df = None
            if scenario['isotopes']:
                iso_df = st.dataframe(scenario['isotopes'], use_container_width=True)
        except Exception as e:
            st.warning(f"Isotope table error: {e}")
        if iso_df is not None:
            csv = io.StringIO()
            import pandas as pd
            pd.DataFrame(scenario['isotopes']).to_csv(csv, index=False)
            st.download_button('Download Isotope Table (CSV)', csv.getvalue(), file_name='isotopes.csv')

    # Prepare GNN input (numeric features only)
    feature_keys = ['half_life', 'radiotox', 'decay_heat']
    x = []
    for iso in scenario['isotopes']:
        features = []
        for k in feature_keys:
            v = iso.get(k, 0.0)
            features.append(float(v) if is_number(v) else 0.0)
        cs = list(iso.get('cross_sections', {}).values())
        cs_val = 0.0
        if cs:
            try:
                cs_val = float(cs[0]) if is_number(cs[0]) else 0.0
            except Exception:
                cs_val = 0.0
        features.append(cs_val)
        x.append(features)
    x = torch.tensor(x, dtype=torch.float)
    node_list = list(G.nodes)
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    edge_list = [(node_to_idx[a], node_to_idx[b]) for a, b in G.edges]
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2,0), dtype=torch.long)
    from torch_geometric.data import Data as GeoData
    geo_data = GeoData(x=x, edge_index=edge_index)

    with tab3:
        st.subheader('GNN Output')
        if show_help:
            st.info("""
            **GNN Output Tab:**
            Shows the output of the Graph Neural Network for the current scenario. Each row is an isotope, columns are learned features.
            """)
        # Advanced GNN controls
        gnn_type = st.selectbox('GNN Model Type', ['gcn', 'gat'], index=0)
        hidden_dim = st.slider('Hidden Layer Size', 4, 64, 8)
        out_dim = st.slider('Output Layer Size', 2, 16, 4)
        model = gnn.TransmutationGNN(in_channels=4, hidden_channels=hidden_dim, out_channels=out_dim, model_type=gnn_type)
        out = model(geo_data)
        st.write('GNN output tensor:', out.detach().numpy())
        st.dataframe(out.detach().numpy(), use_container_width=True)
        # Download GNN output
        gnn_csv = io.StringIO()
        pd.DataFrame(out.detach().numpy()).to_csv(gnn_csv, index=False)
        st.download_button('Download GNN Output (CSV)', gnn_csv.getvalue(), file_name='gnn_output.csv')

    with tab4:
        st.subheader('Optimization')
        if show_help:
            st.info("""
            **Optimization Tab:**
            Shows the multi-objective optimization cost for the scenario. Lower is better. You can adjust weights and objectives.
            """)
        # Advanced optimization controls
        radiotox_weight = st.slider('Radiotoxicity Weight', 0.0, 2.0, 1.0)
        decay_heat_weight = st.slider('Decay Heat Weight', 0.0, 2.0, 0.5)
        cost_fn = optimization.MultiObjectiveCost({'radiotox': radiotox_weight, 'decay_heat': decay_heat_weight})
        mask = torch.arange(len(scenario['isotopes']))
        objectives = {'radiotox': 1, 'decay_heat': 2}
        cost = cost_fn(out, mask, objectives)
        st.metric('Optimization cost', float(cost) if hasattr(cost, 'item') else cost)
        # Download optimization result
        st.download_button('Download Optimization Result', str(float(cost) if hasattr(cost, 'item') else cost), file_name='optimization_result.txt')

    with tab5:
        st.subheader('Visualizations')
        if show_help:
            st.info("""
            **Visualizations Tab:**
            Explore advanced visualizations: Sankey diagrams, 2D/3D networks, and more.
            """)
        st.write('Sankey diagram:')
        # Dynamically generate Sankey flows from scenario data
        try:
            labels = []
            flows = []
            label_to_idx = {}
            for rxn in scenario['reactions']:
                for iso in [rxn['from_iso'], rxn['to_iso']]:
                    if iso not in label_to_idx:
                        label_to_idx[iso] = len(labels)
                        labels.append(iso)
            sankey_flows = [0] * len(labels)
            for rxn in scenario['reactions']:
                from_idx = label_to_idx[rxn['from_iso']]
                to_idx = label_to_idx[rxn['to_iso']]
                sankey_flows[from_idx] += 1
                sankey_flows[to_idx] -= 1
            flows = [f for f in sankey_flows if f != 0]
            labels = [l for f, l in zip(sankey_flows, labels) if f != 0]
            if not flows:
                flows = [1, -1]
                labels = ['Input', 'Output']
            fig2 = plt.figure(figsize=(6, 3))
            visualization.plot_sankey_flows(flows, labels, title="Isotope Flow Sankey Diagram")
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"Sankey plot error: {e}")
        st.write('2D Network:')
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        pos = visualization.nx.spring_layout(G)
        visualization.nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=800, ax=ax4)
        st.pyplot(fig4)
        st.write('3D Network:')
        try:
            fig3d = get_3d_network_figure(G, title="3D Isotope Network")
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.warning(f"3D plot error: {e}")
else:
    st.info('Please select or upload a scenario from the sidebar.') 