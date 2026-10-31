"""pyvis rendering of an API /graph/neighbours result (colors follow the entity label)."""

from pyvis.network import Network

COLORS = {"Organization": "#2a78d6", "Person": "#eb6834", "AIModel": "#1baf7a", "Paper": "#8a63d2",
          "Technology": "#eda100", "Dataset": "#d6457a", "Product": "#4aa3a2", "Location": "#7a7a7a"}


def subgraph_html(g: dict) -> str:
    net = Network(height="400px", width="100%", directed=True, bgcolor="#ffffff", font_color="#141b2d",
                  cdn_resources="in_line")
    c = g["center"]
    net.add_node(c["id"], label=c["name"], color=COLORS.get(c["label"], "#999"), size=28, title=c["label"])
    for e in g["edges"]:
        net.add_node(e["id"], label=e["name"], color=COLORS.get(e["label"], "#999"), size=16, title=e["label"])
        net.add_edge(e["source"], e["target"], label=e["relation"], title=(e.get("evidence") or "")[:200],
                     font={"size": 9}, color="#9aa3b2")
    net.set_options('{"physics": {"barnesHut": {"springLength": 140}}, "edges": {"arrows": {"to": {"scaleFactor": 0.5}}}}')
    return net.generate_html(notebook=False)
