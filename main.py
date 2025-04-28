from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from dbscan_clusterer import DBSCANClusterer, optimize_eps_and_min_samples
import os
from sklearn.cluster import DBSCAN

load_dotenv()

app = FastAPI()

clusterer = DBSCANClusterer(
    username=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD"),
    host=os.getenv("POSTGRES_HOST"),
    port=int(os.getenv("POSTGRES_PORT")),
    database=os.getenv("POSTGRES_DB")
)

GRAPH_DIR = "./graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

@app.get("/cluster/countries")
def cluster_countries():
    query = """
    SELECT c.COUNTRY, 
           COUNT(DISTINCT od.order_id) AS total_order_count,
           AVG(od.unit_price * od.quantity) AS avg_order_value,
           SUM(od.quantity) * 1.0 / COUNT(DISTINCT od.order_id) AS avg_quantity_per_order
    FROM order_details od
    JOIN orders o ON od.order_id = o.order_id
    JOIN customers c ON o.customer_id = c.customer_id
    GROUP BY c.COUNTRY
    """
    df = clusterer.fetch_data(query)

    elbow_path = f"{GRAPH_DIR}/countries_elbow.png"
    scatter_path = f"{GRAPH_DIR}/countries_scatter.png"

    X = df[["total_order_count", "avg_order_value", "avg_quantity_per_order"]]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eps_initial = clusterer.find_optimal_eps(X_scaled, min_samples=3, save_path=elbow_path)

    optimal_eps, optimal_min_samples = optimize_eps_and_min_samples(X_scaled, eps_range=(0.5, 5.0), min_samples_range=(2, 10))

    dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
    labels = dbscan.fit_predict(X_scaled)
    df["cluster"] = labels

    clusterer.plot_clusters(df, x_feature="total_order_count", y_feature="avg_order_value", save_path=scatter_path)

    outliers = df[df["cluster"] == -1]

    return {
        "optimal_eps": optimal_eps,
        "optimal_min_samples": optimal_min_samples,
        "elbow_graph_path": elbow_path,
        "scatter_graph_path": scatter_path,
        "outliers": outliers.to_dict(orient="records"),
        "n_clusters": df["cluster"].nunique() - (1 if -1 in df["cluster"].unique() else 0)
    }

@app.get("/cluster/products")
def cluster_products():
    query = """
    SELECT 
        p.product_id,
        AVG(od.unit_price * od.quantity) AS avg_order_value,
        COUNT(od.quantity) AS sales_frequency,
        SUM(od.quantity) * 1.0 / COUNT(DISTINCT od.order_id) AS avg_quantity_per_order,
        COUNT(DISTINCT o.customer_id) as different_customer
    FROM 
        order_details od 
    JOIN  
        products p ON p.product_id = od.product_id
    JOIN orders o on od.order_id = o.order_id
    GROUP BY 
        p.product_id
    """
    df = clusterer.fetch_data(query)

    elbow_path = f"{GRAPH_DIR}/products_elbow.png"
    scatter_path = f"{GRAPH_DIR}/products_scatter.png"

    X = df[["avg_order_value", "sales_frequency", "avg_quantity_per_order", "different_customer"]]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eps_initial = clusterer.find_optimal_eps(X_scaled, min_samples=3, save_path=elbow_path)

    optimal_eps, optimal_min_samples = optimize_eps_and_min_samples(X_scaled, eps_range=(0.5, 5.0), min_samples_range=(2, 10))

    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
    labels = dbscan.fit_predict(X_scaled)
    df["cluster"] = labels

    clusterer.plot_clusters(df, x_feature="avg_order_value", y_feature="sales_frequency", save_path=scatter_path)

    outliers = df[df["cluster"] == -1]

    return {
        "optimal_eps": optimal_eps,
        "optimal_min_samples": optimal_min_samples,
        "elbow_graph_path": elbow_path,
        "scatter_graph_path": scatter_path,
        "outliers": outliers.to_dict(orient="records"),
        "n_clusters": df["cluster"].nunique() - (1 if -1 in df["cluster"].unique() else 0)
    }


@app.get("/cluster/suppliers")
def cluster_suppliers():
    query = """
    SELECT 
        sup.supplier_id,
        COUNT(DISTINCT p.product_id) AS total_products,
        SUM(od.quantity) AS total_quantity_sold,
        AVG(od.unit_price) AS avg_unit_price,
        COUNT(DISTINCT o.customer_id) AS distinct_customers
    FROM 
        suppliers sup
    JOIN 
        products p ON sup.supplier_id = p.supplier_id
    JOIN 
        order_details od ON p.product_id = od.product_id
    JOIN 
        orders o ON od.order_id = o.order_id
    GROUP BY 
        sup.supplier_id
    ORDER BY 
        sup.supplier_id
    """
    df = clusterer.fetch_data(query)

    elbow_path = f"{GRAPH_DIR}/suppliers_elbow.png"
    scatter_path = f"{GRAPH_DIR}/suppliers_scatter.png"

    X = df[["total_products", "total_quantity_sold", "avg_unit_price", "distinct_customers"]]
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eps_initial = clusterer.find_optimal_eps(X_scaled, min_samples=3, save_path=elbow_path)

    optimal_eps, optimal_min_samples = optimize_eps_and_min_samples(X_scaled, eps_range=(0.5, 5.0), min_samples_range=(2, 10))

    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=optimal_eps, min_samples=optimal_min_samples)
    labels = dbscan.fit_predict(X_scaled)
    df["cluster"] = labels

    clusterer.plot_clusters(df, x_feature="total_products", y_feature="total_quantity_sold", save_path=scatter_path)

    outliers = df[df["cluster"] == -1]

    return {
        "optimal_eps": optimal_eps,
        "optimal_min_samples": optimal_min_samples,
        "elbow_graph_path": elbow_path,
        "scatter_graph_path": scatter_path,
        "outliers": outliers.to_dict(orient="records"),
        "n_clusters": df["cluster"].nunique() - (1 if -1 in df["cluster"].unique() else 0)
    }


@app.get("/countries/graph")
def get_graph(type: str = Query(..., enum=["elbow", "scatter"])):
    if type == "elbow":
        filename = "countries_elbow.png"
    elif type == "scatter":
        filename = "countries_scatter.png"
    else:
        raise HTTPException(status_code=400, detail="Invalid graph type")
    
    filepath = os.path.join(GRAPH_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/products/graph")
def get_graph(type: str = Query(..., enum=["elbow", "scatter"])):
    if type == "elbow":
        filename = "products_elbow.png"
    elif type == "scatter":
        filename = "products_scatter.png"
    else:
        raise HTTPException(status_code=400, detail="Invalid graph type")
    
    filepath = os.path.join(GRAPH_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="File not found")

@app.get("/suppliers/graph")
def get_graph(type: str = Query(..., enum=["elbow", "scatter"])):
    if type == "elbow":
        filename = "suppliers_elbow.png"
    elif type == "scatter":
        filename = "suppliers_scatter.png"
    else:
        raise HTTPException(status_code=400, detail="Invalid graph type")
    
    filepath = os.path.join(GRAPH_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="File not found")