# Sales Data Clustering

This project clusters sales data from the Northwind database using the **DBSCAN** algorithm.  
It automatically finds the best `eps` and `min_samples` values, detects clusters and outliers, and provides API access via **FastAPI**.

## Features

- 📊 Automatic clustering (countries, products, suppliers)
- ⚡ Auto-optimization of DBSCAN parameters
- 🖼️ Elbow and scatter plot generation
- 🚀 FastAPI server
- 🛢️ PostgreSQL integration
- 🔒 Secure config with `.env`

## Endpoints

- `GET /cluster/countries`
- `GET /cluster/products`
- `GET /cluster/suppliers`
- `GET /countries/graph?type=elbow/scatter`
- `GET /products/graph?type=elbow/scatter`
- `GET /suppliers/graph?type=elbow/scatter`

## Setup

1. Clone the repo:
    ```bash
    git clone https://github.com/ilaydaakyuz/sales-data-clustering.git
    cd sales-data-clustering
    ```

2. Create `.env`:
    ```dotenv
    POSTGRES_USER=your_user
    POSTGRES_PASSWORD=your_password
    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432
    POSTGRES_DB=GYK2Northwind
    ```

3. Install packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run server:
    ```bash
    uvicorn main:app --reload
    ```

Access API docs:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## License

Licensed under the MIT License.
