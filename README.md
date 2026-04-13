# Real-Time Fraud Detection System

A production-grade FinTech fraud detection system using Big Data, ML, and streaming architecture.

## Project Structure

```
/src
  /api          - API endpoints and services
  /ml           - Machine learning models and pipelines
  /streaming    - Streaming data processing components
  /features     - Feature engineering modules
  /utils        - Utility functions and helpers
  /data_processing - Data ingestion and preprocessing
/notebooks     - Jupyter notebooks for analysis and exploration
/data          - Datasets and data files
/dashboards    - Dashboard applications
/tests         - Unit and integration tests
/docker        - Docker configurations
```

## Setup

1. Create virtual environment:
```bash
# On Linux/Mac
python -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify setup by running the verification script:
```bash
python verify_environment.py
```

## Architecture

The system follows a streaming architecture:

Kafka → PySpark Streaming → Feature Engineering → Redis → ML Model → FastAPI → Response

## Day 1 Deliverables

- Project structure setup
- Environment configuration
- Dataset exploration (PaySim)
- Architecture design
- Project planning and KPIs

## License

MIT License
