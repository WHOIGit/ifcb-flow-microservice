# IFCB Flow Metric Microservice

A stateless microservice that computes flow metric anomaly scores for IFCB (Imaging FlowCytobot) bins using a pre-trained Isolation Forest model.

## Overview

This service extracts 26 statistical features from IFCB point cloud data (ADC files) and computes anomaly scores using an Isolation Forest classifier. Higher scores indicate more unusual flow distributions that may indicate instrument issues or data quality problems.

## API

### GET /flow_metric/{bin_id}

Computes the flow metric anomaly score for the specified IFCB bin.

**Parameters:**
- `bin_id` (path): IFCB bin identifier (e.g., `D20120101_T120000`)

**Response:**
- Content-Type: `text/plain`
- Body: Single float value representing the anomaly score

**Example:**
```bash
curl http://localhost:8001/flow_metric/D20120101_T120000
```

**Response:**
```
-0.0523
```

## Setup

### Prerequisites

1. IFCB data directory with bin files (must contain ADC files)
2. Pre-trained classifier model file (`classifier.pkl`)

### Environment Variables

Required environment variables (set these on your host):

- `IFCB_DATA_DIR`: Path to IFCB data directory on host (e.g., `/path/to/ifcb/data`)
- `MODEL_DIR`: Path to directory containing classifier.pkl on host (e.g., `/path/to/models`)

You can copy `.env.template` to `.env` and modify the paths:

```bash
cp .env.template .env
# Edit .env with your actual paths
```

### Running with Docker Compose

```bash
docker compose up --build
```

The service will be available at `http://localhost:8001`

```bash
# Replace with an actual bin ID from your data
curl http://localhost:8001/flow_metric/D20120101_T120000
```

