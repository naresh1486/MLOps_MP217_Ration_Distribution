# PART 13 REPORT — Kubernetes Deployment

**Project:** MP217 — Ration Shop Distribution (Kochi)  
**Group:** G230  
**Course:** SCSE3040 MLOps  
**Date:** November 24, 2026  
**Status:** PASS  

---

## 1. Executive Summary

In Part 13, the containerized FastAPI prediction application was deployed and validated on a live local Kubernetes cluster (`kind-mp217-cluster`). The deployment and service manifests were constructed to adhere strictly to `PROBLEM.md` Section 6 and `STUDENT_HANDBOOK.md` Sections 5, 9, 11 (Criteria I):

- **Deployment Name:** `mp217-api`
- **Replicas:** Exactly 4 replicas, all verified `4/4 Ready` and `Available`.
- **Image:** `mp217-api:1.0` (loaded from local Docker daemon into the cluster).
- **Probes:** Both readiness and liveness probes configured on HTTP `/health` on port 8000; liveness configured with a longer initial delay (`15s`) than readiness (`5s`).
- **Service Name:** `mp217-svc`
- **Service Routing:** Type `ClusterIP`, listening on port `80` and routing to container target port `8000`.

The cluster was exercised via service port forwarding. Live HTTP probes confirmed that `GET /health` (`200 OK`) and `POST /predict` (`200 OK`, returning `1177.92 kg`) operate with zero downtime across the replica set, while client input validation rules (`HTTP 422`) remain strictly enforced. The full 62-test automated test suite passed with 100% compliance.

---

## 2. Kubernetes Environment

- **Cluster Tool:** `kind` (Kubernetes in Docker, version 0.33.0).
- **Cluster Name:** `mp217-cluster` (Context: `kind-mp217-cluster`).
- **Kubernetes Node Image:** `kindest/node:v1.37.0`.
- **Node Status:** `mp217-cluster-control-plane` (Ready, Control-Plane).
- **Image Staging:** Docker image `mp217-api:1.0` loaded directly into the node storage using `kind load docker-image mp217-api:1.0 --name mp217-cluster`.

---

## 3. Kubernetes Configuration Files

### A. Deployment Configuration (`k8s/deployment.yaml`)

[`k8s/deployment.yaml`](file:///Users/nareshyadav/Desktop/mlops_project/k8s/deployment.yaml) defines the stateless replica set and health monitoring:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mp217-api
  labels:
    app: mp217-api
spec:
  replicas: 4
  selector:
    matchLabels:
      app: mp217-api
  template:
    metadata:
      labels:
        app: mp217-api
    spec:
      containers:
        - name: mp217-api
          image: mp217-api:1.0
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
              name: http
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 2
            failureThreshold: 3
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 15
            timeoutSeconds: 2
            failureThreshold: 3
```

#### Key Properties:
1. **Replicas (4):** Exactly 4 pods running concurrently to meet the client's high-availability specification.
2. **Matching Selector:** `spec.selector.matchLabels.app: mp217-api` perfectly matches `template.metadata.labels.app: mp217-api`.
3. **Probe Timings:**
   - Readiness: `initialDelaySeconds: 5`, `periodSeconds: 10`
   - Liveness: `initialDelaySeconds: 15`, `periodSeconds: 15` (strictly satisfies the requirement that liveness probe waits longer before beginning checks).

---

### B. Service Configuration (`k8s/service.yaml`)

[`k8s/service.yaml`](file:///Users/nareshyadav/Desktop/mlops_project/k8s/service.yaml) defines the internal load balancer:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mp217-svc
  labels:
    app: mp217-api
spec:
  type: ClusterIP
  selector:
    app: mp217-api
  ports:
    - name: http
      port: 80
      targetPort: 8000
      protocol: TCP
```

#### Key Properties:
1. **Service Name:** `mp217-svc`.
2. **Type:** `ClusterIP`.
3. **Port Forwarding:** Service port `80` routes to container target port `8000`.
4. **Pod Routing:** `selector.app: mp217-api` routes traffic evenly across all 4 healthy pods.

---

## 4. Deployment Verification & Inspection

### A. Deployment Status
```text
$ kubectl get deployment mp217-api
NAME        READY   UP-TO-DATE   AVAILABLE   AGE
mp217-api   4/4     4            4           6m
```

### B. Pod Replicas Status
```text
$ kubectl get pods -l app=mp217-api -o wide
NAME                        READY   STATUS    RESTARTS   AGE    IP           NODE                          NOMINATED NODE   READINESS GATES
mp217-api-88f5cb54d-cb8gb   1/1     Running   0          6m     10.244.0.6   mp217-cluster-control-plane   <none>           <none>
mp217-api-88f5cb54d-fv9lq   1/1     Running   0          6m     10.244.0.7   mp217-cluster-control-plane   <none>           <none>
mp217-api-88f5cb54d-s6tvc   1/1     Running   0          6m     10.244.0.8   mp217-cluster-control-plane   <none>           <none>
mp217-api-88f5cb54d-xgxb4   1/1     Running   0          6m     10.244.0.5   mp217-cluster-control-plane   <none>           <none>
```

### C. Service Status
```text
$ kubectl get service mp217-svc
NAME        TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
mp217-svc   ClusterIP   10.96.21.211   <none>        80/TCP    6m
```

---

## 5. Live Endpoint Verification via Service

Port forwarding was established to route traffic through the Kubernetes Service `mp217-svc` on port 80:

```bash
kubectl port-forward svc/mp217-svc 8080:80
```

### A. Health Probe Endpoint (`GET /health` via Service Port 80)
```bash
curl -i http://127.0.0.1:8080/health
```

#### Output:
```http
HTTP/1.1 200 OK
date: Mon, 23 Nov 2026 23:00:15 GMT
server: uvicorn
content-length: 74
content-type: application/json

{"status":"ok","model_name":"mp217_ration_model","model_alias":"champion"}
```

---

### B. Prediction Endpoint (`POST /predict` via Service Port 80)
```bash
curl -i -X POST http://127.0.0.1:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "cards_attached": 1150,
    "working_hours": 7.0,
    "biometric_failures": 35,
    "temp_c": 29.5,
    "stock_kg_morning": 1600.0,
    "days_to_month_end": 12,
    "shop_area": "tribal",
    "block": "Block North"
  }'
```

#### Output:
```http
HTTP/1.1 200 OK
date: Mon, 23 Nov 2026 23:00:15 GMT
server: uvicorn
content-length: 84
content-type: application/json

{"prediction":1177.92,"predicted_grain_kg":1177.92,"grain_kg":1177.92,"status":"ok"}
```

---

### C. Validation Enforcement via Service (`HTTP 422`)

1. **Rule 1 (`biometric_failures: 180` > 150):**
   ```http
   HTTP/1.1 422 Unprocessable Entity
   {"detail":[{"type":"less_than_equal","loc":["body","biometric_failures"],"msg":"Input should be less than or equal to 150","input":180,"ctx":{"le":150}}]}
   ```
2. **Rule 2 (`working_hours: 1.0` < 1.5):**
   ```http
   HTTP/1.1 422 Unprocessable Entity
   {"detail":[{"type":"greater_than_equal","loc":["body","working_hours"],"msg":"Input should be greater than or equal to 1.5","input":1.0,"ctx":{"ge":1.5}}]}
   ```

---

## 6. Pod Logs Verification

Inspecting logs from the replica set confirms continuous health checks by the kubelet and successful handling of prediction events:

```text
[pod/mp217-api-88f5cb54d-s6tvc/mp217-api] INFO: 10.244.0.1:50680 - "GET /health HTTP/1.1" 200 OK
[pod/mp217-api-88f5cb54d-xgxb4/mp217-api] INFO: 10.244.0.1:38648 - "GET /health HTTP/1.1" 200 OK
[pod/mp217-api-88f5cb54d-cb8gb/mp217-api] INFO: 127.0.0.1:60674 - "GET /health HTTP/1.1" 200 OK
[pod/mp217-api-88f5cb54d-cb8gb/mp217-api] {"timestamp": "2026-11-23 23:00:16,212", "level": "INFO", "logger": "mp217.api", "message": "Prediction generated successfully", "event": "prediction", "predicted_grain_kg": 1177.92, "cards_attached": 1150, "working_hours": 7.0, "biometric_failures": 35, "temp_c": 29.5, "stock_kg_morning": 1600.0, "days_to_month_end": 12, "shop_area": "tribal", "block": "Block North"}
[pod/mp217-api-88f5cb54d-cb8gb/mp217-api] INFO: 127.0.0.1:60676 - "POST /predict HTTP/1.1" 200 OK
[pod/mp217-api-88f5cb54d-cb8gb/mp217-api] INFO: 127.0.0.1:60682 - "POST /predict HTTP/1.1" 422 Unprocessable Entity
[pod/mp217-api-88f5cb54d-fv9lq/mp217-api] INFO: 10.244.0.1:54382 - "GET /health HTTP/1.1" 200 OK
```

---

## 7. Python Test Suite Execution

All 62 tests across all 8 modules continue to pass cleanly:

```text
============================= test session starts ==============================
platform darwin -- Python 3.11.9, pytest-8.3.4, pluggy-1.7.0
rootdir: /Users/nareshyadav/Desktop/mlops_project
configfile: pytest.ini
collected 62 items

tests/test_api.py ......................                                 [ 35%]
tests/test_cleaning.py ........                                          [ 48%]
tests/test_cleaning_validation.py .....                                  [ 56%]
tests/test_config_and_utils.py ....                                      [ 62%]
tests/test_evaluate.py .....                                             [ 70%]
tests/test_model_inference.py .....                                      [ 79%]
tests/test_quality_checks.py ........                                    [ 91%]
tests/test_train.py .....                                                [100%]

======================= 62 passed, 18 warnings in 6.31s ========================
```

---

## 8. Files Created or Modified

| File | Status | Description |
|---|---|---|
| [`k8s/deployment.yaml`](file:///Users/nareshyadav/Desktop/mlops_project/k8s/deployment.yaml) | Created | Kubernetes Deployment specification (4 replicas, image `mp217-api:1.0`, container port 8000, readiness probe delay 5s, liveness probe delay 15s) |
| [`k8s/service.yaml`](file:///Users/nareshyadav/Desktop/mlops_project/k8s/service.yaml) | Created | Kubernetes Service specification (`mp217-svc`, ClusterIP, port 80 forwarding to targetPort 8000) |
| [`reports/PART_13_REPORT.md`](file:///Users/nareshyadav/Desktop/mlops_project/reports/PART_13_REPORT.md) | Created | Part 13 technical report |
| [`G230_MP217/reports/PART_13_REPORT.md`](file:///Users/nareshyadav/Desktop/mlops_project/G230_MP217/reports/PART_13_REPORT.md) | Created | Synchronized report copy in group distribution directory |
