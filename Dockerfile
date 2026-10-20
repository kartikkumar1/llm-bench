# Gateway + built dashboard in one image.
FROM node:20.18.1-alpine AS dashboard
WORKDIR /app
COPY dashboard/package.json ./
RUN npm install --no-audit --no-fund
COPY dashboard/ ./
RUN npm run build

FROM python:3.12.8-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY config.yaml ./
COPY bench/ bench/
COPY gateway/ gateway/
COPY --from=dashboard /app/dist dashboard/dist
EXPOSE 8000
CMD ["uvicorn", "gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
