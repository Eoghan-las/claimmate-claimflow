FROM python:3.11-slim
WORKDIR /app
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt
COPY rag_assist ./rag_assist
COPY demo_app.py .
COPY LICENSE ./LICENSE
ENV CLAIMMATE_STATE_DIR=/app/runtime/demo
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "demo_app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
