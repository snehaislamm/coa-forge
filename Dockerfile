FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY coaforge ./coaforge
COPY scenarios ./scenarios
ENTRYPOINT ["python", "-m", "coaforge"]
CMD ["scenarios/northern_bastion.yaml", "--out", "/app/results"]
