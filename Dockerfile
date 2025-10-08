FROM python:3.11-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# COPY ./src /app/src
# Copy everything including data
COPY . .

COPY ./src/data/kokokah_lms_faqs.csv /app/src/data/kokokah_lms_faqs.csv

# Double-check CSV presence
RUN ls -R /app/src/data || echo "No data folder found"

EXPOSE 8000

CMD ["uvicorn", "src.chatbot:app", "--host", "0.0.0.0", "--port", "8000"]