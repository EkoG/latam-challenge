FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./ /app
RUN pip install fastapi uvicorn pandas xgboost scikit-learn
EXPOSE 8080
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]