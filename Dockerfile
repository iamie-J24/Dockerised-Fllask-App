FROM python:3.9

WORKDIR /carev

ADD . /carev


RUN pip install -r requirements.txt

EXPOSE 5001

CMD ["python", "knn.py"]