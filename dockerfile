FROM python:3.8-slim
WORKDIR /stage-app
ADD / $HOME/
RUN pip install -r /app/requirements.txt 
EXPOSE 5000
ENV FLASK_APP $HOME/app/main.py
ENV FLASK_ENV development
CMD ["flask","run","--host","0.0.0.0"]