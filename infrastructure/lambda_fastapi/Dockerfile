FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt --target /var/task

COPY app ./app

# Handler de Lambda (módulo.ruta_función)
CMD ["app.main.handler"]
