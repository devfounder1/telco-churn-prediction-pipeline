# FROM указывает, с чего начинаем. python:3.11-slim — это легкий 
# (slim) образ Linux с уже установленным Python 3.11. 
# Мы не берем полный образ, чтобы контейнер весил меньше.
FROM python:3.11-slim

# WORKDIR создает папку /app внутри контейнера и делает её текущей.
WORKDIR /app

# COPY копирует файл requirements_api.txt с твоего Mac 
COPY requirements_api.txt .

# RUN выполняет команду внутри контейнера.
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements_api.txt --extra-index-url https://download.pytorch.org/whl/cpu

# COPY . . копирует ВСЕ файлы из текущей папки на твоем Mac в папку /app внутри контейнера.
COPY . .

# Открытие порта
# EXPOSE документирует, что приложение внутри контейнера 
# будет слушать сетевой порт 8000 (стандарт для Uvicorn/FastAPI).
EXPOSE 8000

# Команда запуска
# CMD указывает команду, которая выполнится автоматически 
# при старте контейнера. Мы запускаем uvicorn, указывая 
# имя файла (api), имя приложения (app) и хост 0.0.0.0 
# (чтобы API был доступен не только внутри контейнера, но и снаружи).
CMD [ "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

