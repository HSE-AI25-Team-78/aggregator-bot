# FastAPI-сервис для получения категорий новостных постов

##### Запуск сервера

Команды выполняются в корневой директории репозитория

```bash
docker compose build
docker compose up
```

После запуска сервер доступен для запросов по адресу [http://localhost:8080](http://localhost:8080). Документация: [http://localhost:8080/docs](http://localhost:8080/docs).

##### Запуск сервера в контейнере

```bash
cd ./service
uvicorn app:app --host 0.0.0.0
```
