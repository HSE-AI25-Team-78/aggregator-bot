from fastapi import FastAPI, status, HTTPException, Depends, File, UploadFile, Form, Header
import process as p
import db
from api_models import *
from datetime import datetime as dt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import JWTError, jwt
import os
from dotenv import load_dotenv
import numpy as np


load_dotenv()

MY_ENV_VAR = os.getenv('MY_ENV_VAR')


# Настройки JWT
SECRET_KEY = os.getenv('SECRET_KEY')
ALGORITHM = os.getenv('ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv('ACCESS_TOKEN_EXPIRE_MINUTES'))

# Настройки для удаления истории
DELETE_TOKEN = os.getenv('DELETE_TOKEN')


app = FastAPI(
    title='Aggregator-bot',
    description='Aggregate posts of telegram channels',
    version='1.0.0'
)

dao = db.DataAccessObject('v0.1')
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Пользователи в памяти (в проде - в БД)
# fake_users_db = {
#     "admin": {
#         "username": "admin",
#         "full_name": "Admin User",
#         "email": "admin@example.com",
#         "hashed_password": pwd_context.hash("admin123"),  # Пароль: admin123
#         "role": "admin",
#         "disabled": False,
#     },
#     "user": {
#         "username": "user",
#         "full_name": "Regular User",
#         "email": "user@example.com",
#         "hashed_password": pwd_context.hash("user123"),  # Пароль: user123
#         "role": "user",
#         "disabled": False,
#     }
# }

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str):
    user = dao.get_user_by_username(username)
    if not user or not user.is_active:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = dao.get_user_by_username(username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def verify_delete_token(x_delete_token: Optional[str] = Header(None)):
    if x_delete_token != DELETE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid delete token"
        )
    return True


@app.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}



@app.get('/', tags=['root'])
async def root():
    """
    Docstring for root
    """
    return {
        'message': 'welcome!'
    }


@app.post('/forward', status_code=status.HTTP_200_OK)
async def forward(item: PredictRequest) -> PredictResponse:
    log_data = {
        'timestamp': dt.now(),
        'work_time': None,
        'text': item.text,
        'label': None,
        'success': True
    }

    try:
        label = p.run_pipeline(item.text)
        if label is None:
            raise Exception('не удалось определить класс')
        log_data['label'] = label
    except Exception as e:
        print(f'Exception: {e}')
        log_data['comment'] = 'модель не смогла обработать данные'
        log_data['success'] = False
    
    log_data['work_time'] = (dt.now() - log_data['timestamp']).total_seconds()
    dao.add_history(**log_data)
    if not log_data['success']:
        raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=log_data['comment']
            )

    return PredictResponse(
        text=log_data['text'],
        label=log_data['label']
    )


@app.get('/history')
async def get_history():
    items = [HistoryItem(**row) for row in dao.get_history()]
    return HistoryResponse(
        items=items,
        count=len(items)
    )

@app.delete('/history', dependencies=[Depends(get_current_admin)])
async def delete_history(confirm_token: str = Header(..., alias="X-Confirm-Token")):
    """
    Удаление истории вызовов.
    Требуется подтверждающий токен в заголовке X-Confirm-Token.
    Доступно только администраторам.
    """
    # Проверяем подтверждающий токен (в реальном приложении можно использовать более сложную логику)
    if confirm_token != DELETE_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Неверный подтверждающий токен"
        )
    
    try:
        dao.delete_all_history()
        return {"message": "История успешно удалена"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при удалении истории: {str(e)}"
        )
        
@app.get('/stats', dependencies=[Depends(get_current_admin)])
async def get_stats():
    """
    Получение статистики запросов.
    Доступно только администраторам.
    """
    try:
        history_data = dao.get_history()
        
        if not history_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Нет данных для статистики"
            )
        
        records = [dict(row) for row in history_data]
        
        successful_records = [r for r in records if r['success']]
        
        if not successful_records:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Нет успешных запросов для статистики"
            )
        
        work_times = [r['work_time'] for r in successful_records]
        work_times_array = np.array(work_times)
        
        text_lengths = []
        token_counts = []
        
        for r in successful_records:
            if r['text']:
                text_lengths.append(len(r['text']))
                tokens = len(str(r['text']).split()) #TODO Tokenizer
                token_counts.append(tokens)
        
        # Вычисляем статистики
        def calculate_stats(data):
            if not data:
                return None
            
            arr = np.array(data)
            return {
                "mean": float(np.mean(arr)),
                "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
                "count": len(data)
            }
        
        # Статистика времени обработки
        time_stats = calculate_stats(work_times)
        
        # Статистика характеристик входных данных
        input_stats = {
            "text_length": calculate_stats(text_lengths) if text_lengths else None,
            "token_count": calculate_stats(token_counts) if token_counts else None
        }
        
        # Общая статистика
        overall_stats = {
            "total_requests": len(records),
            "successful_requests": len(successful_records),
            "success_rate": len(successful_records) / len(records) * 100 if records else 0,
            "last_request": max(r['timestamp'] for r in records) if records else None,
            "first_request": min(r['timestamp'] for r in records) if records else None
        }
        
        return StatsResponse(
            request_stats={
                "processing_time": time_stats,
                "overall": overall_stats
            },
            input_stats=input_stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ошибка при расчете статистики: {str(e)}"
        )
