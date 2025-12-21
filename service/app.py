from fastapi import FastAPI, status, HTTPException
import process as p
import db
from api_models import *
from datetime import datetime as dt


app = FastAPI(
    title='Aggregator-bot',
    description='Aggregate posts of telegram channels',
    version='1.0.0'
)

dao = db.DataAccessObject('v0.1')


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
