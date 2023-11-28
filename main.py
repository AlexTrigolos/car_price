import pickle
import pandas as pd

from pydantic import BaseModel
from typing import List
from sklearn.linear_model import Ridge
from fastapi import FastAPI, UploadFile
from starlette.responses import FileResponse

app = FastAPI(title='Cars')

model = Ridge(alpha=7)  # Создание лучшей модели
with open('model_weights.pickle', 'rb') as file:  # Восстановление весов модели из калаба
    model.intercept_ = pickle.load(file)
    model.coef_ = pickle.load(file)

with open('encoder.pickle', 'rb') as file:
    encoder = pickle.load(file)  # Восстановление one hot кодировки

with open('train_medians.pickle', 'rb') as file:
    train_medians = dict(pickle.load(file))  # Получение медиан для всех колонок в случае наличия нулей в данных

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def mileage(df):
    mileage = []  # в зависимости от размерности либо просто добавляем ее, либо умножаем на 1.4
    for i in df['mileage']:
        if str(i).endswith('km/kg'):
            i = i[:-6]
            i = float(i) * 1.40
            mileage.append(float(i))
        elif str(i).endswith('kmpl'):
            i = i[:-6]
            mileage.append(float(i))
        else:
            mileage.append(i)
    return mileage


def missing_and_dimension_columns(df):
    # избавляемся от размерностей и при необходимости приводим к новому типу
    df['mileage'] = mileage(df)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(float)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').astype(float)
    missing_columns = list(df.columns[df.isna().sum().ne(0)])  # получаем массив колонок с пропусками
    for missing_column in missing_columns:
        df[missing_column] = df[missing_column].fillna(train_medians[missing_column])  # все пропуски заполняем медианами сохраненными ранее
    return df


def preproc_data_frame(df):
    df = missing_and_dimension_columns(df)  # Обработка пропусков и полей с размерностями

    features = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']  # Колонки которые надо изменить one hot кодировкой
    df = df.drop(['selling_price', 'name', 'torque'], axis=1)  # Поля, которые не должны быть в модели

    encoded_features = encoder.transform(df[features]).toarray()  # изменяем датафрейм на пред обученном one hot encoder'е
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(features))  # Переделываем one hot кодирование в датафрейм с новыми колонками
    return pd.concat([df.drop(features, axis=1), encoded_df], axis=1)  # Склеиваем новый датафрейм из старого без изменяемых столбцов с датафреймом из новых столбцов


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = preproc_data_frame(pd.DataFrame(dict(item), index=[0]))  # Преобразуем в словарь и в датафрейм, после чего передаем в функцию препроцесса
    return model.predict(df)  # Возвращаем предсказание


@app.post("/predict_items")
def predict_items(file: UploadFile) -> FileResponse:  # Изменил получаемый и возвращаемый тип, потому что в задании было разное описание, но более интересным показалось получать файл и возвращать новый файл, через сваггер (по пути /docs) делается просто
    df = pd.read_csv(file.filename)  # Получаем файл из файла в датафрейм
    file_name = 'df_test_with_predict.csv'  # Новое имя для файла с предсказанием

    preproced_df = preproc_data_frame(df)  # Подготовить данные под обученную модель
    predictions = model.predict(preproced_df)  # Сделать предсказание подготовленного датафрейма
    df['predictions'] = predictions  # Записать предсказания в начальный датафрейм

    df.to_csv(file_name, index=False)  # Записать датафрейм с предсказанием в новый файл
    return FileResponse(f'./{file_name}', media_type='application/octet-stream', filename=file_name)  # Отдать файл на скачивание
