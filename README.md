# Missing Word Substitution
Модель обучается подставлять пропущенные слова в предложения и на основе весов первого слоя определять синонимы слова terrible
Датасет можно взять с сайта [kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Используемые технологии
* Python - v3.10.6
* pip - v22.0.2

## Сборка и запуск
### Установка зависимостей
~~~
pip install -r requirements.txt
~~~

## Конфигурация
```bash
cp .env.example .env
```
_Прописать в `.env` параметры подключений._

```bash
source .env
```

### Запуск
~~~bash
python ./missing_word_substitution.py
~~~