import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zss import simple_distance, Node


df = pd.read_excel('chai.xlsx', engine='openpyxl')
df['Время заваривания (секунд)'] = df['Время заваривания (секунд)'].astype(int)

tastes = np.array(df['Вкус'])

# функция, которая создает дерево вкусов
def create_node(extra_node, object):
    extra = object.split('; ')
    new_list = []
    for element in extra:
        if ',' in element:
            new_list.append(element.split(', '))
        else:
            new_list.append(element)
    for item in new_list:
        if type(item) == list:
            next_node = Node(item[0])
            for i in range(1, len(item)):
                next_node.addkid(Node(item[i]))
            extra_node.addkid(next_node)
        else:
            extra_node.addkid(Node(item))


# Древесная мера близости по основному вкусовому оттенку
objects = tastes
matrix_tree_main = np.zeros((objects.shape[0], objects.shape[0]))
for i in range(objects.shape[0]):
    for j in range(objects.shape[0]):
        main_taste_node_1 = Node("Основной")
        create_node(main_taste_node_1, objects[i])
        main_taste_node_2 = Node("Основной")
        create_node(main_taste_node_2, objects[j])
        matrix_tree_main[i, j] = simple_distance(main_taste_node_1, main_taste_node_2)


plt.imshow(matrix_tree_main, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()


kat_dict = {"Пуэр": 0, "Улун": 1, "Габа": 2, "ЗЕЛЕНЫЙ ЧАЙ": 3, "КРАСНЫЙ ЧАЙ": 4, "ЖЕЛТЫЙ ЧАЙ": 5, "БЕЛЫЙ ЧАЙ": 6,
            "Травяной": 7}
kat_matr = [
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 0.8, 0.7, 0.6, 0.5, 1.0],
    [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.8, 1.0, 0.0, 0.7, 0.6, 0.3, 0.2],
    [1.0, 0.7, 1.0, 0.7, 0.0, 0.9, 0.9, 0.9],
    [1.0, 0.6, 1.0, 0.6, 0.9, 0.0, 0.5, 0.7],
    [1.0, 0.5, 1.0, 0.3, 0.9, 0.5, 0.0, 0.3],
    [1.0, 1.0, 1.0, 0.2, 0.9, 0.7, 0.3, 0.0],
]

objects = np.array(df['Категория'])
matrix_elem_kat = np.zeros((objects.shape[0], objects.shape[0]))
for i in range(objects.shape[0]):
    for j in range(objects.shape[0]):
        matrix_elem_kat[i, j] = kat_matr[kat_dict[objects[i]]][kat_dict[objects[j]]]


# Построение матрицы с цветовыми значениями
plt.imshow(matrix_elem_kat, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

plt.show()


# Функция для расчета евклидовой меры близости между объектами по времени заваривания
def similarity_measure(obj1, obj2):
    return np.linalg.norm(obj1 - obj2)

# получаем массив объектов по времени заваривания
objects = np.array(df['Время заваривания (секунд)'])
matrix_evklid_time = np.zeros((objects.shape[0], objects.shape[0]))


# Расчет значений меры для каждой пары объектов
for i in range(objects.shape[0]):
    for j in range(objects.shape[0]):
        matrix_evklid_time[i, j] = similarity_measure(objects[i], objects[j])

# Построение матрицы с цветовыми значениями
plt.imshow(matrix_evklid_time, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

# получаем массив объектов по температуре заваривания
objects = np.array(df['Температура заваривания C'])
matrix_evklid_temp = np.zeros((objects.shape[0], objects.shape[0]))

# Расчет значений меры для каждой пары объектов
for i in range(objects.shape[0]):
    for j in range(objects.shape[0]):
        matrix_evklid_temp[i, j] = similarity_measure(objects[i], objects[j])


plt.imshow(matrix_evklid_temp, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

# Функция для расчета меры близости по коэффициенту Жаккара
def jaccard_similarity_measure(obj1, obj2):
    intersection = np.sum(np.logical_and(obj1, obj2))
    union = np.sum(np.logical_or(obj1, obj2))
    if intersection == 0 and union == 0:
        return 1
    else:
        return (intersection / union)


# получаем массив объектов по признаку веганский или нет (бинарная ассоциативная)
objects = np.array(df['Есть ароматизатор'].replace(to_replace=['False', 'True'], value=['0', '1'])).astype(int)
matrix_binary = np.zeros((objects.shape[0], objects.shape[0]))
for i in range(objects.shape[0]):
    for j in range(objects.shape[0]):
        matrix_binary[i, j] = jaccard_similarity_measure(objects[i], objects[j])

plt.imshow(matrix_binary, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

# создание общей меры близости
recomender_matrix = matrix_tree_main + matrix_elem_kat + matrix_evklid_temp * 0.01 + matrix_evklid_time * 0.01 + matrix_binary * 2

# Построение матрицы с цветовыми значениями
plt.imshow(recomender_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

def get_recommendations_by_likes(items, data=df):
    # Разбиваем входную строку на отдельные наименования
    titles = items.split(', ')
    num_objects = len(titles)

    # Индекс чая в датасете
    indices = pd.Series(df.index, index=df['Название']).drop_duplicates()

    # переменная для хранения общей близости
    total_similarity = 0.0

    for title in titles:
        # Получение индекса чая
        idx = indices[title]
        # Расчет близости для одного объекта и добавление в общую
        total_similarity = total_similarity + recomender_matrix[idx]

    # Расчет среднего значения близости
    average_similarity = total_similarity / num_objects

    # Список чая по близости
    sim_scores = list(enumerate(average_similarity))

    # Сортировка чая по близости
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=False)

    # Получение индексов наиболее похожего чая
    chocolate_indices = [i[0] for i in sim_scores]
    for title in titles:
        idx = indices[title]
        chocolate_indices.remove(idx)

    # Фильтруем индексы
    indices_from_filtered_df = (pd.Series(data.index, index=data['Название']).drop_duplicates()).tolist()
    indices_after_filter = []
    for index in chocolate_indices:
        if index in indices_from_filtered_df:
            indices_after_filter.append(index)

    # Вывод рекомендаций
    return data.loc[indices_after_filter]

# рекомендация по лайкам
#item = input("Введите название понравившегося чая:")
recommendations = get_recommendations_by_likes('Шайцинь Юцзи Шэн')
recommendations

def do_fliter(recomendation, name=None, category=None, form=None, is_arom=None, temp=[None, None], time=[None, None],
              price=None, taste=None):
    # from expert_options.csv
    term_to_price = {
        'Очень дешевый': lambda x: int(x) <= 1000,
        'Дешевый': lambda x: 1000 < int(x) <= 1750,
        'Средний': lambda x: 1750 < int(x) <= 3000,
        'Дорогой': lambda x: 3000 < int(x) <= 4750,
        'Очень дорогой': lambda x: int(x) > 4750,
    }

    if name is not None:
        recomendation = recomendation[recomendation['Название'].apply(lambda x: x.find(name) != -1)]

    if category is not None:
        recomendation = recomendation[recomendation['Категория'].apply(lambda x: x == category)]

    if form is not None:
        recomendation = recomendation[recomendation['Форма упаковки'].apply(lambda x: x == form)]

    if is_arom is not None:
        recomendation = recomendation[recomendation['Есть ароматизатор'].apply(lambda x: x == is_arom)]

    if temp[0] is not None and temp[1] is not None:
        recomendation = recomendation[
            recomendation['Температура заваривания C'].apply(lambda x: temp[0] < int(x) <= temp[1])]

    if time[0] is not None and time[1] is not None:
        recomendation = recomendation[
            recomendation['Время заваривания (секунд)'].apply(lambda x: time[0] < int(x) <= time[1])]

    if taste is not None:
        recomendation = recomendation[recomendation['Вкус'].apply(lambda x: x.find(name) != -1)]

    if price is not None:
        recomendation = recomendation[recomendation['стоимость'].apply(term_to_price[price])]

    return recomendation

result = do_fliter(recommendations,name='а', is_arom=False, temp=[0, 100])
result