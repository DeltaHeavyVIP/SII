import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from zss import simple_distance, Node

df = pd.read_excel('chai.xlsx', engine='openpyxl')
df['Время заваривания (секунд)'] = df['Время заваривания (секунд)'].astype(int)

tastes = np.array(df['Вкус'])
tastes

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


class TeaCat:
    def __init__(self, name, par) -> None:
        self.name = name
        self.par = par
        self.children = []
        if par is not None:
            par.children.append(self)

catRoot = TeaCat('root', None)
catSublim = TeaCat('Растворимый', catRoot)
catCustard = TeaCat('Заварной', catRoot)
catHerbal = TeaCat('Травяной', catCustard)
catTea = TeaCat('Чайный', catCustard)
catGreen = TeaCat('ЗЕЛЕНЫЙ ЧАЙ', catTea)
catRed = TeaCat('КРАСНЫЙ ЧАЙ', catTea)
catWhite = TeaCat('БЕЛЫЙ ЧАЙ', catTea)
catYellow = TeaCat('ЖЕЛТЫЙ ЧАЙ', catTea)
catOolong = TeaCat('Улун', catTea)
catPuer = TeaCat('Пуэр', catTea)
catGaba = TeaCat('Габа', catTea)

cats = {
    'root': catRoot,
    'Растворимый': catSublim,
    'Заварной': catCustard,
    'Травяной': catHerbal,
    'Чайный': catTea,
    'ЗЕЛЕНЫЙ ЧАЙ': catGreen,
    'БЕЛЫЙ ЧАЙ': catWhite,
    'ЖЕЛТЫЙ ЧАЙ': catYellow,
    'КРАСНЫЙ ЧАЙ': catRed,
    'Улун': catOolong,
    'Пуэр': catPuer,
    'Габа': catGaba,
}


def metricTree(a, b):
    c1, c2 = cats[a], cats[b]

    if c1.name == c2.name:
        return 0

    anc1 = {}
    cur = c1
    cntr = 0
    while cur is not None:
        anc1[cur.name] = cntr
        cntr += 1
        cur = cur.par

    cur = c2
    cntr = 0
    while cur.name not in anc1.keys():
        cntr += 1
        cur = cur.par
        if cur is None:
            raise

    return (cntr + anc1[cur.name]) / 4.0

objects = np.array(df['Категория'])
matrix_elem_kat = np.zeros((objects.shape[0], objects.shape[0]))
for i in range(objects.shape[0]):
    for j in range(objects.shape[0]):
        matrix_elem_kat[i, j] = metricTree(objects[i],objects[j])


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

# получаем массив объектов по температуре заваривания
objects = np.array(df['Температура заваривания C'])
matrix_evklid_temp = np.zeros((objects.shape[0], objects.shape[0]))


# Расчет значений меры для каждой пары объектов
for i in range(objects.shape[0]):
    for j in range(objects.shape[0]):
        matrix_evklid_temp[i, j] = similarity_measure(objects[i], objects[j])

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


# создание общей меры близости
recomender_matrix = matrix_tree_main + matrix_elem_kat + matrix_evklid_temp * 0.01 + matrix_evklid_time * 0.01 + matrix_binary * 2

# Построение матрицы с цветовыми значениями
plt.imshow(recomender_matrix, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()

def get_recomendation(likes, dislikes, data):
    def find(name):
        return data.loc[data['Название'] == name]

    recomendation = [0.5 for i in range(len(data))]
    for i in likes:
        dish_i = find(i).index[0]
        recomendation[dish_i] = None
        for j in range(len(recomender_matrix[dish_i])):
            if recomendation[j] is None:
                continue

            recomendation[j] -= recomender_matrix[dish_i][j] / len(likes) * 0.5

    for i in dislikes:
        dish_i = find(i).index[0]
        recomendation[dish_i] = None
        for j in range(len(recomender_matrix[dish_i])):
            if recomendation[j] is None:
                continue

            recomendation[j] += recomender_matrix[dish_i][j] / len(dislikes) * 0.5

    result = data
    result['Рекомендация'] = recomendation
    result = result.sort_values(by=['Рекомендация'], ascending=False)

    return result


result = get_recomendation(['Шайцинь Юцзи Шэн', 'Путь Сердца'], ['Юцзи Гаоцзи Шу Ч'], df)
result

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

result = do_fliter(result, name='а', is_arom=True, temp=[0, 100])
result