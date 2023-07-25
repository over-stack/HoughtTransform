import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict

# создаем аккумулятор
def accumulate_hough_line(image, bottom_threshold=50, top_threshold=150):
    # находим грани и выделяем их
    edges = cv2.Canny(image, bottom_threshold, top_threshold)
    nonzero_y, nonzero_x = np.nonzero(edges)

    thetas = np.deg2rad(np.arange(0, 180, 1))
    # длина диагонали - максимальное возможное значение для rho, минимальное соответственно - минус длина диагонали
    diagonal_length = int(np.ceil(np.sqrt(image.shape[1] ** 2 + image.shape[0] ** 2)))
    # заполняем нулями
    accumulator = np.zeros((2 * diagonal_length, len(thetas)))

    # преобразование Хафа
    x_cos_theta = np.dot(nonzero_x.reshape((-1, 1)), np.cos(thetas).reshape((1, -1)))
    y_sin_theta = np.dot(nonzero_y.reshape((-1, 1)), np.sin(thetas).reshape((1, -1)))
    calculated_rhos = np.round(x_cos_theta + y_sin_theta).astype(np.int16)

    for theta in range(len(thetas)):
        unique_rhos, unique_rhos_counts = np.unique(calculated_rhos[:, theta], return_counts=True)
        # добавляем длину диагонали, чтобы избежать отрицательных значений индекса
        accumulator[unique_rhos + diagonal_length, theta] = unique_rhos_counts

    return accumulator


# в случае количества точек пересечения линий != 4 бросит исключение
def get_rect_lines(accumulator, threshold=100):
    peaks = list()

    # окно для поиска локальных максимумов
    rho_window, theta_window = 4, 4
    for rho in range(accumulator.shape[0]):
        for theta in range(accumulator.shape[1]):
            if accumulator[rho, theta] >= threshold:
                peak = accumulator[rho, theta]

                # проверка точки rho, theta на локальный максимум
                for drho in range(-rho_window, rho_window + 1):
                    for dtheta in range(-theta_window, theta_window + 1):
                        if 0 <= drho + rho < accumulator.shape[0] and 0 <= dtheta + theta < accumulator.shape[1]:
                            if accumulator[rho + drho, theta + dtheta] > peak:
                                peak = accumulator[rho + drho, theta + dtheta]
                                break

                if peak > accumulator[rho, theta]:
                    continue

                peaks.append([rho - accumulator.shape[0] // 2, theta])

    if len(peaks) != 4:
        raise Exception('Количество локальных максимумо не равно 4! Это не прямоугольник!')

    lines = defaultdict(list)
    # ищем точки пересечений
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            # если углы равны (прямые параллельны), то пересечение не ищем
            if peaks[i][1] == peaks[j][1]:
                continue

            rho1 = peaks[i][0]
            theta1 = peaks[i][1]
            rad_theta1 = np.deg2rad(peaks[i][1])

            rho2 = peaks[j][0]
            theta2 = peaks[j][1]
            rad_theta2 = np.deg2rad(peaks[j][1])

            A = np.array([[np.cos(rad_theta1), np.sin(rad_theta1)],
                          [np.cos(rad_theta2), np.sin(rad_theta2)]])
            b = np.array([rho1, rho2]).T
            x, y = np.linalg.solve(A, b)
            x, y = int(np.round(x)), int(np.round(y))

            lines[(rho1, theta1)].append([x, y])
            lines[(rho2, theta2)].append([x, y])

    return lines


def visualize(image, accumulator, lines, output=None):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].imshow(image)
    for params, line in lines.items():
        xs = [line[0][0], line[1][0]]
        ys = [line[0][1], line[1][1]]
        ax[0].plot(xs, ys, linewidth=3)
    ax[0].set_title('Image (with lines)')
    ax[0].set_xlabel('X')
    ax[0].set_ylabel('Y')

    ax[1].imshow(accumulator, cmap='jet', extent=[0, 180, -accumulator.shape[0] // 2, accumulator.shape[0] // 2])
    ax[1].set_aspect(0.1)
    ax[1].set_title('Accumulator')
    ax[1].set_xlabel('Theta')
    ax[1].set_ylabel('Rho')

    if output:
        plt.savefig(output)
    #plt.show()


def score(data_folder, output_folder):
    images = os.listdir(data_folder + '/images')

    true_points = defaultdict(list)
    # парсинг данных из файла
    with open(data_folder + '/points.txt') as f:
        for line in f.readlines():
            splitted_line = line.split(';')[:-1]
            filename = splitted_line[0]
            for str_point in splitted_line[1:]:
                coordinates = str_point.split(',')
                true_points[filename].append([int(coordinates[0]), int(coordinates[1])])

    predicted_points = dict()
    for image_filename in images:
        image = cv2.imread(data_folder + '/images/' + image_filename)

        accumulator = accumulate_hough_line(image)
        lines = get_rect_lines(accumulator)
        visualize(image, accumulator, lines, output_folder + '/' + image_filename.split('.')[0] + '_vis.jpg')

        # словарь нужен для того, чтобы потом из ключей получить 4 уникальные точки, счетчики сами по себе смысла не несут
        dict_points = defaultdict(int)
        for line in lines.values():
            dict_points[tuple(line[0])] += 1
            dict_points[tuple(line[1])] += 1
        points = list(dict_points.keys())

        visualize(image, accumulator, lines)

        # сортируем точки в нужном порядке
        sorted_points = list()
        sorted_points.append(min(points, key=lambda x: x[1]))
        sorted_points.append(max(points, key=lambda x: x[0]))
        sorted_points.append(max(points, key=lambda x: x[1]))
        sorted_points.append(min(points, key=lambda x: x[0]))

        predicted_points[image_filename] = sorted_points

    scores = dict()
    for key in images:
        tp = true_points[key]
        pp = predicted_points[key]

        l2 = 0
        for i in range(4):
            l2 += np.sqrt((tp[i][0] - pp[i][0]) ** 2 + (tp[i][1] - pp[i][1]) ** 2)
        scores[key] = l2 / 4  # усредняем метрику

        print(key)
        print('Правильные точки:', tp)
        print('Предсказанные точки:', pp)
        print()

    print('mean(sum(L2)) метрика (среднее расстояние от правильной точки до предсказанной):')
    for key, value in scores.items():
        print(f'{key}: {value}')

    return scores


if __name__ == '__main__':
    score('data', 'data/output')