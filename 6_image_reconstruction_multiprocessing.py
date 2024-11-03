import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
from PIL import Image, ImageOps
import time
from scipy.special import rel_entr
from skimage.metrics import structural_similarity

# KL-дивергенция для сравнения распределений
def is_similar(img_matrix, agent_matrix):
    if np.all(img_matrix == 0) or np.all(agent_matrix == 0):
        # являются ли обе матрицы нулевыми
        return np.all(img_matrix == 0) and np.all(agent_matrix == 0)
    else:
        # малое смещение
        c = 1e-12
        P = img_matrix / np.sum(img_matrix) + c
        Q = agent_matrix / np.sum(agent_matrix) + c
        return np.sum(rel_entr(P, Q))

# структурное сходство (SSIM)
def similarity(image1, image2):
    similarity_index, _ = structural_similarity(image1, image2, full=True)
    return similarity_index

# приведение матрицы к целым числам от 0 до 255
def scaling(agent_matrix):
    if np.all(agent_matrix == 0):
        return agent_matrix
    
    min_val = np.min(agent_matrix)
    max_val = np.max(agent_matrix)
    scaled_agent_matrix = np.round(255 * (agent_matrix - min_val) / (max_val - min_val))
    return scaled_agent_matrix.astype(np.uint8)

# получение соседних координат
def get_neighbor_coords(x, y, max_x, max_y):
    x = int(x)
    y = int(y)
    neighbor_coords = np.array([(x + dx, y + dy) 
                                for dx in (-1, 0, 1) 
                                for dy in (-1, 0, 1) 
                                if (dx != 0 or dy != 0) and x + dx >= 0 and y + dy >= 0 and x + dx < max_x and y + dy < max_y])
    return neighbor_coords

# получение следующих оптимальных координат для движения агента (тех при которых разница распределений минимальна)
def get_next_step_coords(x, y, img_matrix, agent_matrix):
    neighbor_coords = get_neighbor_coords(x, y, max_x=img_matrix.shape[0], max_y=img_matrix.shape[1])

    # массив распределений в соседних точках оригинальной матрицы
    img_distributions = img_matrix[neighbor_coords[:, 0], neighbor_coords[:, 1]] / np.sum(img_matrix)

    # массив потенциальных распределений в соседних точках матрицы для агентов
    agent_potential_distributions = (1 + agent_matrix[neighbor_coords[:, 0], neighbor_coords[:, 1]]) / (1 + np.sum(agent_matrix))

    dist_diff = np.abs(img_distributions - agent_potential_distributions)

    return neighbor_coords[np.argmin(dist_diff)]

# обновление матрицы числа посещений агентами соответствующих ячеек
def update_agent_matrix(agent_coords, img_matrix, agent_matrix, num_iter):
    new_agent_matrix = np.copy(agent_matrix)
    
    # назначенные агентам индексы
    agent_idxs = np.arange(agent_coords.shape[0])

    # установка агентов в начальные координаты
    if np.all(new_agent_matrix == 0):
        new_agent_matrix[agent_coords[:, 0], agent_coords[:, 1]] = 1

    # число итераций перед усреднением матрицы агентов между процессами
    for _ in range(num_iter):
        # перемешивание индексов агентов
        shuffled_agent_idxs = np.random.permutation(agent_idxs)

        for agent_idx in shuffled_agent_idxs:
            # поиск новой координаты в зависимости от распределения
            new_x, new_y = get_next_step_coords(*agent_coords[agent_idx], img_matrix, new_agent_matrix)

            if new_agent_matrix[new_x, new_y] != 255:
                # изменение количества посещений агентами позиции
                new_agent_matrix[new_x, new_y] += 1

            # обновление текущих координат каждого агента
            agent_coords[agent_idx] = new_x, new_y

    return new_agent_matrix, agent_coords

# воосстановление исходного изображения с помощью агентов
def image_reconstruction(img_matrix, agent_num, num_proc, ssim_accuracy=None):
    # число итераций перед усреднением значений м ежду процессами
    num_iter = 2000

    agent_matrix = np.zeros_like(img_matrix, dtype=np.uint8)
    new_agent_matrix = agent_matrix

    # сходство между изображениями
    ssim = similarity(img_matrix, agent_matrix)
    new_ssim = ssim

    start_time = time.time()

    if num_proc == 1:
        while new_ssim >= ssim:
            ssim = new_ssim
            agent_matrix = new_agent_matrix

            if ssim_accuracy and ssim >= ssim_accuracy:
                break

            # случайные начальные координаты
            agent_coords = np.random.randint(low=0, high=100, size=(agent_num, 2))
            new_agent_matrix, agent_coords = update_agent_matrix(agent_coords, img_matrix, agent_matrix, num_iter)
            new_ssim = similarity(img_matrix, new_agent_matrix)

    else:
        # массив для хранения случайных начальных координат агентов всех процессов (при первом и последующих запусках)
        all_agent_coords = np.empty((num_proc, agent_num, 2), np.uint32)

        with mp.Pool(processes=num_proc) as pool:
            while new_ssim >= ssim:
                ssim = new_ssim
                agent_matrix = new_agent_matrix

                if ssim_accuracy and ssim >= ssim_accuracy:
                    break

                # генерация начальных координат агентов
                for i in range(num_proc):
                    all_agent_coords[i, :, 0] = np.random.randint(0, img_matrix.shape[0], size=agent_num)
                    all_agent_coords[i, :, 1] = np.random.randint(0, img_matrix.shape[1], size=agent_num)

                # аргументы целевой функции
                args = [(all_agent_coords[n], img_matrix, agent_matrix, num_iter) for n in range(num_proc)]

                results = pool.starmap_async(update_agent_matrix, args).get()
                result_agent_matrix_list = np.array([res[0] for res in results])

                # попадание агентов в те же координаты для соответствующих процессов
                agent_coords = np.array([res[1] for res in results])
        
                # усреднение матрицы
                new_agent_matrix = scaling(np.sum(result_agent_matrix_list, axis=0) / num_proc)
                new_ssim = similarity(img_matrix, new_agent_matrix)

            # закрытие пула
            pool.close()
            # ожидание завершения процессов
            pool.join()
        
    end_time = time.time()

    return agent_matrix, ssim, end_time - start_time

def plot_time_dependence(nums_processes, times):
    plt.title('Зависимость времени вычислений\nот количества выделенных процессов')
    plt.plot(nums_processes, times, label='Изменение времени по результатам эксперимента')

    # график при начальном времени (1 процесс) и постепенном разбиении на подпроцессы (эталон: гипербола)
    plt.plot(nums_processes, [times[0] / i for i in nums_processes], label='Эталонное изменение времени')
    plt.ylabel('Время выполнения в секундах')
    plt.xlabel('Количество выделенных процессов')
    plt.legend()
    plt.show()

def show_images(images, titles):
    num_images = len(images)

    # размер сетки
    cols = int(np.ceil(np.sqrt(num_images)))
    rows = int(np.ceil(num_images / cols))

    # параметры отображения
    _, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(images[i], cmap='gray')
            ax.set_title(titles[i])
        ax.axis('off') 

    plt.show()

def main():
    # загрузка изображения
    original_img = Image.open('image.jpg')
    gray_img = ImageOps.grayscale(original_img)

    img_matrix = np.asarray(gray_img, dtype=np.uint8)

    # список разного числа используемых процессов (зависит от числа доступных ЦПУ)
    nums_processes = [i + 1 for i in range(mp.cpu_count())]

    # список измерений времени для разного количества используемых процессов
    times = []

    # число агентов
    agent_num = int(img_matrix.size * 0.01)

    images = [img_matrix]
    titles = ['Оригинал']

    # искомая степень сходства изображений
    ssim_accuracy = None

    # эксперимент для разного числа процессов
    for num_proc in nums_processes:
        if num_proc == 1:
            agent_matrix, ssim, time = image_reconstruction(img_matrix, agent_num, num_proc)
            ssim_accuracy = ssim
        else:
            agent_matrix, ssim, time = image_reconstruction(img_matrix, agent_num, num_proc, ssim_accuracy)
        
        images.append(agent_matrix)
        titles.append(f'CPU: {num_proc}, схожесть: {ssim*100:.1f}%')
        times.append(time)

    show_images(images, titles)
    plot_time_dependence(nums_processes, times)

if __name__ == '__main__':
    main()
