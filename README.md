## Задание

**Презентация**
https://drive.google.com/file/d/1Qi_tAZqvXM02eB4fv8FBi5dWg9hV-wMM/view

**Задание:**
https://docs.google.com/document/d/1JWhVope32HIfrfZDNMNk7XSk83DBd7xh3ZMEcW0H8Bk/edit?tab=t.0

**Наш вариант:**
2 - min-max фильтр

## Теория

**Фильтры** 
Применение min-фильтра и max-фильтра к изображению приводит к интересному эффекту в обработке изображений.

1) Min-фильтр: Этот фильтр заменяет каждый пиксель изображением минимального значения из его соседних пикселей. Это обычно используется для удаления шума, особенно "соль" в изображениях, поскольку он помогает сгладить яркие пиксели, заменяя их более темными значениями.

2) Max-фильтр: Этот фильтр работает наоборот, заменяя каждый пиксель изображением максимального значения из его соседей. Max-фильтр часто применяется для удаления "перца" в изображениях, что помогает выделить яркие участки и сгладить темные шумы.

Когда вы сначала применяете min-фильтр, а затем max-фильтр, происходит следующее:
1) Сначала min-фильтр сглаживает изображение, уменьшая влияние ярких пикселей и оставляя более темные области.
2) Затем max-фильтр, применяемый к уже обработанному изображению, будет находить и усиливать яркие участки, которые остались после применения min-фильтра.
3) В результате такого последовательного применения фильтров изображение может стать более сглаженным с уменьшением шума, но также могут потеряться некоторые детали, поскольку min-фильтр "сглаживает" яркие детали, а max-фильтр "усиливает" уже сглаженные области. Это может привести к эффекту, где сохранены только крупные структуры изображения, а мелкие детали могут быть потеряны.

**Изображения**

PGM (Portable Gray Map) и PPM (Portable Pixmap) — это форматы файлов для хранения изображения, разработанные как часть семейства форматов "Portable Image Format" (PBM). Они используются для представления изображений в простом текстовом или бинарном виде. Вот основные характеристики каждого из них:

1) PGM (Portable Gray Map)

- PGM используется для хранения градаций серого. Каждый пиксель представлен значением яркости, обычно в диапазоне от 0 (черный) до 255 (белый).
- Структура файла:
    -   Заголовок, содержащий информацию о формате (например, `P2` для текстового формата или `P5` для бинарного).
    -   Размеры изображения (ширина и высота).
    -   Максимальное значение яркости (обычно 255).
    -   Данные пикселей (последовательно для каждого пикселя).
    
2) PPM (Portable Pixmap)

- PPM используется для хранения цветных изображений. Каждый пиксель представлен тремя значениями (красный, зеленый и синий), которые обычно находятся в диапазоне от 0 до 255.  
- Структура файла:
    -   Заголовок, аналогичный PGM, с указанием формата (например, `P3` для текстового формата или `P6` для бинарного).
    -   Размеры изображения (ширина и высота).
    -   Максимальное значение цвета (обычно 255).
    -   Данные пикселей (последовательно для каждого пикселя, начиная с красного, затем зеленого и синего).
    
## Код

1) Функции загрузки, cохранения, сравнения изображений (взяты готовые):

```cpp
bool loadImage(const std::string& filename, int& width, int& height, std::vector<unsigned char>& data, bool& isGrayscale) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) return false;

    std::string header;
    file >> header;
    if (header == "P6") {
        std::cout << "Loaded RGB image\n";
        isGrayscale = false;
    }
    else if (header == "P5") {
        std::cout << "Loaded greyscale image\n";
        isGrayscale = true;
    }
    else {
        return false;
    }

    file >> width >> height;
    int maxval;
    file >> maxval;
    file.get();

    int numChannels = isGrayscale ? 1 : 3;
    data.resize(width * height * numChannels);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    return true;
}
```
 ```bool& isGrayscale``` возвращает true, если изображение в градации серого (PGM), и false, если в цветном формате (PPM).
```cpp
bool saveImage(const std::string& filename, int width, int height, const std::vector<unsigned char>& data, bool isGrayscale) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    file << (isGrayscale ? "P5" : "P6") << "\n";
    file << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
    return true;
}

void compareImages(const std::vector<unsigned char>& img1, const std::vector<unsigned char>& img2) {
    bool match = true;
    for (size_t i = 0; i < img1.size(); ++i) {
        if (img1[i] != img2[i]) {
            match = false;
            std::cout << "Pixel is not equal[" << i << "]: img1 = " << static_cast<int>(img1[i]) << ", img2 = " << static_cast<int>(img2[i]) << "\n";
            break;
        }
    }
    if (match) {
        std::cout << "Images are equal.\n";
    }
    else {
        std::cout << "Images are not equal.\n";
    }
}
```
2) Функции min-max фильтра к цветному формату:
- Для CPU:
```cpp
void applyMinMaxFilterCpu(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height, bool isMinPass) {
    int channels = 3;
    for (int y = MASK_RADIUS; y < height - MASK_RADIUS; ++y) {
        for (int x = MASK_RADIUS; x < width - MASK_RADIUS; ++x) {
            for (int c = 0; c < channels; ++c) {
                int pixelIndex = (y * width + x) * channels + c;
                int result = isMinPass ? 255 : 0;
                for (int dy = -MASK_RADIUS; dy <= MASK_RADIUS; ++dy) {
                    for (int dx = -MASK_RADIUS; dx <= MASK_RADIUS; ++dx) {
                        int neighborIndex = ((y + dy) * width + (x + dx)) * channels + c;
                        int neighborPixel = input[neighborIndex];
                        result = isMinPass ? std::min(result, neighborPixel) : std::max(result, neighborPixel);
                    }
                }
                output[pixelIndex] = static_cast<unsigned char>(result);
            }
        }
    }
}
```
- Для GPU:
```cpp
__global__ void applyMinMaxFilterGpu(unsigned char* input, unsigned char* output, int width, int height, bool isMinPass) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channels = 3;

    if (x >= MASK_RADIUS && x < width - MASK_RADIUS && y >= MASK_RADIUS && y < height - MASK_RADIUS) {
        for (int c = 0; c < channels; ++c) {
            int pixelIndex = (y * width + x) * channels + c;
            int result = isMinPass ? 255 : 0;
            for (int dy = -MASK_RADIUS; dy <= MASK_RADIUS; ++dy) {
                for (int dx = -MASK_RADIUS; dx <= MASK_RADIUS; ++dx) {
                    int neighborIndex = ((y + dy) * width + (x + dx)) * channels + c;
                    int neighborPixel = input[neighborIndex];
                    result = isMinPass ? min(result, neighborPixel) : max(result, neighborPixel);
                }
            }
            output[pixelIndex] = static_cast<unsigned char>(result);
        }
    }
}
```
3) Функции min-max фильтра к формату в градации серого:
- Для CPU:
```cpp
void applyMinMaxFilterGsCpu(const std::vector<unsigned char>& input, std::vector<unsigned char>& output, int width, int height, bool isMinPass) {
    for (int y = MASK_RADIUS; y < height - MASK_RADIUS; ++y) {
        for (int x = MASK_RADIUS; x < width - MASK_RADIUS; ++x) {
            int pixelIndex = y * width + x;
            int result = isMinPass ? 255 : 0;
            for (int dy = -MASK_RADIUS; dy <= MASK_RADIUS; ++dy) {
                for (int dx = -MASK_RADIUS; dx <= MASK_RADIUS; ++dx) {
                    int neighborIndex = (y + dy) * width + (x + dx);
                    int neighborPixel = input[neighborIndex];
                    result = isMinPass ? std::min(result, neighborPixel) : std::max(result, neighborPixel);
                }
            }
            output[pixelIndex] = static_cast<unsigned char>(result);
        }
    }
}
```
- Для GPU:
```cpp
__global__ void applyMinMaxFilterGsGpu(unsigned char* input, unsigned char* output, int width, int height, bool isMinPass) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= MASK_RADIUS && x < width - MASK_RADIUS && y >= MASK_RADIUS && y < height - MASK_RADIUS) {
        int pixelIndex = y * width + x;
        int result = isMinPass ? 255 : 0;
        for (int dy = -MASK_RADIUS; dy <= MASK_RADIUS; ++dy) {
            for (int dx = -MASK_RADIUS; dx <= MASK_RADIUS; ++dx) {
                int neighborIndex = (y + dy) * width + (x + dx);
                int neighborPixel = input[neighborIndex];
                result = isMinPass ? min(result, neighborPixel) : max(result, neighborPixel);
            }
        }
        output[pixelIndex] = static_cast<unsigned char>(result);
    }
}
```

**Отличия**
По факту отличие реализации для RGB и GS форматов в том, что в RGB мы учитываем 3 канала, а в GS нет.

**Реализация**
В лабе рассказывается, как работать с граничными условиями. Тут мы используем только те точки, которые находятся минимум на расстоянии радиуса маски от края изображения. То есть у нашего изображения остаются точки, к которым не применялся фильтр.

  
