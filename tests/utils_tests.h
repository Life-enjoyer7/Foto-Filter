#ifndef UTILS_TESTS_H
#define UTILS_TESTS_H

#include "../src/filter.h"

// Имена стратегий (индекс == strategy_id, который принимает pipeline_run).
extern const char *strategy_names[7];

// Тот же порядок, в котором pipeline_run (см. src/pipeline.c) строит свой
// внутренний filters[27] - нужен тестам 5/6, чтобы посчитать эталонный
// результат для конкретного filter_id независимо от pipeline.
extern Filter (*const filter_factories[27])(void);

// Тесты 1-4 сами по себе - перебор 6 конфигураций потоков x 7 стратегий
// (а TEST 3 - ещё и x 5 фильтров) на всех картинках из images/. Полный
// warm-up+repeat в духе task1/2 (там было 3-5) здесь утроил/учетверил бы и без
// того тяжёлый прогон, поэтому здесь ограничиваемся repeat=2 (1 прогрев + 2
// замера, берём медиану) - шум одиночного замера уже уходит, а make test
// остаётся вменяемым по времени.
#define PIPELINE_BENCH_REPEAT 2

// Параметры TEST 5 (см. testRandomizedProperties в tests.c).
#define MAX_TRIAL_IMAGES 5
#define RANDOM_TEST_TRIALS 30

// Генераторы случайных данных для TEST 5.
IplImage *createRandomImage(int w, int h);

// Часть прогонов - явные крайние случаи (1x1, 2x2, 3x3), остальные - случайный
// размер из широкого диапазона.
int randomImageDim(void);

// applyFilter и все параллельные стратегии (см. src/filter.c) вычисляют индекс
// как (coord - filterDim/2 + tap + dim) % dim; если filterDim/2 > dim, то
// однократного "+ dim" не хватает и получаем UB (см. находку в задачах 1/2).
// filter.c трогать нельзя, поэтому здесь просто не выбираем из каталога
// фильтр, чей half превышает наименьшее измерение картинки в этом прогоне.
int randomFilterIdForDim(int minDim);

// Эталонная свёртка через cv::filter2D (для сверки в TEST 6).
IplImage *referenceApplyFilter(const IplImage *src, const Filter *f);

// Сравнение с допуском (для TEST 6 - запас на порядок суммирования double).
int imagesApproxEqual(const IplImage *a, const IplImage *b, int tolerance);

#endif
