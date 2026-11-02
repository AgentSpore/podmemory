# Порядок глав EPUB

## Статус
Реализован узкий обход primary spine. RED на настоящем синтетическом EPUB подтвердил manifest-порядок и включение лишних документов; GREEN и полный набор проверяются в evidence текущего цикла. Независимое production-ревью не выявило дефектов. PR7 (backup) завершён; не затрагивается.

## Проблема
extract_from_epub обещает reading order, но get_items_of_type обходит manifest. Книга с manifest C,A,B и spine A,B,C передаёт анализу главы C,A,B.

## Acceptance
- Реальный синтетический EPUB с разными manifest/spine: chapters и объединённый text следуют spine.
- Документы вне spine не добавляются к основному тексту; вспомогательные linear=no не входят в основной последовательный текст.
- Метаданные title/author, названия глав, word_count и существующие проверки пустого текста сохраняются.
- Не меняются API/UI, PDF/URL extraction, зависимости/lock или модельные вызовы.
- RED до исправления и GREEN после на настоящем ebooklib архиве; ordered/default-linear, reordered manifest, auxiliary/unreferenced документы и пустое содержание.
- Существующий полный Python-набор с синтетическими настройками; независимое ревью точного diff. Нет утверждения о реальном AI-анализе.

## Источники
https://www.w3.org/TR/epub-33/ — spine определяет default reading order; linear=no обозначает вспомогательное содержимое.
https://docs.sourcefabric.org/projects/ebooklib/en/latest/tutorial.html — чтение элементов по идентификатору.

## Коррекция P2: manifest fallback
Координатор воспроизвёл потерю поддерживаемой XHTML-главы, когда primary spine ссылается на foreign resource с fallback. Прежний кандидат и evidence сохраняются неизменными; коррекция в отдельном worktree.

Acceptance: цепочки fallback разрешаются до первого поддерживаемого XHTML-документа на месте исходного spine item; сохранены порядок и linear=no. Обход конечен, missing/self/cycle/некорректные ссылки пропускаются без зависания и без подстановки всех manifest-документов. Валидная книга не отвергается ради обхода регрессии. Реальные ZIP EPUB: extraction и HTTP upload, direct/multihop fallback и повреждённые цепочки. Используется тот же package document, который выбирает установленный EbookLib. Без новых зависимостей и UI.
Источник: https://www.w3.org/TR/epub-33/#sec-manifest-fallbacks
