"""
Скрипт, который на этапе build создаёт Markdown-файлы
с директивой ::: для каждого модуля, найденного в пакете.
"""
from pathlib import Path
import pkgutil
import importlib
import mkdocs_gen_files

PACKAGE = "embeddings_squeeze"  # имя корневого пакета

def iter_modules(package_name: str):
    """Рекурсивно перечисляет все под-модули пакета."""
    pkg = importlib.import_module(package_name)
    yield pkg.__name__
    pkg_path = Path(pkg.__file__).parent
    for mod in pkgutil.walk_packages([str(pkg_path)], prefix=pkg.__name__ + "."):
        yield mod.name

# Главная страница API
with mkdocs_gen_files.open("api/README.md", "w") as nav_file:
    nav_file.write(f"# API Reference for `{PACKAGE}`\n\n")
    nav_file.write("::: " + PACKAGE + "\n")

# Генерация документации для каждого модуля
for module in iter_modules(PACKAGE):
    # Путь к .md файлу: api/models/quantizers.md
    doc_path = Path("api") / Path(*module.split("."))  # api/models/quantizers
    md_file = str(doc_path.with_suffix(".md"))  # преобразуем в строку

    # ✅ Создаём родительскую директорию вручную (было ensure_path)
    full_dir = Path("site") / md_file
    full_dir.parent.mkdir(parents=True, exist_ok=True)

    # Пишем сам Markdown-файл
    with mkdocs_gen_files.open(md_file, "w") as f:
        f.write(f"# `{module}`\n\n")
        f.write(f"::: {module}\n")

    # Добавляем в навигацию literate-nav
    edit_path = Path(module.replace(".", "/")).with_suffix(".py")
    mkdocs_gen_files.set_edit_path(md_file, edit_path)