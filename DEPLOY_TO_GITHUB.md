# Как выложить проект в GitHub и заменить старый репозиторий

Репозиторий: **https://github.com/2serenity/gaze_tracker_module**

Выполните команды **в терминале** (PowerShell или cmd) из папки проекта. Должен быть установлен [Git](https://git-scm.com/).

## 1. Откройте папку проекта

```bash
cd D:\profs\gaze_tracker_module
```

## 2. Инициализация репозитория и первый коммит

```bash
git init
git add .
git commit -m "Standalone gaze tracker: MediaPipe + Ridge, no EyeGestures"
git branch -M main
```

## 3. Подключите удалённый репозиторий

Если `origin` уже был добавлен ранее:

```bash
git remote set-url origin https://github.com/2serenity/gaze_tracker_module.git
```

Если подключаете впервые:

```bash
git remote add origin https://github.com/2serenity/gaze_tracker_module.git
```

## 4. Заменить содержимое репозитория на GitHub (удалить старые файлы и загрузить этот проект)

**Внимание:** команда ниже **полностью перезапишет** ветку `main` на GitHub текущим состоянием папки. Старые файлы (в т.ч. EyeGestures-main, старый README и т.д.) исчезнут.

```bash
git push -f origin main
```

При запросе авторизации используйте ваш GitHub-аккаунт (логин и пароль или [Personal Access Token](https://github.com/settings/tokens)).

---

После этого в репозитории будет только этот проект: `api/`, `core/`, `gui/`, `docs/`, `README.md`, `requirements.txt` и т.д. Папка EyeGestures-main и старое содержимое будут удалены.
