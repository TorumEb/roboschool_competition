# E2E Проверка (Sim + ROS2 + RViz)

Папка для полной проверки, что связка работает целиком:
- контейнер симуляции,
- контейнер ROS 2 Jazzy,
- sim-side контроллер,
- ROS bridge,
- камера в топиках и отображение в `rviz2`.

## Полный запуск

Из корня репозитория:

```bash
bash ros2_isaac_bridge/e2e_check/run_everything.sh
```

Скрипт делает всё сам:
1. поднимает `aliengo-competition`,
2. билдит/поднимает `ros2-jazzy`,
3. стартует `isaac_controller.py`,
4. стартует `bridge_node`,
5. проверяет, что приходят:
   - `/aliengo/camera/color/image_raw`
   - `/aliengo/camera/depth/image_raw`
6. открывает `rviz2` с профилем `camera_topics.rviz`.

Опции:
- `--skip-build` пропускает сборку образов и просто поднимает контейнеры.
- `--no-rviz` не открывает GUI, только проверяет, что камеры реально публикуются.

Пример smoke-check без GUI:

```bash
bash ros2_isaac_bridge/e2e_check/run_everything.sh --skip-build --no-rviz
```

На первом старте симулятора проверка может идти дольше (собирается `gymtorch`).

## Остановка

```bash
bash ros2_isaac_bridge/e2e_check/stop_everything.sh
```

Этот скрипт останавливает процессы bridge/controller и оба контейнера.

## Полезно

Если `rviz2` не открывается, обычно проблема в X11/`DISPLAY`.
В этом проекте доступ уже настраивается через `docker/ctl.sh` (`xhost` + проброс X11).
