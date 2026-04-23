import os
import sys
import time

import pygame


DEADZONE = 0.15


def axis_value(v, deadzone=DEADZONE):
    return 0.0 if abs(v) < deadzone else v


def _event_ids(event):
    ids = []
    for attr_name in ("instance_id", "joy", "which"):
        attr_value = getattr(event, attr_name, None)
        if attr_value is not None and attr_value not in ids:
            ids.append(f"{attr_name}={attr_value}")
    return ", ".join(ids) if ids else "nessun-id"


def _init_first_joystick():
    if pygame.joystick.get_count() == 0:
        return None

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    return joystick


def _print_controller_info(joystick):
    if joystick is None:
        print("Controller non disponibile.")
        return

    try:
        instance_id = joystick.get_instance_id()
    except Exception:
        instance_id = None

    try:
        legacy_id = joystick.get_id()
    except Exception:
        legacy_id = None

    print("=" * 60)
    print("CONTROLLER RILEVATO")
    print(f"Nome       : {joystick.get_name()}")
    print(f"Assi       : {joystick.get_numaxes()}")
    print(f"Pulsanti   : {joystick.get_numbuttons()}")
    print(f"Hat/D-pad  : {joystick.get_numhats()}")
    print(f"instance_id: {instance_id}")
    print(f"legacy_id  : {legacy_id}")
    print("=" * 60)
    print("Premi i tasti per vedere il loro indice.")
    print("Muovi gli stick per vedere gli assi.")
    print("Chiudi la finestra oppure premi ESC per uscire.")
    print("Lo script stampa anche gli identificativi evento per diagnosticare mismatch pygame.")
    print("=" * 60)


def main():
    # Permette la ricezione degli eventi joystick anche quando la finestra perde il focus.
    os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")

    pygame.init()
    pygame.joystick.init()

    screen = pygame.display.set_mode((760, 220))
    pygame.display.set_caption("Test controller - pulsanti e assi")

    joystick = _init_first_joystick()
    if joystick is None:
        print("Nessun controller rilevato.")
        print("Collega il controller e rilancia lo script.")
        pygame.quit()
        return

    _print_controller_info(joystick)

    clock = pygame.time.Clock()
    last_axis_print = 0.0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

            elif event.type == pygame.JOYBUTTONDOWN:
                print(f"[BUTTON DOWN] indice={event.button} | {_event_ids(event)}")

            elif event.type == pygame.JOYBUTTONUP:
                print(f"[BUTTON UP  ] indice={event.button} | {_event_ids(event)}")

            elif event.type == pygame.JOYHATMOTION:
                print(f"[HAT] indice={event.hat} valore={event.value} | {_event_ids(event)}")

            elif event.type == pygame.JOYDEVICEADDED:
                print(
                    f"[INFO] Controller collegato: device_index={getattr(event, 'device_index', None)} | {_event_ids(event)}"
                )
                if joystick is None:
                    joystick = _init_first_joystick()
                    _print_controller_info(joystick)

            elif event.type == pygame.JOYDEVICEREMOVED:
                print(f"[INFO] Controller scollegato: {_event_ids(event)}")
                if joystick is not None:
                    try:
                        joystick.quit()
                    except Exception:
                        pass
                    joystick = None

        # Stampa periodica dello stato degli assi.
        # Se il controller viene scollegato durante il test, il ciclo continua senza
        # tentare letture su un oggetto joystick non più valido.
        now = time.time()
        if joystick is not None and now - last_axis_print > 0.2:
            axis_values = []
            try:
                num_axes = joystick.get_numaxes()
            except Exception:
                num_axes = 0

            for i in range(num_axes):
                try:
                    v = axis_value(joystick.get_axis(i))
                except Exception:
                    v = 0.0
                axis_values.append(f"asse {i}: {v:+.3f}")

            if axis_values:
                print("[AXES] " + " | ".join(axis_values))
            last_axis_print = now

        # Riempimento semplice della finestra
        screen.fill((30, 30, 30))
        pygame.display.flip()
        clock.tick(60)

    if joystick is not None:
        try:
            joystick.quit()
        except Exception:
            pass

    pygame.joystick.quit()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
