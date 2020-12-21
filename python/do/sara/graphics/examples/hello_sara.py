import do.sara as sara


def user_main():
    w = sara.create_window(800, 600)

    for y in range(600):
        for x in range(800):
            sara.draw_point(x, y, (255, 0, 0))


if __name__ == '__main__':
    sara.run_graphics(user_main)
