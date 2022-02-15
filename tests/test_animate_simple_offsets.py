import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def main():
    numframes = 100
    numpoints = 10
    x = np.linspace(0, 1, numpoints)
    y = np.linspace(0, 1, numpoints)
    t = np.linspace(0, 1, numframes)

    fig, axs = plt.subplots(figsize=(8,3), ncols=2)
    scat0 = axs[0].scatter(x, y, c='k', s=100)
    scat1 = axs[1].scatter(y, x, c='k', s=100)
    txt = axs[0].text(0.03, 0.03, f't={t[0]:.3f}', transform=axs[0].transAxes,
                      ha='left',va='bottom', color='k', fontsize='x-small')

    plt.tight_layout()

    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(t, x, y, scat0, scat1, txt))
    ani.save('test_movie_offsets.mp4')


def update_plot(i, t, x, y, scat0, scat1, txt):

    scat0.set_offsets(np.c_[x+t[i], y+t[i]])
    scat1.set_offsets(np.c_[y-t[i], x-t[i]])
    txt.set_text(f't={t[i]:.3f}')

    return scat0, scat1, txt,

if __name__ == "__main__":
	main()
