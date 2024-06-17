import matplotlib.pyplot as plt
from scipy import integrate
import numpy as np
import matplotlib.animation as animation


###############################ANIMACJA KULKI DLA OKRESU RZECZYWISTEGO I PRZYBLIŻONEGO##################################

g = 9.81
l = 1  # dlugosc (m)
teta0 = np.radians(45)  # kąt pocz (radians)
vstycz = 0  # vpocz (radians/s)
dt = 0.01  # odstepy czasu (s)

t_max = 10
t1 = np.arange(0, t_max, dt)


# funkcja rozwiazujaca rownanie rozniczkowe,
# potrzebna teraz i do drugiej części zadania
def rowDrugStopnia(y, t):
    # początkowo
    y, f = y
    dydt = [f, (-1) * np.sin(y) * g / l]
    return dydt



# z rownania rozniczkowego
y0 = [teta0, vstycz]
rozw = integrate.odeint(rowDrugStopnia, y0, t1)
theta_odeint = rozw[:, 0]

# przyblizenie malych kątów
omega = np.sqrt(g / l)
theta_small_angle = teta0 * np.cos(omega * t1)


# funkcja do obliczenia położenia chwilowego w zależności od wychylenia
def kordykuli(th):
    return l * np.sin(th), -l * np.cos(th)


# położenie poczatkowe
x0, y0 = kordykuli(teta0)

radius = 0.08
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'aspect': 'equal'})
# wykres z równania różniczkowego
line2, = ax.plot([0, x0], [0, y0], lw=3, c='k', label='Rzeczywiste')
kolo2 = ax.add_patch(plt.Circle(kordykuli(teta0), radius, fc='r', zorder=3))

# wykres dla małych kątów
line3, = ax.plot([0, x0], [0, y0], lw=3, c='b',alpha=0.5, label='Przybliżenie małych kątów')
kolo3 = ax.add_patch(plt.Circle(kordykuli(teta0), radius, fc='b',alpha=0.5, zorder=3))

ax.set_xlim(-l * 1.2, l * 1.2)
ax.set_ylim(-l * 1.2, l * 1.2)
ax.set_title('Animacja wahadła matematycznego')
ax.legend()


def animate(i):
    # animacja dla rzeczywistego
    x_harm, y_harm = kordykuli(theta_odeint[i])
    line2.set_data([0, x_harm], [0, y_harm])
    kolo2.set_center((x_harm, y_harm))

    # animacja dla przyblizonego
    x_small, y_small = kordykuli(theta_small_angle[i])
    line3.set_data([0, x_small], [0, y_small])
    kolo3.set_center((x_small, y_small))


nframes = len(theta_odeint)
interval = dt * 1000
ani = animation.FuncAnimation(fig, animate, frames=nframes, repeat=True, interval=interval)
plt.show()


################################ROZWIĄZANIE TRANSFORMATY FOURIERA I ZALEŻNOŚĆ T/T0######################################

g = 9.81
length = 0.3

pmk = 2*np.pi*np.sqrt(length/g)
#lista czestotliwosci z rozwiazanego rownania rozniczkowego
okresyRR = []
#i dla przybliżenia małych kątów
stosunek = []

# theta - kat wychylenia, t - czas

# aby rozwiazac rownanie rozniczkowe drugiego stopnia w pythonie
# trzeba podstawic ze f = d0/dt i potem df/dt mamy to jest wlasnie drugiego stopnia juz

# y - theta
def rowDrugStopnia(y, t):
    # początkowo
    y, f = y
    dydt = [f, (-1) * np.sin(y) * g / length]
    return dydt


# N - liczba próbek, ts - czas probkowania fs - czestotliwosc probkowania, fstep - krok probkowania ,
# tstep - co ile czasu próbka
N = 150000
ts = 150
fs = N/ts
tstep = 1 / fs
fstep = fs / N

t = np.linspace(0, ts, int(fs*ts))

# [-np.pi/2, 0] to warunki poczatkowe, najpierw dla samego theta i dla d0/dt
for n in np.arange(8, 1.9, -0.1):
    # range(50, 14, -1):

    print(n)
    rozw = integrate.odeint(rowDrugStopnia, [-np.pi / n, 0], t)

    rozw_transformata = np.fft.fft(rozw[:, 0])
    rozw_transformata = rozw_transformata[:len(rozw_transformata)//2]
    rozw_transformata_mag = np.abs(rozw_transformata)
    rf = np.fft.fftfreq(len(rozw[:, 0]), d=1/fs)[:len(rozw[:, 0])//2]

    f_loc = np.argmax(rozw_transformata_mag)
    okresyRR.append(1 / rf[f_loc])


print(len(okresyRR))
print(okresyRR)

for t in okresyRR:
    stosunek.append(t/pmk)

plt.plot(np.arange(8, 1.9, -0.1), stosunek)
plt.gca().invert_xaxis()
plt.gca().set_title("Zależność okresu drgań wahadła T od amplitudy drgań")
plt.gca().set_xlabel("Wychylenie początkowe pi/x")
plt.gca().set_ylabel("T0 / T")



plt.show()


